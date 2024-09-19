import os
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

# Define the device for computation (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='europarl/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--bert-config-path', default='bert/cased_L-12_H-768_A-12/bert_config.json', type=str)
parser.add_argument('--bert-checkpoint-path', default='bert/cased_L-12_H-768_A-12/bert_model.ckpt', type=str)
parser.add_argument('--bert-dict-path', default='bert/cased_L-12_H-768_A-12/vocab.txt', type=str)

# Parse arguments
args = parser.parse_args()

# Similarity Class Using BERT for Semantic Similarity
class Similarity:
    def __init__(self, model_name='bert-base-uncased'):
        # Load pre-trained BERT model and tokenizer from Hugging Face
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)

    def compute_similarity(self, real_sentences, predicted_sentences):
        # Encode sentences using BERT tokenizer
        real_embeddings = self._get_embeddings(real_sentences)
        predicted_embeddings = self._get_embeddings(predicted_sentences)

        # Compute cosine similarity between real and predicted embeddings
        similarity_scores = self._cosine_similarity(real_embeddings, predicted_embeddings)
        return similarity_scores

    def _get_embeddings(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Get the embeddings from the [CLS] token
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

    def _cosine_similarity(self, emb1, emb2):
        # Normalize embeddings
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)

        # Compute cosine similarity
        similarity = torch.mm(emb1, emb2.T)
        similarity_scores = torch.diag(similarity)  # Get diagonal elements for pairwise similarity
        return similarity_scores.cpu().numpy().tolist()

# Performance evaluation function based on BERT similarity
def performance(args, SNR, net):
    similarity = Similarity('bert-base-uncased')  # Load BERT for similarity computation

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    StoT = SeqtoText(token_to_idx, end_idx)
    sim_score = []  # Track similarity scores
    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for sents in test_iterator:
                    sents = sents.to(device)
                    target = sents

                    out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel)

                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                Tx_word.append(word)
                Rx_word.append(target_word)

            sim_score_epoch = []
            for sent1, sent2 in zip(Tx_word, Rx_word):
                # Compute BERT similarity
                sim_score_epoch.append(similarity.compute_similarity(sent1, sent2)) 

            # Append similarity scores
            if len(sim_score_epoch) > 0:
                sim_score_epoch = np.mean(sim_score_epoch)
            else:
                sim_score_epoch = 0

            sim_score.append(sim_score_epoch)

    score2 = np.mean(np.array(sim_score), axis=0)

    return score2  # Return similarity scores

if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0, 3, 6, 9, 12, 15, 18]

    args.vocab_file = '/content/drive/MyDrive/DeepSC/data/' + args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    # Define the DeepSC model and move it to the correct device
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab, args.d_model, args.num_heads, args.dff, 0.1).to(device)

    # Load model from checkpoint
    model_paths = []
    for fn in os.listdir(args.checkpoint_path):
        if not fn.endswith('.pth'):
            continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])  # Extract the index of the model
        model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

    model_paths.sort(key=lambda x: x[1])  # Sort by index
    model_path, _ = model_paths[-1]

    # Load the model checkpoint
    checkpoint = torch.load(model_path)
    deepsc.load_state_dict(checkpoint)
    print('Model loaded!')

    # Run performance evaluation
    sim_score = performance(args, SNR, deepsc)
    print(f"Similarity Score: {sim_score}")
