import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=20, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Seed for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Validation function
def validate(epoch, args, net, pad_idx, criterion):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total_loss = 0
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            loss = val_step(net, sents, sents, 0.1, pad_idx, criterion, args.channel)
            total_loss += loss
            pbar.set_description(f'Epoch: {epoch + 1}; Type: VAL; Loss: {loss:.5f}')
    return total_loss / len(test_iterator)

# Training function
def train(epoch, args, net, pad_idx, criterion, optimizer, mi_net=None, mi_opt=None):
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)
    total_loss = 0
    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))

    for sents in pbar:
        sents = sents.to(device)
        optimizer.zero_grad()

        if mi_net is not None:
            mi = train_mi(net, mi_net, sents, 0.1, pad_idx, mi_opt, args.channel)
            loss = train_step(net, sents, sents, 0.1, pad_idx, optimizer, criterion, args.channel, mi_net)
            pbar.set_description(f'Epoch: {epoch + 1};  Type: Train; Loss: {loss:.5f}; MI {mi:.5f}')
        else:
            loss = train_step(net, sents, sents, noise_std[0], pad_idx, optimizer, criterion, args.channel)
            pbar.set_description(f'Epoch: {epoch + 1};  Type: Train; Loss: {loss:.5f}')

        # Simply add the loss (no .item() needed)
        total_loss += loss  # Assume train_step returns a float

    return total_loss / len(train_iterator)

# Function to plot training and validation losses
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    args.vocab_file = '/content/drive/MyDrive/DeepSC-master/data/' + args.vocab_file  # Correct path

    """ Prepare the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ Define optimizer and loss function """
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab, args.d_model, args.num_heads, args.dff, 0.1).to(device)
    mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(deepsc.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    
    initNetParams(deepsc)

    # Initialize lists to store training and validation losses
    train_losses = []
    val_losses = []

    record_acc = 10  # To store the best validation accuracy

    for epoch in range(args.epochs):
        start = time.time()

        # Training and validation
        train_loss = train(epoch, args, deepsc, pad_idx, criterion, optimizer, mi_net, mi_opt)
        val_loss = validate(epoch, args, deepsc, pad_idx, criterion)

        # Append losses for plotting later
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save model if validation accuracy improves
        if val_loss < record_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            torch.save(deepsc.state_dict(), f'{args.checkpoint_path}/checkpoint_{str(epoch + 1).zfill(2)}.pth')
            record_acc = val_loss

    # Plot training and validation losses
    plot_losses(train_losses, val_losses)
