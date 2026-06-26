# 🔊 Performance Analysis and Enhancement of DeepSC

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Enhancing the DeepSC semantic communication model using advanced deep learning techniques to improve information encoding, transmission, and reconstruction over noisy channels.

---

## 📌 What is DeepSC?

**DeepSC** (Deep learning-based Semantic Communication) is a system that transmits the *meaning* of a message rather than raw bits. Unlike traditional communication systems that focus on perfect bit recovery, DeepSC uses neural networks to encode and decode the semantic content of text — making it far more robust under low signal-to-noise ratio (SNR) conditions.

This project analyses the baseline DeepSC model and proposes enhancements to improve its performance across varying channel conditions.

---

## 🏗️ Architecture Overview

```
Input Text
    │
    ▼
Semantic Encoder (Transformer-based)
    │
    ▼
Channel Encoder  ──►  Noisy Channel (AWGN)  ──►  Channel Decoder
                                                        │
                                                        ▼
                                               Semantic Decoder
                                                        │
                                                        ▼
                                               Reconstructed Text
```

---



---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Install dependencies

```bash
git clone https://github.com/varshawali/Performance-Analysis-and-Enhancement-of-DeepSC.git
cd Performance-Analysis-and-Enhancement-of-DeepSC
pip install -r requirements.txt
```

### Run training

```bash
python train.py --snr 10 --epochs 50 --batch_size 64
```

### Run evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --snr 10
```

---

## 📁 Project Structure

```
├── data/               # Dataset preprocessing scripts
├── models/             # Semantic encoder/decoder & channel models
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── requirements.txt    # Dependencies
└── README.md
```

---

## 🔖 Topics

`deep-learning` `semantic-communication` `pytorch` `nlp` `transformer` `channel-coding` `python`

---

## 📄 Reference

Based on the original DeepSC paper:
> Xie, H., Qin, Z., Li, G. Y., & Juang, B. H. (2021). Deep learning enabled semantic communication systems. *IEEE Transactions on Signal Processing*.

---

## 📬 Contact

**Varsha Wali** — [LinkedIn](https://www.linkedin.com/in/varshawali) · [GitHub](https://github.com/varshawali)
