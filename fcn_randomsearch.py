#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
# Train & test a 1-D Fully-Convolutional Network (FCN) on “resistance_values.csv”
#     – mirrors convtran_randomsearch.py so that results are comparable –
#     – PyTorch implementation, random-search over a few key hyper-parameters –
# ──────────────────────────────────────────────────────────────────────────────
import os
import argparse
import logging
import random
from collections import defaultdict
from datetime import datetime
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ───────────────────────────── CLI ───────────────────────────────────────────
parser = argparse.ArgumentParser()
# dataset / I-O
parser.add_argument('--data_path',   default='resistance_values.csv',
                    help='Path to resistance_values.csv')
parser.add_argument('--output_dir',  default='Results_FCN',
                    help='Directory where models / plots are written (will be created)')
# system
parser.add_argument('--device',      choices=['cuda', 'cpu'], default='cuda',
                    help='Force a device (falls back to cpu if CUDA unavailable)')
# training
parser.add_argument('--sequence_len', type=int, default=10,
                    help='Number of consecutive readings per sample (must match ConvTran run)')
parser.add_argument('--search_trials', type=int, default=10,
                    help='Random-search budget (how many hyper-parameter configs)')
# reproducibility
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

# ───────────────────────── Reproducibility ───────────────────────────────────
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# ───────────────────────── Logging helper ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger()

# ───────────────────────── 1. Load CSV & sanity-check ────────────────────────
raw = pd.read_csv(args.data_path)
required_cols = {'Resistance Gassensor', 'label'}
if not required_cols.issubset(raw.columns):
    raise ValueError(f"CSV must contain columns {required_cols}, found {set(raw.columns)}")

values = raw['Resistance Gassensor'].values.astype('float32')
labels = raw['label'].values.astype('int64')

# ───────────────────────── 2. Slice into non-overlapping sequences ───────────
L = args.sequence_len
features, seq_labels = [], []
for start in range(0, len(values) - L + 1, L):   # step == L  → non-overlap
    end = start + L
    features.append(values[start:end])
    seq_labels.append(labels[end - 1])           # **label = last reading** (same as ConvTran)

features   = np.stack(features)                  # (N, L)
seq_labels = np.array(seq_labels)

unique_labels = np.sort(np.unique(seq_labels))
label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
y = np.array([label2idx[lbl] for lbl in seq_labels], dtype=np.int64)

# ───────────────────────── 3. Train / val / test split (80/10/10) ────────────
X_train, X_tmp, y_train, y_tmp = train_test_split(
    features, y, test_size=0.20, random_state=args.seed, stratify=y)

X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=args.seed, stratify=y_tmp)

# ───────────────────────── 4. Standardisation ────────────────────────────────
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train).astype('float32')
X_val_std   = scaler.transform(X_val).astype('float32')
X_test_std  = scaler.transform(X_test).astype('float32')

# tensors / datasets  → shape (N, 1, L) for Conv1d
train_ds = TensorDataset(torch.from_numpy(X_train_std[:, None, :]),
                         torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(X_val_std[:, None, :]),
                         torch.from_numpy(y_val))
test_ds  = TensorDataset(torch.from_numpy(X_test_std[:, None, :]),
                         torch.from_numpy(y_test))

# ───────────────────────── 5. FCN building block ─────────────────────────────
class FCN1D(nn.Module):
    """
    Very small FCN for time-series classification
      – Conv1d-BN-ReLU ×3
      – Global average pooling
      – Linear soft-max
    """
    def __init__(self, in_channels: int, num_classes: int,
                 C1: int = 128, C2: int = 256, C3: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, C1, kernel_size=8, padding='same'),
            nn.BatchNorm1d(C1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(C1, C2, kernel_size=5, padding='same'),
            nn.BatchNorm1d(C2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(C2, C3, kernel_size=3, padding='same'),
            nn.BatchNorm1d(C3),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(C3, num_classes)

    def forward(self, x):               # x: (B, C_in=1, L)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.mean(dim=2)               # global average pooling
        x = self.dropout(x)
        return self.head(x)

# ───────────────────────── 6. Hyper-parameter search space ───────────────────
HSPACE = {
    'C1':         [64, 128],
    'C2':         [128, 256],
    'C3':         [64, 128],
    'epochs':     [200, 500, 1000],
    'lr':         [1e-3, 1e-4, 1e-5],
    'batch_size': [16, 32, 64],
    'dropout':    [0.0, 0.2]
}

DEVICE  = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available())
                       else 'cpu')
PIN_MEM = DEVICE.type == 'cuda'

criterion = nn.CrossEntropyLoss()

def build_loader(ds, bs, shuffle):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      pin_memory=PIN_MEM, num_workers=4)

best_cfg, best_acc, best_state = None, -1.0, None

log.info(f'Random-search for {args.search_trials} configs …')
for trial in range(args.search_trials):
    cfg = {k: random.choice(v) for k, v in HSPACE.items()}

    model = FCN1D(1, len(unique_labels),
                  C1=cfg['C1'], C2=cfg['C2'], C3=cfg['C3'],
                  dropout=cfg['dropout']).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    train_loader = build_loader(train_ds, cfg['batch_size'], shuffle=True)
    val_loader   = build_loader(val_ds,   cfg['batch_size'], shuffle=False)

    # ── training loop
    for epoch in range(cfg['epochs']):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

    # ── validation accuracy
    model.eval()
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb).argmax(dim=1)
            n_correct += (preds == yb).sum().item()
            n_total   += yb.size(0)
    val_acc = n_correct / n_total

    log.info(f'Trial {trial+1:02d}/{args.search_trials} – cfg={cfg} – '
             f'val_acc={val_acc:0.4f}')

    if val_acc > best_acc:
        best_acc   = val_acc
        best_cfg   = cfg
        best_state = copy.deepcopy(model.state_dict())

log.info('──────────────── BEST CONFIG ─────────────────────')
log.info(f'{best_cfg}\nValidation accuracy = {best_acc:0.4f}')

# ───────────────────────── 7. Final training & test evaluation ───────────────
train_loader = build_loader(train_ds, best_cfg['batch_size'], shuffle=True)
val_loader   = build_loader(val_ds,   best_cfg['batch_size'], shuffle=False)
test_loader  = build_loader(test_ds,  best_cfg['batch_size'], shuffle=False)

model = FCN1D(1, len(unique_labels),
              C1=best_cfg['C1'], C2=best_cfg['C2'], C3=best_cfg['C3'],
              dropout=best_cfg['dropout']).to(DEVICE)
model.load_state_dict(best_state)
opt = torch.optim.Adam(model.parameters(), lr=best_cfg['lr'])

history = defaultdict(list)
for epoch in range(best_cfg['epochs']):
    # training
    model.train()
    running_loss, running_correct, running_total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        opt.step()

        running_loss    += loss.item() * yb.size(0)
        running_correct += (logits.argmax(dim=1) == yb).sum().item()
        running_total   += yb.size(0)

    train_loss = running_loss / running_total
    train_acc  = running_correct / running_total

    # validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss   = criterion(logits, yb)

            val_loss  += loss.item() * yb.size(0)
            val_correct += (logits.argmax(dim=1) == yb).sum().item()
            val_total   += yb.size(0)

    val_loss /= val_total
    val_acc   = val_correct / val_total

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

# ── final test metrics
model.eval()
all_preds, all_tgts = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb).argmax(dim=1)
        all_preds.append(preds.cpu())
        all_tgts.append(yb.cpu())

all_preds = torch.cat(all_preds).numpy()
all_tgts  = torch.cat(all_tgts).numpy()

conf_mat  = confusion_matrix(all_tgts, all_preds)
test_acc  = (all_preds == all_tgts).mean()
test_f1   = f1_score(all_tgts, all_preds, average='macro')
test_prec = precision_score(all_tgts, all_preds, average='macro')

log.info('──────────────── TEST METRICS ───────────────────')
log.info(f'Accuracy : {test_acc:0.4f}')
log.info(f'Macro F1 : {test_f1:0.4f}')
log.info(f'Macro Pr : {test_prec:0.4f}')

# ───────────────────────── 8. I-O – save artefacts ───────────────────────────
os.makedirs(args.output_dir, exist_ok=True)
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

torch.save(model.state_dict(),
           os.path.join(args.output_dir, f'fcn_best_{stamp}.pth'))
np.save(os.path.join(args.output_dir, 'confusion_matrix.npy'), conf_mat)
np.save(os.path.join(args.output_dir, 'label_mapping.npy'), unique_labels)
# persist the scaler so you can replicate inference-time preprocessing
import pickle
with open(os.path.join(args.output_dir, 'feature_scaler.pkl'), 'wb') as fh:
    pickle.dump(scaler, fh)

# ── plots
def _plot_history(metric, ylabel):
    plt.figure(figsize=(8, 6))
    plt.plot(history[f'train_{metric}'], label='train')
    plt.plot(history[f'val_{metric}'],   label='val')
    plt.xlabel('epoch'); plt.ylabel(ylabel); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,
                f'{metric}_curve_{stamp}.png'))
    plt.close()

_plot_history('loss', 'Cross-entropy loss')
_plot_history('acc',  'Accuracy')

plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, f'confusion_matrix_{stamp}.png'))
plt.close()

log.info(f'All artefacts written to:  {os.path.abspath(args.output_dir)}')