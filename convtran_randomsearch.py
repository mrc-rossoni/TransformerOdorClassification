#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
# Train & test ConvTran on the “resistance_values.csv” dataset
# (corrected version – handles Lazy layers when re‑loading weights)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# 0. Imports & CLI
# ──────────────────────────────────────────────────────────────────────────────
import os
import argparse
import logging
import random
from datetime import datetime
from collections import defaultdict
import copy  # ① new

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

# ── project‑local imports (unchanged, already on PYTHONPATH) ───────────────
from utils import Setup, Initialization, Data_Verifier               # noqa: F401  (left untouched)
from Models.model import model_factory, count_parameters             # noqa: F401
from Models.optimizers import get_optimizer
from Models.loss import get_loss_module
from Training import SupervisedTrainer

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
# dataset / I/O
parser.add_argument('--data_path',   default='resistance_values.csv',
                    help='Path to resistance_values.csv')
parser.add_argument('--output_dir',  default='Results',
                    help='Directory where models / plots are written (will be created)')
# system
parser.add_argument('--device',      choices=['cuda', 'cpu'], default='cuda',
                    help='Force a device (falls back to cpu if CUDA unavailable)')
# training
parser.add_argument('--sequence_len', type=int, default=10,
                    help='Number of consecutive readings per sample')
parser.add_argument('--search_trials', type=int, default=10,
                    help='Random‑search budget (how many hyper‑parameter configs)')
# reproducibility
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# 1. Reproducibility helpers
# ──────────────────────────────────────────────────────────────────────────────
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Load & inspect raw CSV
# ──────────────────────────────────────────────────────────────────────────────
raw = pd.read_csv(args.data_path)
required_cols = {'Resistance Gassensor', 'label'}
if not required_cols.issubset(raw.columns):
    raise ValueError(f"CSV must contain columns {required_cols}, found {set(raw.columns)}")

values = raw['Resistance Gassensor'].values.astype('float32')
labels = raw['label'].values.astype('int64')

# ──────────────────────────────────────────────────────────────────────────────
# 3. Slice into *non‑overlapping* sequences of length L
# ──────────────────────────────────────────────────────────────────────────────
L = args.sequence_len
features, seq_labels = [], []
for start in range(0, len(values) - L + 1, L):           # step = L → non‑overlap
    end = start + L
    features.append(values[start:end])
    seq_labels.append(labels[end - 1])                   # use last row’s label

features = np.stack(features)                            # (N, L)
seq_labels = np.array(seq_labels)

# label → index mapping (store so that inference can invert it)
unique_labels = np.sort(np.unique(seq_labels))
label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
y = np.array([label2idx[lbl] for lbl in seq_labels], dtype=np.int64)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Train / val / test split  (80 / 10 / 10)  (stratified)
# ──────────────────────────────────────────────────────────────────────────────
X_train, X_tmp, y_train, y_tmp = train_test_split(
    features, y, test_size=0.20, random_state=args.seed, stratify=y)

X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=args.seed, stratify=y_tmp)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Standardise (fit on *train* only, apply everywhere)
# ──────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train).astype('float32')
X_val_scaled   = scaler.transform(X_val).astype('float32')
X_test_scaled  = scaler.transform(X_test).astype('float32')

# tensors / datasets
train_ds = TensorDataset(torch.from_numpy(X_train_scaled),
                         torch.from_numpy(y_train.astype('int64')))
val_ds   = TensorDataset(torch.from_numpy(X_val_scaled),
                         torch.from_numpy(y_val.astype('int64')))
test_ds  = TensorDataset(torch.from_numpy(X_test_scaled),
                         torch.from_numpy(y_test.astype('int64')))

# ──────────────────────────────────────────────────────────────────────────────
# 6. Hyper‑parameter random search
# ──────────────────────────────────────────────────────────────────────────────
HSPACE = {
    'emb_size':      [32, 64, 128],
    'num_heads':     [2, 4, 8],
    'dim_ff':        [64, 128, 256],
    'epochs':        [200, 500, 1000],
    'lr':            [1e-3, 1e-4, 1e-5],
    'batch_size':    [16, 32, 64],
    'dropout':       [0.0, 0.1, 0.2],
}

DEVICE = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available())
                      else 'cpu')
PIN_MEM = DEVICE.type == 'cuda'

best_cfg, best_acc, best_model_state = None, -1.0, None

print(f"Running random search with {args.search_trials} trials …")
for trial in tqdm(range(args.search_trials), unit='cfg'):
    # ── sample until a valid (emb_size % num_heads == 0) combo
    while True:
        cfg = {k: random.choice(v) for k, v in HSPACE.items()}
        if cfg['emb_size'] % cfg['num_heads'] == 0:
            break

    cfg.update({
        'Net_Type':       'C-T',
        'Data_shape':     [1, 1, L],       # <C, L>
        'num_labels':     len(unique_labels),
        'Fix_pos_encode': 'tAPE',
        'Rel_pos_encode': 'eRPE',
    })

    # ── build objects
    model        = model_factory(cfg).to(DEVICE)
    optimizer    = get_optimizer("Adam")(model.parameters(), lr=cfg['lr'])
    loss_module  = get_loss_module()
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True,  pin_memory=PIN_MEM, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'],
                              shuffle=False, pin_memory=PIN_MEM, num_workers=4)

    trainer  = SupervisedTrainer(model, train_loader, DEVICE, loss_module, optimizer)
    val_eval = SupervisedTrainer(model, val_loader, DEVICE, loss_module)

    # ── training loop
    for epoch in range(cfg['epochs']):
        trainer.train_epoch(epoch)
    val_metrics, _ = val_eval.evaluate()

    if val_metrics.get('accuracy', 0.0) > best_acc:
        best_acc = val_metrics['accuracy']
        best_cfg = cfg
        # ② deep copy of *current* weights
        best_model_state = copy.deepcopy(model.state_dict())

print("\n──────────────── BEST CONFIG ────────────────")
print(best_cfg, f"\nValidation accuracy = {best_acc:0.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Final training of the best configuration and test evaluation
# ──────────────────────────────────────────────────────────────────────────────
train_loader = DataLoader(train_ds, batch_size=best_cfg['batch_size'],
                          shuffle=True,  pin_memory=PIN_MEM, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=best_cfg['batch_size'],
                          shuffle=False, pin_memory=PIN_MEM, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=best_cfg['batch_size'],
                          shuffle=False, pin_memory=PIN_MEM, num_workers=4)

model       = model_factory(best_cfg).to(DEVICE)

# ③ warm‑up forward pass – materialise Lazy layers before loading weights
seq_len = best_cfg['Data_shape'][2]          # 10 in your run
dummy    = torch.zeros(1, seq_len, device=DEVICE)  # shape (1, 10)
with torch.no_grad():
    model(dummy)                             # instantiates Lazy layers

model.load_state_dict(best_model_state)  # strict=True by default – now safe

optimizer   = get_optimizer("Adam")(model.parameters(), lr=best_cfg['lr'])
loss_module = get_loss_module()

trainer = SupervisedTrainer(model, train_loader, DEVICE, loss_module, optimizer,
                            print_conf_mat=False)
val_eval = SupervisedTrainer(model, val_loader, DEVICE, loss_module)

history = defaultdict(list)
for epoch in range(best_cfg['epochs']):
    train_loss, train_acc = trainer.train_epoch(epoch)      # ← unpack numbers
    vl_metrics, _ = val_eval.evaluate()                     # ← dict, as before

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(vl_metrics['loss'])
    history['val_acc'].append(vl_metrics['accuracy'])

# ── final test evaluation
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

print("\n──────────────── TEST METRICS ───────────────")
print(f"Accuracy : {test_acc:0.4f}")
print(f"Macro F1 : {test_f1:0.4f}")
print(f"Macro Pr : {test_prec:0.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 8. I/O – save artefacts
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(args.output_dir, exist_ok=True)

stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# model & helpers
torch.save(model.state_dict(),
           os.path.join(args.output_dir, f'convtran_best_{stamp}.pth'))

with open(os.path.join(args.output_dir, 'feature_scaler.pkl'), 'wb') as fh:
    pickle.dump(scaler, fh)
with open(os.path.join(args.output_dir, 'label_mapping.pkl'), 'wb') as fh:
    pickle.dump(label2idx, fh)
with open(os.path.join(args.output_dir, 'best_config.txt'), 'w') as fh:
    fh.write(repr(best_cfg) + '\n')

# plots

def _plot_history(metric_key, ylabel):
    plt.figure(figsize=(8, 6))
    plt.plot(history[f'train_{metric_key}'], label='train')
    plt.plot(history[f'val_{metric_key}'],   label='val')
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,
                f'{metric_key}_curve_{stamp}.png'))
    plt.close()

_plot_history('loss', 'Cross‑entropy loss')
_plot_history('acc',  'Accuracy')

plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, f'confusion_matrix_{stamp}.png'))
plt.close()

print(f"\nAll artefacts written to:  {os.path.abspath(args.output_dir)}")
