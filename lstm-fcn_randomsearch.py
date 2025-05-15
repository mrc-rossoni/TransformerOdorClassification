#!/usr/bin/env python3
"""
Train & evaluate an LSTM‑FCN model on the “resistance_values.csv” dataset
using the **same pre‑processing pipeline, splits and metrics** as the ConvTran
script.  A small random‑search over hyper‑parameters is used to select the best
configuration, after which the model is re‑trained and tested.  All artefacts
(model, scaler, label mapping, plots, etc.) are written under `--output_dir` so
that results are directly comparable to the ConvTran run.

Usage (default values reproduce the ConvTran setup):
    python lstmfcn_randomsearch.py \
        --data_path resistance_values.csv \
        --output_dir Results_LSTMFCN \
        --device gpu \
        --sequence_len 10 \
        --search_trials 10 \
        --seed 42
"""

# ─────────────────────────── 0. Imports & CLI ───────────────────────────────
import os
import argparse
import random
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, f1_score, precision_score)

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Input, LSTM, Dropout, Permute, Conv1D,
                                     BatchNormalization, Activation,
                                     GlobalAveragePooling1D, concatenate,
                                     Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',   default='resistance_values.csv',
                    help='Path to resistance_values.csv')
parser.add_argument('--output_dir',  default='Results_LSTMFCN',
                    help='Directory where models / plots are written')
parser.add_argument('--device',      choices=['gpu', 'cpu'], default='gpu',
                    help='Force CPU or (first) GPU')
parser.add_argument('--sequence_len', type=int, default=10,
                    help='Number of consecutive readings per sample')
parser.add_argument('--search_trials', type=int, default=10,
                    help='Random‑search budget (how many hyper‑parameter configs)')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# ─────────────────────── 1. Reproducibility & device ────────────────────────
K.clear_session()

# Make execution deterministic (to the extent possible)
np.random.seed(args.seed)
random.seed(args.seed)
tf.random.set_seed(args.seed)

if args.device == 'cpu':
    # Disable all GPUs
    tf.config.set_visible_devices([], 'GPU')
else:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass  # happens if GPUs have already been initialised

# ────────────────────────── 2. Load & inspect CSV ───────────────────────────
raw = pd.read_csv(args.data_path)
required_cols = {'Resistance Gassensor', 'label'}
if not required_cols.issubset(raw.columns):
    raise ValueError(f"CSV must contain columns {required_cols}, found {set(raw.columns)}")

values = raw['Resistance Gassensor'].values.astype('float32')
labels = raw['label'].values.astype('int64')

# ───────────── 3. Slice into *non‑overlapping* sequences of length L ─────────
L = args.sequence_len
features, seq_labels = [], []
for start in range(0, len(values) - L + 1, L):
    end = start + L
    features.append(values[start:end])
    seq_labels.append(labels[end - 1])  # label of the last row in the window

features = np.stack(features)  # (N, L)
seq_labels = np.array(seq_labels)

# Map original labels → indices (sorted for stability)
unique_labels = np.sort(np.unique(seq_labels))
label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}

y = np.array([label2idx[lbl] for lbl in seq_labels], dtype=np.int64)

# ──────────────── 4. Train / val / test split (80 / 10 / 10) ────────────────
X_train, X_tmp, y_train, y_tmp = train_test_split(
    features, y, test_size=0.20, random_state=args.seed, stratify=y)

X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=args.seed, stratify=y_tmp)

# ──────────────── 5. Standardise (fit on *train* only) ──────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype('float32')
X_val_scaled   = scaler.transform(X_val).astype('float32')
X_test_scaled  = scaler.transform(X_test).astype('float32')

# Add channel dimension expected by Conv1D: (N, L, 1)
X_train_scaled = X_train_scaled[..., np.newaxis]
X_val_scaled   = X_val_scaled[..., np.newaxis]
X_test_scaled  = X_test_scaled[..., np.newaxis]

# One‑hot encode targets
num_classes = len(unique_labels)

y_train_cat = to_categorical(y_train, num_classes)
y_val_cat   = to_categorical(y_val,   num_classes)
y_test_cat  = to_categorical(y_test,  num_classes)

# ─────────────────────── 6. Model‑building helper ───────────────────────────

def build_lstm_fcn(seq_len: int, nb_class: int, *, num_cells: int = 128,
                   dropout: float = 0.8) -> Model:
    """Generate an LSTM‑FCN model as per Karim et al. (2019)."""
    ip = Input(shape=(seq_len, 1))

    # LSTM branch
    x = LSTM(num_cells)(ip)
    x = Dropout(dropout)(x)

    # FCN branch
    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    z = concatenate([x, y])
    out = Dense(nb_class, activation='softmax')(z)

    return Model(ip, out)

# ──────────────────── 7. Hyper‑parameter random search ──────────────────────
HSPACE = {
    'epochs':        [200, 500, 1000],
    'batch_size':    [16, 32, 64],
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'num_cells':     [32, 64, 128],
    'dropout':       [0.5, 0.8],
}

best_cfg, best_acc, best_weights = None, -1.0, None

print(f"Running random search with {args.search_trials} trials …")
for cfg in tqdm(ParameterSampler(HSPACE, n_iter=args.search_trials,
                                 random_state=args.seed),
                total=args.search_trials, unit='cfg'):
    K.clear_session()

    model = build_lstm_fcn(L, num_classes,
                           num_cells=cfg['num_cells'], dropout=cfg['dropout'])
    optimizer = Adam(learning_rate=cfg['learning_rate'])
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train_scaled, y_train_cat,
              epochs=cfg['epochs'], batch_size=cfg['batch_size'],
              validation_data=(X_val_scaled, y_val_cat), verbose=0)

    _, val_acc = model.evaluate(X_val_scaled, y_val_cat,
                                batch_size=cfg['batch_size'], verbose=0)

    if val_acc > best_acc:
        best_acc = val_acc
        best_cfg = cfg
        best_weights = model.get_weights()  # deep‑copy via numpy

print("\n──────────────── BEST CONFIG ────────────────")
print(best_cfg, f"\nValidation accuracy = {best_acc:0.4f}")

# ──────────────────────── 8. Final training & test ──────────────────────────
K.clear_session()
model = build_lstm_fcn(L, num_classes,
                       num_cells=best_cfg['num_cells'], dropout=best_cfg['dropout'])
model.set_weights(best_weights)  # start from the best random‑search weights

optimizer = Adam(learning_rate=best_cfg['learning_rate'])
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy', metrics=['accuracy'])

history = defaultdict(list)
for epoch in range(best_cfg['epochs']):
    hist = model.fit(X_train_scaled, y_train_cat,
                     epochs=1, batch_size=best_cfg['batch_size'],
                     validation_data=(X_val_scaled, y_val_cat), verbose=0)
    history['train_loss'].append(hist.history['loss'][0])
    history['train_acc'].append(hist.history['accuracy'][0])
    history['val_loss'].append(hist.history['val_loss'][0])
    history['val_acc'].append(hist.history['val_accuracy'][0])

# ───── final test evaluation ─────
_, test_acc = model.evaluate(X_test_scaled, y_test_cat,
                            batch_size=best_cfg['batch_size'], verbose=0)

y_pred_probs = model.predict(X_test_scaled, batch_size=best_cfg['batch_size'], verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

conf_mat = confusion_matrix(y_test, y_pred)

test_f1   = f1_score(y_test, y_pred, average='macro')
test_prec = precision_score(y_test, y_pred, average='macro')

print("\n──────────────── TEST METRICS ───────────────")
print(f"Accuracy : {test_acc:0.4f}")
print(f"Macro F1 : {test_f1:0.4f}")
print(f"Macro Pr : {test_prec:0.4f}")

# ─────────────────────────── 9. I/O – save artefacts ────────────────────────
os.makedirs(args.output_dir, exist_ok=True)

stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# model & helpers
model.save(os.path.join(args.output_dir, f'lstmfcn_best_{stamp}.h5'))

import pickle
with open(os.path.join(args.output_dir, 'feature_scaler.pkl'), 'wb') as fh:
    pickle.dump(scaler, fh)
with open(os.path.join(args.output_dir, 'label_mapping.pkl'), 'wb') as fh:
    pickle.dump(label2idx, fh)
with open(os.path.join(args.output_dir, 'best_config.txt'), 'w') as fh:
    fh.write(repr(best_cfg) + '\n')

# plots
plt.figure(figsize=(8, 6))
plt.plot(history['train_loss'], label='train')
plt.plot(history['val_loss'],   label='val')
plt.xlabel('epoch')
plt.ylabel('Cross‑entropy loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, f'loss_curve_{stamp}.png'))
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(history['train_acc'], label='train')
plt.plot(history['val_acc'],   label='val')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, f'acc_curve_{stamp}.png'))
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, f'confusion_matrix_{stamp}.png'))
plt.close()

print(f"\nAll artefacts written to:  {os.path.abspath(args.output_dir)}")
