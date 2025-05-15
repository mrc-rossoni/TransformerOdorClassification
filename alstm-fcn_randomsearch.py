#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
# Train & test ALSTM-FCN on the “resistance_values.csv” dataset
# (pipeline aligned with convtran_randomsearch.py)
# ──────────────────────────────────────────────────────────────────────────────
import os
import argparse
import random
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Input, Dense, LSTM, Dropout, Permute,
                                     Conv1D, BatchNormalization,
                                     GlobalAveragePooling1D, Reshape,
                                     concatenate, Layer, Activation)
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ──────────────────────────────────────────────────────────────────────────────
# 0. CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
# dataset / I/O
parser.add_argument('--data_path',   default='resistance_values.csv')
parser.add_argument('--output_dir',  default='Results_ALSTM')
# system
parser.add_argument('--device',      choices=['cuda', 'cpu'], default='cuda')
# training
parser.add_argument('--sequence_len', type=int, default=10)
parser.add_argument('--search_trials', type=int, default=10)
# reproducibility
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# 1. Reproducibility helpers
# ──────────────────────────────────────────────────────────────────────────────
os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

if args.device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ──────────────────────────────────────────────────────────────────────────────
# 2. Load CSV & slice into *non-overlapping* sequences  (exactly like ConvTran)
# ──────────────────────────────────────────────────────────────────────────────
raw = pd.read_csv(args.data_path)
required_cols = {'Resistance Gassensor', 'label'}
if not required_cols.issubset(raw.columns):
    raise ValueError(f"CSV must contain columns {required_cols}")

values = raw['Resistance Gassensor'].values.astype('float32')
labels = raw['label'].values.astype('int64')

L = args.sequence_len
features, seq_labels = [], []
for start in range(0, len(values) - L + 1, L):
    end = start + L
    features.append(values[start:end])
    seq_labels.append(labels[end-1])          # last point’s label

features   = np.stack(features)              # (N, L)
seq_labels = np.array(seq_labels)

unique_labels = np.sort(np.unique(seq_labels))
label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
y_idx = np.array([label2idx[lbl] for lbl in seq_labels], dtype=np.int64)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Train / val / test split  (80 / 10 / 10)  (stratified) – same RNG & seed
# ──────────────────────────────────────────────────────────────────────────────
X_train, X_tmp, y_train, y_tmp = train_test_split(
    features, y_idx, test_size=0.20, random_state=args.seed, stratify=y_idx)

X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=args.seed, stratify=y_tmp)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Standardise (fit on *train* only, apply everywhere) – same as ConvTran
# ──────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype('float32')
X_val   = scaler.transform(X_val).astype('float32')
X_test  = scaler.transform(X_test).astype('float32')

# add channel dimension expected by ALSTM-FCN: (N, 1, L)
X_train = X_train[:, np.newaxis, :]
X_val   = X_val[:,   np.newaxis, :]
X_test  = X_test[:,  np.newaxis, :]

# one-hot labels for Keras
y_train_cat = to_categorical(y_train, num_classes=len(unique_labels))
y_val_cat   = to_categorical(y_val,   num_classes=len(unique_labels))
y_test_cat  = to_categorical(y_test,  num_classes=len(unique_labels))

# ──────────────────────────────────────────────────────────────────────────────
# 5. Model definition (ALSTM-FCN with Attention-LSTM branch)
# ──────────────────────────────────────────────────────────────────────────────
class AttentionLSTM(Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='uniform', trainable=True)
        self.b = self.add_weight(
            shape=(self.units,), initializer='uniform', trainable=True)
        self.u = self.add_weight(
            shape=(self.units, 1), initializer='uniform', trainable=True)
        super().build(input_shape)

    def call(self, x):
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.squeeze(ait, -1)
        ait = tf.exp(ait)
        ait /= tf.reduce_sum(ait, axis=1, keepdims=True)
        weighted = x * tf.expand_dims(ait, -1)
        return tf.reduce_sum(weighted, axis=1)

def build_alstm_fcn(seq_len, n_classes, num_cells=64, dropout=0.8):
    ip = Input(shape=(1, seq_len))
    x = Reshape((seq_len, 1))(ip)                   # → (L, 1)

    # Attention-LSTM branch
    lstm_out = AttentionLSTM(num_cells)(x)
    lstm_out = Dropout(dropout)(lstm_out)

    # FCN branch
    y = Permute((2, 1))(ip)                         # → (1, L)
    y = Conv1D(128, 8, padding='same',
               kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y); y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same',
               kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y); y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same',
               kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y); y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    out = concatenate([lstm_out, y])
    out = Dense(n_classes, activation='softmax')(out)
    return Model(ip, out)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Hyper-parameter random search (budget = args.search_trials)
#    pick best on *val* accuracy
# ──────────────────────────────────────────────────────────────────────────────
HSPACE = {
    'epochs':        [200, 500, 1000],
    'batch_size':    [64, 128, 256],
    'learning_rate': [1e-4, 1e-5, 1e-6],
    'num_cells':     [32, 64, 128],
}
best_cfg, best_acc, best_weights = None, -1.0, None

print(f"\nRunning random search with {args.search_trials} trials …")
for cfg in tqdm(ParameterSampler(HSPACE, n_iter=args.search_trials,
                                 random_state=args.seed),
                total=args.search_trials, unit='cfg'):

    K.clear_session()
    model = build_alstm_fcn(L, len(unique_labels), cfg['num_cells'])
    model.compile(Adam(cfg['learning_rate']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train_cat,
              epochs=cfg['epochs'],
              batch_size=cfg['batch_size'],
              validation_data=(X_val, y_val_cat),
              verbose=0)

    val_loss, val_acc = model.evaluate(
        X_val, y_val_cat, batch_size=cfg['batch_size'], verbose=0)

    if val_acc > best_acc:
        best_acc   = val_acc
        best_cfg   = cfg
        best_weights = model.get_weights()

print("\n──────────────── BEST CONFIG ────────────────")
print(best_cfg, f"\nValidation accuracy = {best_acc:0.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Final training with best hyper-parameters & test evaluation
# ──────────────────────────────────────────────────────────────────────────────
K.clear_session()
model = build_alstm_fcn(L, len(unique_labels), best_cfg['num_cells'])
model.compile(Adam(best_cfg['learning_rate']),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.set_weights(best_weights)     # warm-start from searched weights

history = defaultdict(list)
for epoch in range(best_cfg['epochs']):
    hist = model.fit(X_train, y_train_cat,
                     batch_size=best_cfg['batch_size'],
                     epochs=1, verbose=0)
    val_loss, val_acc = model.evaluate(
        X_val, y_val_cat, batch_size=best_cfg['batch_size'], verbose=0)

    history['train_loss'].append(hist.history['loss'][0])
    history['train_acc'].append(hist.history['accuracy'][0])
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

# ── Test metrics
test_loss, test_acc = model.evaluate(
    X_test, y_test_cat, batch_size=best_cfg['batch_size'], verbose=0)

y_pred = model.predict(
    X_test, batch_size=best_cfg['batch_size'], verbose=0).argmax(1)
f1  = f1_score(y_test, y_pred, average='macro')
prec = precision_score(y_test, y_pred, average='macro')
cm  = confusion_matrix(y_test, y_pred)

print("\n──────────────── TEST METRICS ───────────────")
print(f"Accuracy : {test_acc:0.4f}")
print(f"Macro F1 : {f1:0.4f}")
print(f"Macro Pr : {prec:0.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 8. I/O – save artefacts (same filenames / plots as ConvTran)
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(args.output_dir, exist_ok=True)
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

model.save(os.path.join(args.output_dir,
            f'alstm_fcn_best_{stamp}.h5'))
with open(os.path.join(args.output_dir, 'feature_scaler.pkl'), 'wb') as fh:
    pickle.dump(scaler, fh)
with open(os.path.join(args.output_dir, 'label_mapping.pkl'), 'wb') as fh:
    pickle.dump(label2idx, fh)
with open(os.path.join(args.output_dir, 'best_config.txt'), 'w') as fh:
    fh.write(repr(best_cfg) + '\n')

def _plot_curve(key, ylabel):
    plt.figure(figsize=(8,6))
    plt.plot(history[f'train_{key}'], label='train')
    plt.plot(history[f'val_{key}'],   label='val')
    plt.xlabel('epoch'); plt.ylabel(ylabel); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,
                f'{key}_curve_{stamp}.png'))
    plt.close()

_plot_curve('loss', 'Cross-entropy loss')
_plot_curve('acc',  'Accuracy')

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
plt.savefig(os.path.join(args.output_dir,
            f'confusion_matrix_{stamp}.png'))
plt.close()

print(f"\nAll artefacts written to:  {os.path.abspath(args.output_dir)}")
