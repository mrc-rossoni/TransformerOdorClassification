#!/usr/bin/env python3
"""
Train & test **InceptionTime** on the “resistance_values.csv” dataset
using **exactly the same preprocessing pipeline** as ConvTran
(StandardScaler fitted on the *training* set only, non‑overlapping
sequences of length *L*) **plus a random‑search hyper‑parameter sweep**.

The script mirrors the structure of `convtran_train.py` so that results
(plots, saved weights, confusion matrix, etc.) are written side‑by‑side
under a user‑chosen `--output_dir` for an apples‑to‑apples comparison.
"""
# ──────────────────────────────────────────────────────────────────────────────
# 0. Imports & CLI
# ──────────────────────────────────────────────────────────────────────────────
import os
import argparse
import logging
import random
from datetime import datetime
from collections import defaultdict
import copy

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score, precision_score

import matplotlib.pyplot as plt
import seaborn as sns

# ‑‑–––– reproducibility helpers ‑‑––––
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Command‑line arguments
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
# dataset / I/O
parser.add_argument('--data_path',   default='resistance_values.csv',
                    help='Path to resistance_values.csv')
parser.add_argument('--output_dir',  default='Results_InceptionTime',
                    help='Directory where models / plots are written (will be created)')
# system
parser.add_argument('--device',      choices=['gpu', 'cpu'], default='gpu',
                    help='Force a device (falls back to CPU if no GPU found)')
# preprocessing
parser.add_argument('--sequence_len', type=int, default=10,
                    help='Number of consecutive readings per sample')
# random‑search budget
parser.add_argument('--search_trials', type=int, default=10,
                    help='How many hyper‑parameter configurations to try')
# misc
parser.add_argument('--verbose', action='store_true', help='Verbose Keras logs')
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# 2. Device set‑up (GPU memory growth, optional CPU pinning)
# ──────────────────────────────────────────────────────────────────────────────
physical_gpus = tf.config.list_physical_devices('GPU')
if args.device == 'gpu' and physical_gpus:
    try:
        tf.config.experimental.set_memory_growth(physical_gpus[0], True)
        DEVICE = '/GPU:0'
        print("[INFO] Using GPU")
    except RuntimeError as e:
        print(f"[WARN] Could not set GPU memory growth: {e}; using CPU")
        DEVICE = '/CPU:0'
else:
    DEVICE = '/CPU:0'
    print("[INFO] Using CPU")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Data loading & slicing into non‑overlapping sequences
# ──────────────────────────────────────────────────────────────────────────────
raw = pd.read_csv(args.data_path)
required_cols = {'Resistance Gassensor', 'label'}
if not required_cols.issubset(raw.columns):
    raise ValueError(f"CSV must contain columns {required_cols}, found {set(raw.columns)}")

values = raw['Resistance Gassensor'].values.astype('float32')
labels_raw = raw['label'].values  # keep original dtype for mapping

L = args.sequence_len
features, seq_labels = [], []
for start in range(0, len(values) - L + 1, L):  # non‑overlap
    end = start + L
    features.append(values[start:end])
    seq_labels.append(labels_raw[end - 1])  # last row’s label

features = np.stack(features)              # shape (N, L)
seq_labels = np.array(seq_labels)

unique_labels = np.sort(np.unique(seq_labels))
label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}

y_int = np.array([label2idx[lbl] for lbl in seq_labels], dtype=np.int64)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Train / val / test split (80 / 10 / 10, stratified)
# ──────────────────────────────────────────────────────────────────────────────
X_train, X_tmp, y_train_int, y_tmp_int = train_test_split(
    features, y_int, test_size=0.20, random_state=SEED, stratify=y_int)
X_val, X_test, y_val_int, y_test_int = train_test_split(
    X_tmp, y_tmp_int, test_size=0.50, random_state=SEED, stratify=y_tmp_int)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Standardisation (fit on *train* only)
# ──────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype('float32')
X_val_scaled   = scaler.transform(X_val).astype('float32')
X_test_scaled  = scaler.transform(X_test).astype('float32')

# reshape for Conv1D: (samples, timesteps, channels)
X_train_scaled = X_train_scaled[..., np.newaxis]
X_val_scaled   = X_val_scaled[...,   np.newaxis]
X_test_scaled  = X_test_scaled[...,  np.newaxis]

# one‑hot encode labels for Keras
encoder = OneHotEncoder(sparse_output=False)
y_train_oh = encoder.fit_transform(y_train_int.reshape(-1, 1)).astype('float32')
y_val_oh   = encoder.transform(y_val_int.reshape(-1, 1)).astype('float32')
y_test_oh  = encoder.transform(y_test_int.reshape(-1, 1)).astype('float32')
num_classes = y_train_oh.shape[1]

# ──────────────────────────────────────────────────────────────────────────────
# 6. InceptionTime implementation (minimal, self‑contained)
# ──────────────────────────────────────────────────────────────────────────────
class InceptionTimeClassifier:
    """Single‑branch InceptionTime as per Fawaz et al.
       All hyper‑parameters passed via kwargs."""

    def __init__(self, input_shape, num_classes, **cfg):
        self.cfg = cfg
        self.model = self._build(input_shape, num_classes)

    # ‑‑ internal helpers ‑‑
    def _inception_module(self, x, stride=1):
        k = self.cfg['kernel_size']
        bneck = self.cfg['use_bottleneck'] and x.shape[-1] > 1
        if bneck:
            x_in = keras.layers.Conv1D(self.cfg['bottleneck_size'], 1, padding='same', use_bias=False)(x)
        else:
            x_in = x
        kernel_sizes = [k // (2 ** i) for i in range(3)]
        convs = [keras.layers.Conv1D(self.cfg['nb_filters'], ks, strides=stride,
                                     padding='same', use_bias=False)(x_in)
                 for ks in kernel_sizes]
        pool = keras.layers.MaxPool1D(3, strides=stride, padding='same')(x)
        conv_pool = keras.layers.Conv1D(self.cfg['nb_filters'], 1, padding='same', use_bias=False)(pool)
        convs.append(conv_pool)
        x_out = keras.layers.Concatenate(axis=2)(convs)
        x_out = keras.layers.BatchNormalization()(x_out)
        x_out = keras.layers.Activation('relu')(x_out)
        return x_out

    def _shortcut(self, input_tensor, out_tensor):
        shortcut = keras.layers.Conv1D(out_tensor.shape[-1], 1, padding='same', use_bias=False)(input_tensor)
        shortcut = keras.layers.BatchNormalization()(shortcut)
        x = keras.layers.Add()([shortcut, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def _build(self, input_shape, num_classes):
        inp = keras.layers.Input(shape=input_shape)
        x, input_res = inp, inp
        for d in range(self.cfg['depth']):
            x = self._inception_module(x)
            if self.cfg['use_residual'] and d % 3 == 2:
                x = self._shortcut(input_res, x)
                input_res = x
        gap = keras.layers.GlobalAveragePooling1D()(x)
        out = keras.layers.Dense(num_classes, activation='softmax')(gap)
        model = keras.Model(inp, out)
        opt = keras.optimizers.Adam(learning_rate=self.cfg['lr'])
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # ‑‑ training / evaluation helpers ‑‑
    def fit(self, X_train, y_train, X_val, y_val):
        callbacks = [keras.callbacks.ReduceLROnPlateau(patience=50, factor=0.5, min_lr=1e-6)]
        history = self.model.fit(X_train, y_train,
                                 epochs=self.cfg['epochs'],
                                 batch_size=self.cfg['batch_size'],
                                 verbose=args.verbose,
                                 validation_data=(X_val, y_val),
                                 callbacks=callbacks)
        # return best val accuracy
        best_acc = max(history.history['val_accuracy'])
        return best_acc, history

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)

# ──────────────────────────────────────────────────────────────────────────────
# 7. Hyper‑parameter random search
# ──────────────────────────────────────────────────────────────────────────────
HSPACE = {
    'nb_filters':      [16, 32, 64],
    'depth':           [6, 9],
    'kernel_size':     [41, 21],
    'batch_size':      [16, 32, 64],
    'use_residual':    [True, False],
    'use_bottleneck':  [True, False],
    'bottleneck_size': [16, 32],
    'lr':              [1e-3, 1e-4, 1e-5],
    'epochs':          [200, 500, 1000],
}

best_cfg, best_val_acc, best_weights = None, -1.0, None
print(f"[INFO] Random‑search with {args.search_trials} trials …")

with tf.device(DEVICE):
    for trial in range(args.search_trials):
        cfg = {k: random.choice(v) for k, v in HSPACE.items()}
        cfg.setdefault('bottleneck_size', 32)
        print(f"\n[Trial {trial+1}/{args.search_trials}] cfg = {cfg}")
        clf = InceptionTimeClassifier(X_train_scaled.shape[1:], num_classes, **cfg)
        val_acc, _ = clf.fit(X_train_scaled, y_train_oh, X_val_scaled, y_val_oh)
        print(f"[Trial {trial+1}] val_accuracy = {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cfg = cfg
            best_weights = clf.model.get_weights()

print("\n──────────────── BEST CONFIG ────────────────")
print(best_cfg, f"\nValidation accuracy = {best_val_acc:0.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 8. Final training with best config & test evaluation
# ──────────────────────────────────────────────────────────────────────────────
with tf.device(DEVICE):
    clf = InceptionTimeClassifier(X_train_scaled.shape[1:], num_classes, **best_cfg)
    clf.model.set_weights(best_weights)  # warm‑start from best trial weights
    # Continue training for the same number of epochs on train set only
    clf.fit(X_train_scaled, y_train_oh, X_val_scaled, y_val_oh)
    test_loss, test_acc = clf.evaluate(X_test_scaled, y_test_oh)
    y_pred_prob = clf.model.predict(X_test_scaled, batch_size=best_cfg['batch_size'])

# confusion matrix & derived metrics
    y_pred = np.argmax(y_pred_prob, axis=1)
    conf_mat = confusion_matrix(y_test_int, y_pred)
    macro_f1 = f1_score(y_test_int, y_pred, average='macro')
    macro_prec = precision_score(y_test_int, y_pred, average='macro')

print("\n──────────────── TEST METRICS ───────────────")
print(f"Accuracy : {test_acc:0.4f}")
print(f"Macro F1 : {macro_f1:0.4f}")
print(f"Macro Pr : {macro_prec:0.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 9. I/O – save artefacts
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(args.output_dir, exist_ok=True)
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# model & helpers
clf.model.save(os.path.join(args.output_dir, f'inceptiontime_best_{stamp}.h5'))
np.savez_compressed(os.path.join(args.output_dir, 'model_weights.npz'), *best_weights)

import pickle
with open(os.path.join(args.output_dir, 'feature_scaler.pkl'), 'wb') as fh:
    pickle.dump(scaler, fh)
with open(os.path.join(args.output_dir, 'label_mapping.pkl'), 'wb') as fh:
    pickle.dump(label2idx, fh)
with open(os.path.join(args.output_dir, 'best_config.txt'), 'w') as fh:
    fh.write(repr(best_cfg) + '\n')

# plots
history = defaultdict(list)
# We didn't store full history across both stages; store the final stage (clf.model.history)
# For brevity, plot placeholder if available
if hasattr(clf.model, 'history') and clf.model.history.history:
    hist_dict = clf.model.history.history
    for k in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
        if k in hist_dict:
            history[k] = hist_dict[k]

    def _plot_curve(key, ylabel):
        plt.figure(figsize=(8,6))
        plt.plot(history[key], label='train')
        val_key = 'val_' + key.split('_')[0] if key.startswith('loss') else 'val_' + key
        if val_key in history:
            plt.plot(history[val_key], label='val')
        plt.xlabel('epoch'); plt.ylabel(ylabel); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'{key}_curve_{stamp}.png'))
        plt.close()

    if history:
        if 'loss' in history:
            _plot_curve('loss', 'Cross‑entropy loss')
        if 'accuracy' in history:
            _plot_curve('accuracy', 'Accuracy')

plt.figure(figsize=(6,5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, f'confusion_matrix_{stamp}.png'))
plt.close()

print(f"\nAll artefacts written to:  {os.path.abspath(args.output_dir)}")