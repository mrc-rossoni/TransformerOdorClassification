#!/usr/bin/env python3
"""
Train & test a 1-D ResNet on the “resistance_values.csv” dataset,
using exactly the same preprocessing, data-splits and evaluation
protocol employed for ConvTran (see convtran_randomsearch.py).
This guarantees apples-to-apples metrics.

Outputs
-------
 * resnet_best_<timestamp>.h5           – trained weights
 * feature_scaler.pkl / label_mapping.pkl
 * best_config.txt                      – chosen hyper-parameters
 * loss_curve_<timestamp>.png
 * acc_curve_<timestamp>.png
 * confusion_matrix_<timestamp>.png
All artefacts are written under --output_dir (default: “Results_ResNet”).
"""
# ──────────────────────────────────────────────────────────────────────────────
# 0. Imports & CLI
# ──────────────────────────────────────────────────────────────────────────────
import os, random, argparse, pickle
from datetime import datetime

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score

import matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── CLI ----------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--data_path',   default='resistance_values.csv')
p.add_argument('--output_dir',  default='Results_ResNet')
p.add_argument('--device',      choices=['gpu', 'cpu'], default='gpu')
p.add_argument('--sequence_len', type=int, default=10)
p.add_argument('--search_trials', type=int, default=10)
p.add_argument('--seed', type=int, default=42)
args = p.parse_args()

# ── reproducibility ----------------------------------------------------------
np.random.seed(args.seed); random.seed(args.seed); tf.random.set_seed(args.seed)
if args.device == 'cpu': tf.config.set_visible_devices([], 'GPU')

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load CSV & slice into *non-overlapping* sequences (identical to ConvTran) :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
# ──────────────────────────────────────────────────────────────────────────────
raw = pd.read_csv(args.data_path)
needed = {'Resistance Gassensor', 'label'}
if not needed <= set(raw.columns):
    raise ValueError(f'CSV needs columns {needed}')

vals   = raw['Resistance Gassensor'].astype('float32').values
labels = raw['label'].astype('int64').values
L      = args.sequence_len

seqs, seq_lbls = [], []
for s in range(0, len(vals) - L + 1, L):                    # step =L → no overlap
    e = s + L
    seqs.append(vals[s:e])
    seq_lbls.append(labels[e - 1])                          # label = last row
seqs      = np.stack(seqs)                                  # (N, L)
seq_lbls  = np.array(seq_lbls)

uniq = np.sort(np.unique(seq_lbls))
label2idx = {lbl: i for i, lbl in enumerate(uniq)}
y = np.array([label2idx[l] for l in seq_lbls], dtype=np.int64)

# ──────────────────────────────────────────────────────────────────────────────
# 2. 80 / 10 / 10 stratified split :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
# ──────────────────────────────────────────────────────────────────────────────
Xtr, Xtmp, ytr, ytmp = train_test_split(seqs, y, test_size=0.20,
                                       stratify=y, random_state=args.seed)
Xval, Xte,  yval, yte = train_test_split(Xtmp, ytmp, test_size=0.50,
                                        stratify=ytmp, random_state=args.seed)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Standardise (fit on *train* only) :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
# ──────────────────────────────────────────────────────────────────────────────
scaler  = StandardScaler()
Xtr = scaler.fit_transform(Xtr).astype('float32')
Xval = scaler.transform(Xval).astype('float32')
Xte  = scaler.transform(Xte ).astype('float32')

def _expand(x): return x.reshape((x.shape[0], x.shape[1], 1, 1))  # → (N,L,1,1)
Xtr, Xval, Xte = map(_expand, (Xtr, Xval, Xte))

num_cls = len(uniq)
Ytr  = keras.utils.to_categorical(ytr,  num_cls)
Yval = keras.utils.to_categorical(yval, num_cls)
Yte  = keras.utils.to_categorical(yte,  num_cls)

# ──────────────────────────────────────────────────────────────────────────────
# 4. ResNet definition (borrowed from your Keras script) :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}
# ──────────────────────────────────────────────────────────────────────────────
def build_resnet(input_shape, n_feat, n_cls):
    inp = keras.Input(shape=input_shape)
    x   = layers.BatchNormalization()(inp)

    def _block(x, n, downsample=False):
        shortcut = x
        if downsample:
            shortcut = layers.Conv2D(n, (1,1), padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Conv2D(n, (8,1), padding='same')(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(n, (5,1), padding='same')(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(n, (3,1), padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Add()([shortcut, x]); x = layers.ReLU()(x)
        return x

    x = _block(x,  n_feat,          downsample=True)   # 1st block
    x = _block(x,  n_feat*2,        downsample=True)   # 2nd block
    x = _block(x,  n_feat*2,        downsample=False)  # 3rd block (same ch)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(n_cls, activation='softmax')(x)
    return keras.Model(inp, out, name='ResNet_TS')

# search space (mirrors ConvTran choices)
HSPACE = {
    'epochs':        [200, 500, 1000],
    'lr':            [1e-3, 1e-4, 1e-5],
    'batch_size':    [16, 32, 64],
    'n_feature_maps':[64, 128],
}

best_cfg, best_vacc, best_w = None, -1., None
print(f'Random-search: {args.search_trials} trials …')
for t in range(args.search_trials):
    cfg = {k: random.choice(v) for k, v in HSPACE.items()}

    model = build_resnet((L,1,1), cfg['n_feature_maps'], num_cls)
    model.compile(keras.optimizers.Adam(cfg['lr']),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    rlr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                           patience=50, min_lr=1e-6, verbose=0)
    model.fit(Xtr, Ytr, validation_data=(Xval, Yval),
              epochs=cfg['epochs'], batch_size=cfg['batch_size'],
              verbose=0, callbacks=[rlr])

    _, vacc = model.evaluate(Xval, Yval, verbose=0)
    print(f'  [{t+1:02}/{args.search_trials}]  val_acc = {vacc:0.4f}   cfg = {cfg}')
    if vacc > best_vacc:
        best_vacc, best_cfg, best_w = vacc, cfg, model.get_weights()

print('\n──────── BEST CONFIG ────────')
print(best_cfg, f'   (val_acc={best_vacc:0.4f})')

# ──────────────────────────────────────────────────────────────────────────────
# 5. Final training with the best HPs
# ──────────────────────────────────────────────────────────────────────────────
model = build_resnet((L,1,1), best_cfg['n_feature_maps'], num_cls)
model.set_weights(best_w)    # warm-start from the best trial
model.compile(keras.optimizers.Adam(best_cfg['lr']),
              loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(Xtr, Ytr, validation_data=(Xval, Yval),
                    epochs=best_cfg['epochs'], batch_size=best_cfg['batch_size'],
                    verbose=0, callbacks=[
                        keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                          factor=0.5,
                                                          patience=50,
                                                          min_lr=1e-6,
                                                          verbose=0)
                    ])

# ──────────────────────────────────────────────────────────────────────────────
# 6. Test metrics (same as ConvTran) :contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}
# ──────────────────────────────────────────────────────────────────────────────
_, acc = model.evaluate(Xte, Yte, verbose=0)
y_pred = np.argmax(model.predict(Xte, verbose=0), axis=1)
f1   = f1_score(yte, y_pred, average='macro')
prec = precision_score(yte, y_pred, average='macro')
cm   = confusion_matrix(yte, y_pred)

print('\n──────── TEST METRICS ────────')
print(f'Accuracy          : {acc:0.4f}')
print(f'Macro-averaged F1 : {f1:0.4f}')
print(f'Macro-avg Prec.   : {prec:0.4f}')

# ──────────────────────────────────────────────────────────────────────────────
# 7. Save artefacts
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(args.output_dir, exist_ok=True)
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

model.save(os.path.join(args.output_dir, f'resnet_best_{stamp}.h5'))
with open(os.path.join(args.output_dir, 'feature_scaler.pkl'), 'wb') as fh:
    pickle.dump(scaler, fh)
with open(os.path.join(args.output_dir, 'label_mapping.pkl'), 'wb') as fh:
    pickle.dump(label2idx, fh)
with open(os.path.join(args.output_dir, 'best_config.txt'), 'w') as fh:
    fh.write(repr(best_cfg) + '\n')

def _plot(train, val, title, ylabel, fname):
    plt.figure(figsize=(8,6))
    plt.plot(train, label='train'); plt.plot(val, label='val')
    plt.xlabel('epoch'); plt.ylabel(ylabel); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.output_dir, fname)); plt.close()

_plot(history.history['loss'], history.history['val_loss'],
      'Cross-entropy loss', 'Loss', f'loss_curve_{stamp}.png')
_plot(history.history['accuracy'], history.history['val_accuracy'],
      'Accuracy', 'Accuracy', f'acc_curve_{stamp}.png')

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=uniq, yticklabels=uniq)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, f'confusion_matrix_{stamp}.png'))
plt.close()

print(f'\nAll artefacts written to:  {os.path.abspath(args.output_dir)}')
