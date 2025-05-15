# Time Series Classification on E-Nose Data with ConvTran and Comparative Models

This repository provides a complete framework for classifying time series collected from gas sensors (e-noses) using a range of deep learning models, including the Transformer-based model **ConvTran**. Two curated datasets, `resistance_values.csv` and `resistance_values7class.csv`, are used to benchmark performance under both balanced and imbalanced class conditions.

---

## 📦 Datasets

| Dataset File                  | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `resistance_values.csv`      | Case Study 1 (**CoffeePow-4**): 4 classes (3 coffee types + ambient air)    |
| `resistance_values_7class.csv`| Case Study 2 (**Aroma-7**): 7 classes (3 coffee type + ambient air + 3 fragranced creams) |

Both datasets were collected using a Bosch **BME688** sensor array, executing controlled 10-step temperature cycles to generate distinctive odor fingerprints.

### 📈 CoffeePow-4 (4 Classes)
- **Classes**: Ambient Air, Borbone Coffee, Lavazza Coffee, Primia Coffee
- **Samples**: 3583 sequences, class-balanced
- **Task**: Evaluate baseline accuracy

### 🌿 Aroma-7 (7 Classes)
- **Classes**: All from CoffeePow-4 + Almond Cream, Thyme Cream, Marigold Cream
- **Samples**: 4750 sequences (imbalanced, 3:1 coffee:cream ratio)
- **Task**: Test robustness under class imbalance and cross-domain volatility

### Sensor Setup

- **Hardware**: Bosch BME688 ×8 array
- **Features used**: Gas resistance (10 readings per heater cycle)
- **Preprocessing**:
  - Sensor-specific standardization (zero mean, unit variance)
  - Stratified 80/10/10 train/val/test split

---

## 🧠 Models Implemented

| Model           | Type        | Key Features                                                            |
|----------------|-------------|--------------------------------------------------------------------------|
| ConvTran        | Transformer | tAPE + eRPE position encodings                                          |
| ALSTM-FCN       | Hybrid      | LSTM + attention + 1D CNN                                               |
| LSTM-FCN        | Hybrid      | LSTM + 1D CNN                                                           |
| InceptionTime   | CNN         | Residual inception modules for multi-scale temporal pattern detection   |
| FCN             | CNN         | Lightweight 3-layer 1D CNN with global pooling                          |
| ResNet          | CNN         | Deep residual CNN architecture                                          |
| Bosch AI Studio | Proprietary | Baseline classifier (black-box)                                         |

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

Requirements include: `torch`, `tensorflow`, `scikit-learn`, `matplotlib`, `numpy`, and `pandas`.

---

## ⚙️ Usage

Each model is executed via its corresponding script. Use either dataset depending on the case study.

### Common CLI Parameters

```bash
--data_path          Path to CSV file (e.g., resistance_values.csv)
--output_dir         Directory for saving outputs
--device             'cuda' or 'cpu' (default: 'cuda')
--sequence_len       Length of input sequences (default: 10)
--search_trials      Number of random search configurations (default: 10)
--seed               Random seed (default: 42)
```

---

## 🧪 Training Pipeline

1. Load and preprocess dataset
2. Execute random hyperparameter search
3. Train best configuration
4. Evaluate on test set
5. Save model, metrics, confusion matrix, and plots

All models use the same preprocessing logic to ensure fair benchmarking.

---

## 🔍 Hyperparameter Spaces

### ConvTran

```python
HSPACE = {
    'emb_size': [32, 64, 128],
    'num_heads': [2, 4, 8],
    'dim_ff': [64, 128, 256],
    'epochs': [200, 500, 1000],
    'lr': [1e-3, 1e-4, 1e-5],
    'batch_size': [16, 32, 64],
    'dropout': [0.0, 0.1, 0.2]
}
```

See scripts for other model-specific hyperparameter spaces.

---

## 📈 Results Summary

### Case Study 1: CoffeePow-4 (Balanced)

| Model         | Accuracy | F1 Score | FPR   | Params    | Latency (ms) |
| ------------- | -------- | -------- | ----- | --------- | ------------ |
| FCN           | 97.21%   | 97.05%   | 0.92% | 92,100    | 0.41         |
| ConvTran      | 96.66%   | 96.42%   | 1.12% | 24,298    | 0.92         |
| InceptionTime | 96.94%   | 96.82%   | 1.04% | 477,700   | 63.0         |
| ALSTM-FCN     | 95.54%   | 95.32%   | 1.50% | 266,440   | 63.3         |
| LSTM-FCN      | 93.87%   | 93.46%   | 2.05% | 292,612   | 62.4         |
| ResNet        | 93.02%   | 92.63%   | 2.35% | 2,012,552 | 65.3         |
| Bosch Model   | 87.81%   | 86.98%   | 4.03% | N/A       | N/A          |

### Case Study 2: Aroma-7 (Imbalanced)

| Model         | Accuracy | F1 Score | FPR   |
| ------------- | -------- | -------- | ----- |
| ConvTran      | 95.16%   | 96.03%   | 0.85% |
| ResNet        | 93.68%   | 94.87%   | 1.11% |
| InceptionTime | 93.47%   | 94.76%   | 1.16% |
| ALSTM-FCN     | 90.11%   | 92.37%   | 1.77% |
| FCN           | 77.47%   | 85.12%   | 3.96% |
| LSTM-FCN      | 69.26%   | 78.82%   | 5.44% |

> 🔍 ConvTran preserves performance under real-world imbalance, outperforming convolutional and recurrent baselines on minority cream classes.

---

## 🧾 Metrics

* **Accuracy**: Overall correct predictions
* **Macro F1 Score**: Harmonic mean of per-class precision and recall
* **False Positive Rate (FPR)**: Rate of incorrect class activations
* **Confusion Matrix**: Insightful view of class confusions

---

## 📚 Citation

---

## 👥 Credits

---

## 🧠 Insights from the Paper

* ConvTran fits within **115 KB** and uses **only 24k parameters**, making it ideal for microcontrollers
* Handles long-range dependencies via **self-attention**
* Maintains performance as the number of odor classes increases
* Clearly superior to proprietary black-box models in both accuracy and interpretability

---

## 📂 Repository Structure

---

## 📌 License

Released under the MIT License. All datasets and model configurations are made available for reproducibility and future research.
