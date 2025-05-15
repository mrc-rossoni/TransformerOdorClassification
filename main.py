import os
import argparse
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from art import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import Project Modules
from utils import Setup, Initialization, Data_Verifier
from Models.model import model_factory, count_parameters
from Models.optimizers import get_optimizer
from Models.loss import get_loss_module
from Models.utils import load_model
from Training import SupervisedTrainer, train_runner

logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()

# Input and Output
parser.add_argument('--data_path', default='resistance_values.csv', help='Path to resistance_values.csv file')
parser.add_argument('--output_dir', default='Results', help='Root output directory. Must exist.')

# Model Parameter and Hyperparameter
parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.01, help='Dropout regularization ratio')
parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='accuracy', help='Metric used for defining best epoch')

# System
parser.add_argument('--device', choices={'cpu', 'cuda'}, default='cuda', help='Device to use for training')

# Example function to load data
def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

def plot_confusion_matrix(conf_mat, filename):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(filename)
    plt.close()

def plot_metrics(metrics_dict, metric_name, color_train, color_val, filename):
    plt.figure(figsize=(10, 7))
    plt.plot(metrics_dict['train'], label=f'Train {metric_name}', color=color_train)
    plt.plot(metrics_dict['val'], label=f'Validation {metric_name}', color=color_val)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    args = parser.parse_args()
    data = load_data(args.data_path)
    print(data.head())
    print("Columns in the dataset:", data.columns)

    # Verifica se CUDA è disponibile
    cuda_available = torch.cuda.is_available()
    print("CUDA available: ", cuda_available)

    # Se CUDA è disponibile, stampa ulteriori informazioni sulla GPU
    if cuda_available:
        print("CUDA version: ", torch.version.cuda)
        print("Number of GPUs: ", torch.cuda.device_count())
        print("GPU Name: ", torch.cuda.get_device_name(0))

    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Extract features and labels
    values = data.iloc[:, 0].values  # Assuming the values are in the first column
    labels = data.iloc[:, 1].values  # Assuming the labels are in the second column

    # Create non-overlapping sequences of 10 values
    sequence_length = 10
    features = []
    sequence_labels = []

    for i in range(0, len(values) - sequence_length + 1, sequence_length):
        features.append(values[i:i + sequence_length])
        sequence_labels.append(labels[i + sequence_length - 1])  # Assuming the label is the last value in the sequence

    features = np.array(features)
    sequence_labels = np.array(sequence_labels)

    # Map labels to integer values if necessary
    unique_labels = np.unique(sequence_labels)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[label] for label in sequence_labels])

    # Example dataset split
    train_features, test_features, train_labels, test_labels = train_test_split(features, mapped_labels, test_size=0.2, random_state=42)
    
    # Further split the training set into training and validation sets
    test_features, val_features, test_labels, val_labels = train_test_split(test_features, test_labels, test_size=0.5, random_state=42)

    # Convert to PyTorch tensors
    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)  # Ensure labels are of type long
    val_features = torch.tensor(val_features, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.long)  # Ensure labels are of type long
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)  # Ensure labels are of type long

    # Create PyTorch datasets
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    # Print the shape of the tensors in the datasets
    print(f"Train features shape: {train_features.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Validation features shape: {val_features.shape}")
    print(f"Validation labels shape: {val_labels.shape}")
    print(f"Test features shape: {test_features.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    # Example configuration
    config = {
        'Net_Type': 'C-T',
        'Data_shape': [train_features.shape[0], 1, train_features.shape[1]],  # Modifica in base ai tuoi dati
        'num_labels': len(unique_labels),  # Number of unique labels
        'dropout': args.dropout,
        'emb_size': 128,
        'num_heads': 4,
        'dim_ff': 256,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'Fix_pos_encode': 'tAPE',
        'Rel_pos_encode': 'eRPE'
    }

    model = model_factory(config)
    print(model)

    # Setup optimizer and loss module
    optimizer_class = get_optimizer("Adam")
    optimizer = optimizer_class(model.parameters(), lr=args.lr)
    loss_module = get_loss_module()

    # Setup data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training
    trainer = SupervisedTrainer(model, train_loader, device, loss_module, optimizer, print_conf_mat=True)
    train_metrics = {'accuracy': [], 'loss': []}
    val_metrics = {'accuracy': [], 'loss': []}
    for epoch in range(args.epochs):
        train_loss, train_accuracy = trainer.train_epoch(epoch)
        train_metrics['loss'].append(train_loss)
        train_metrics['accuracy'].append(train_accuracy)
        print(f"Epoch {epoch+1}/{args.epochs} - Training Loss: {train_loss:.4f} - Training Accuracy: {train_accuracy:.4f}")

        # Evaluate on validation set
        val_trainer = SupervisedTrainer(model, val_loader, device, loss_module)
        val_metrics_epoch, _ = val_trainer.evaluate()
        val_metrics['loss'].append(val_metrics_epoch['loss'])
        val_metrics['accuracy'].append(val_metrics_epoch['accuracy'])
        print(f"Epoch {epoch+1}/{args.epochs} - Validation Loss: {val_metrics_epoch['loss']:.4f} - Validation Accuracy: {val_metrics_epoch['accuracy']:.4f}")

    # Evaluation on test set
    test_trainer = SupervisedTrainer(model, test_loader, device, loss_module)
    test_metrics, test_details = test_trainer.evaluate()
    print("\n--- Test Results ---")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Macro Average F1 Score: {test_metrics['macro_f1_score']:.4f}")

    # Calculate False Positive Rate (FPR)
    conf_mat = confusion_matrix(test_details['targets'], test_details['predictions'])
    FP = conf_mat.sum(axis=0) - np.diag(conf_mat)
    TP = np.diag(conf_mat)
    FPR = FP / (FP + TP)
    mean_FPR = np.mean(FPR)
    print(f"Mean False Positive Rate (FPR): {mean_FPR:.4f}")

    # Save model
    model_save_path = os.path.join(args.output_dir, 'best_trained_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Best model saved to {model_save_path}")

    # Plot confusion matrix
    plot_confusion_matrix(conf_mat, filename=os.path.join(args.output_dir, 'confusion_matrix.png'))

    # Plot metrics
    plot_metrics({'train': train_metrics['accuracy'], 'val': val_metrics['accuracy']}, metric_name='accuracy', color_train='b', color_val='r', filename=os.path.join(args.output_dir, 'accuracy_plot.png'))
    plot_metrics({'train': train_metrics['loss'], 'val': val_metrics['loss']}, metric_name='loss', color_train='g', color_val='orange', filename=os.path.join(args.output_dir, 'loss_plot.png'))

    print("\n--- Final Test Results ---")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Macro Average F1 Score: {test_metrics['macro_f1_score']:.4f}")
    print(f"Mean False Positive Rate (FPR): {mean_FPR:.4f}")