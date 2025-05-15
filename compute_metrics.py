import numpy as np
from typing import Dict, List

def compute_metrics(confusion_matrix: np.ndarray) -> Dict[str, float]:
    """Compute Accuracy, Macro‑Averaged F1 Score, and Macro‑Averaged False Positive Rate (FPR)
    from a square confusion matrix.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Square matrix where rows represent true classes and columns represent predicted classes.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys ``accuracy``, ``macro_f1``, and ``macro_fpr``.
    """

    if confusion_matrix.ndim != 2 or confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("Confusion matrix must be square (n_classes × n_classes).")

    cm = confusion_matrix.astype(float)
    n_classes = cm.shape[0]
    total_instances = cm.sum()

    # True Positives for each class are the diagonal elements
    TP = np.diag(cm)
    # False Negatives: row sum minus TP
    FN = cm.sum(axis=1) - TP
    # False Positives: column sum minus TP
    FP = cm.sum(axis=0) - TP
    # True Negatives: total minus TP, FP, FN for each class
    TN = total_instances - (TP + FP + FN)

    # --- Accuracy ---
    accuracy = TP.sum() / total_instances if total_instances else 0.0

    # --- Precision and Recall per class ---
    with np.errstate(divide="ignore", invalid="ignore"):
        precision_per_class = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) != 0)
        recall_per_class = np.divide(TP, TP + FN, out=np.zeros_like(TP), where=(TP + FN) != 0)

    # Macro Precision (MP) and Macro Recall (MR)
    MP = precision_per_class.mean()
    MR = recall_per_class.mean()

    # --- Macro‑averaged F1 ---
    macro_f1 = (2 * MR * MP) / (MR + MP) if (MR + MP) else 0.0

    # --- False Positive Rate (per class) and Macro FPR ---
    with np.errstate(divide="ignore", invalid="ignore"):
        fpr_per_class = np.divide(FP, FP + TN, out=np.zeros_like(FP), where=(FP + TN) != 0)
    macro_fpr = fpr_per_class.mean()

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "macro_fpr": float(macro_fpr),
    }


def _demo():
    """Demonstration with a 3×3 confusion matrix."""
    demo_cm = np.array([[82,  6,  0,  0,  0,  0,  0],
 [ 1, 100, 0,  0,  0,  0,  0],
 [ 4,  1, 92,  1,  0,  0,  0],
 [ 0,  3, 30, 39,  0,  0,  0],
 [ 0,  0,  0,  0, 37,  0,  0],
 [ 0,  0,  0,  0,  1, 42,  0],
 [ 0,  0,  0,  0,  0,  0, 36]]
)
    metrics = compute_metrics(demo_cm)
    print("Confusion Matrix:\n", demo_cm)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    _demo()