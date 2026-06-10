"""
Metrics calculator cho multi-class classification
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from typing import Dict, Optional
from loguru import logger


class MetricsCalculator:
    """
    Tính toán đầy đủ các metrics:
    - Accuracy
    - F1 (macro, weighted, per-class)
    - Precision, Recall
    - ROC-AUC (OvR)
    - Confusion Matrix
    """

    LABEL_NAMES = ["clean", "toxic", "spam", "adult"]

    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Tính tất cả metrics"""

        metrics = {}

        # Accuracy
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))

        # F1 scores
        metrics['f1_macro'] = float(
            f1_score(y_true, y_pred, average='macro', zero_division=0)
        )
        metrics['f1_weighted'] = float(
            f1_score(y_true, y_pred, average='weighted', zero_division=0)
        )

        # Per-class F1
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, name in enumerate(self.LABEL_NAMES):
            if i < len(f1_per_class):
                metrics[f'f1_{name}'] = float(f1_per_class[i])

        # Precision & Recall (macro)
        metrics['precision_macro'] = float(
            precision_score(y_true, y_pred, average='macro', zero_division=0)
        )
        metrics['recall_macro'] = float(
            recall_score(y_true, y_pred, average='macro', zero_division=0)
        )

        # ROC-AUC nếu có probabilities
        if y_probs is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['roc_auc'] = float(
                    roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
                )
            except Exception:
                metrics['roc_auc'] = 0.0

        return metrics

    def print_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """In classification report chi tiết"""
        report = classification_report(
            y_true, y_pred,
            target_names=self.LABEL_NAMES,
            digits=4
        )
        logger.info(f"\nClassification Report:\n{report}")

        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.info(f"Rows = True labels, Cols = Predicted labels")
        logger.info(f"Labels: {self.LABEL_NAMES}")
