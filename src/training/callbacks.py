"""
Training callbacks: logging, early stopping, LR monitoring
"""
from loguru import logger
from typing import Dict, List
import numpy as np


class EarlyStopping:
    """Early stopping dựa trên val metric"""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = -np.inf if mode == 'max' else np.inf
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping: no improvement for {self.patience} epochs")

        return self.should_stop


class LRMonitor:
    """Theo dõi learning rate"""

    def __init__(self):
        self.lr_history: List[float] = []

    def log(self, lr: float):
        self.lr_history.append(lr)
        if len(self.lr_history) > 1:
            ratio = lr / (self.lr_history[-2] + 1e-10)
            if ratio < 0.5:
                logger.debug(f"LR dropped significantly: {self.lr_history[-2]:.2e} -> {lr:.2e}")
