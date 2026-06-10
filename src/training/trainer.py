"""
Training pipeline đầy đủ cho PhoBERT classifier
"""
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from loguru import logger
from tqdm import tqdm
import time

from src.models.phobert_classifier import PhoBERTClassifier
from src.training.metrics import MetricsCalculator


class PhoBERTTrainer:
    """
    Trainer đầy đủ với:
    - Mixed precision training (FP16)
    - Gradient accumulation
    - Learning rate scheduling (warmup + cosine decay)
    - Early stopping
    - Checkpoint saving
    - Metrics tracking
    """

    def __init__(
        self,
        model: PhoBERTClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        output_dir: str = "models/checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self.model = self.model.to(self.device)

        # Optimizer với weight decay
        self.optimizer = self._create_optimizer()

        # Scheduler
        total_steps = len(train_loader) * config['num_epochs']
        warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Mixed precision
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        logger.info(f"Mixed precision training: {self.use_amp}")

        # Gradient accumulation
        self.grad_accum_steps = config.get('gradient_accumulation_steps', 1)

        # Metrics
        self.metrics_calc = MetricsCalculator(num_classes=3)

        # Tracking
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'val_f1': [], 'val_accuracy': [],
            'learning_rates': []
        }

    def _create_optimizer(self) -> AdamW:
        """
        Tạo optimizer với differential learning rates:
        - PhoBERT layers: lr thấp hơn (fine-tuning)
        - Classifier head: lr cao hơn (training from scratch)
        """
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']

        # Phân nhóm parameters
        phobert_params = []
        classifier_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'phobert' in name:
                phobert_params.append((name, param))
            else:
                classifier_params.append((name, param))

        optimizer_grouped_parameters = [
            # PhoBERT với weight decay
            {
                'params': [p for n, p in phobert_params if not any(nd in n for nd in no_decay)],
                'lr': self.config['phobert_lr'],
                'weight_decay': self.config.get('weight_decay', 0.01)
            },
            # PhoBERT không weight decay
            {
                'params': [p for n, p in phobert_params if any(nd in n for nd in no_decay)],
                'lr': self.config['phobert_lr'],
                'weight_decay': 0.0
            },
            # Classifier với lr cao hơn, weight decay
            {
                'params': [p for n, p in classifier_params if not any(nd in n for nd in no_decay)],
                'lr': self.config['classifier_lr'],
                'weight_decay': self.config.get('weight_decay', 0.01)
            },
            # Classifier không weight decay
            {
                'params': [p for n, p in classifier_params if any(nd in n for nd in no_decay)],
                'lr': self.config['classifier_lr'],
                'weight_decay': 0.0
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            eps=self.config.get('adam_epsilon', 1e-8),
            betas=(0.9, 0.999)
        )

        logger.info(f"Optimizer created:")
        logger.info(f"  PhoBERT LR: {self.config['phobert_lr']}")
        logger.info(f"  Classifier LR: {self.config['classifier_lr']}")

        return optimizer

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train 1 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1} [Train]",
            leave=False
        )

        for step, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass với AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        label_smoothing=self.config.get('label_smoothing', 0.1)
                    )
                    loss = outputs['loss'] / self.grad_accum_steps

                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    label_smoothing=self.config.get('label_smoothing', 0.1)
                )
                loss = outputs['loss'] / self.grad_accum_steps
                loss.backward()

            # Gradient accumulation
            if (step + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('max_grad_norm', 1.0)
                )

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1

            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{current_lr:.2e}'
            })

        avg_loss = total_loss / num_batches
        current_lr = self.scheduler.get_last_lr()[0]

        return {'loss': avg_loss, 'lr': current_lr}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str = "val") -> Dict[str, float]:
        """Evaluate trên val hoặc test set"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(loader, desc=f"[{split.upper()}]", leave=False)

        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

            loss = outputs['loss']
            logits = outputs['logits']

            total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(loader)
        metrics = self.metrics_calc.compute(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        metrics['loss'] = avg_loss

        return metrics

    def train(self) -> Dict:
        """
        Main training loop với early stopping
        """
        num_epochs = self.config['num_epochs']
        patience = self.config.get('early_stopping_patience', 5)

        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        logger.info(f"Early stopping patience: {patience}")

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.evaluate(self.val_loader, split="val")

            # Log
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val F1: {val_metrics['f1_macro']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {train_metrics['lr']:.2e}"
            )

            # Track history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f1'].append(val_metrics['f1_macro'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['learning_rates'].append(train_metrics['lr'])

            # Save best model
            if val_metrics['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_macro']
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(f"  ✅ New best model! F1: {self.best_val_f1:.4f}")
            else:
                self.patience_counter += 1
                self._save_checkpoint(epoch, val_metrics, is_best=False)
                logger.info(
                    f"  ⚠️  No improvement. Patience: {self.patience_counter}/{patience}"
                )

            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE in {total_time/60:.1f} minutes")
        logger.info(f"Best Val F1: {self.best_val_f1:.4f} at epoch {self.best_epoch+1}")
        logger.info("=" * 60)

        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Lưu checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_val_f1': self.best_val_f1,
        }

        # Luôn lưu checkpoint mới nhất
        last_path = self.output_dir / "checkpoint_last.pt"
        torch.save(checkpoint, last_path)

        # Lưu best model riêng
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"  💾 Best checkpoint saved: {best_path}")

    def load_best_checkpoint(self):
        """Load best checkpoint để evaluate hoặc inference"""
        best_path = self.output_dir / "checkpoint_best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {best_path}")

        checkpoint = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(
            f"Loaded best checkpoint from epoch {checkpoint['epoch']+1}, "
            f"F1: {checkpoint['metrics']['f1_macro']:.4f}"
        )
        return checkpoint['metrics']

