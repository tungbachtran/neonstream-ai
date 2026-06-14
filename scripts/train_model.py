"""
Script train model PhoBERT
"""
import sys
import yaml
import pandas as pd
import torch
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import create_dataloaders
from src.models.phobert_classifier import PhoBERTClassifier
from src.training.trainer import PhoBERTTrainer


def main():
    # ── Config ───────────────────────────────────────────
    with open("configs/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.add("logs/training.log", rotation="50 MB")
    logger.info("=" * 60)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("=" * 60)

    # ── Load data ────────────────────────────────────────
    processed_dir = Path(config['paths']['processed_dir'])
    logger.info("Loading datasets...")
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df   = pd.read_csv(processed_dir / "val.csv")
    test_df  = pd.read_csv(processed_dir / "test.csv")

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Tokenizer ────────────────────────────────────────
    model_name = config['model']['name']
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # ── DataLoaders ──────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['model']['max_length'],
        num_workers=config['training']['num_workers']
    )

    # ── Model ────────────────────────────────────────────
    logger.info("Initializing PhoBERT model...")
    model = PhoBERTClassifier(
        model_name=model_name,
        num_labels=config['model']['num_labels'],
        dropout_rate=config['model']['dropout_rate'],
        use_attention_pooling=config['model']['use_attention_pooling'],
        num_dropout_samples=config['model']['num_dropout_samples'],
        freeze_layers=config['model']['freeze_layers']
    )

    # ── Trainer ──────────────────────────────────────────
    train_config = {
        **config['training'],
        'phobert_lr':     config['training']['phobert_lr'],
        'classifier_lr':  config['training']['classifier_lr'],
    }

    trainer = PhoBERTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        output_dir=config['paths']['checkpoint_dir']
    )

    # ── Train ────────────────────────────────────────────
    history = trainer.train()

    # ── Final evaluation on test set ─────────────────────
    logger.info("\n📊 Final evaluation on TEST set...")
    trainer.load_best_checkpoint()
    test_metrics = trainer.evaluate(test_loader, split="test")
    trainer.metrics_calc.print_report(
        *_get_preds(trainer, test_loader)
    )

    logger.info("\n🎯 TEST RESULTS:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    logger.info("\n✅ Training complete!")
    logger.info(f"Best checkpoint: {config['paths']['checkpoint_dir']}/checkpoint_best.pt")


def _get_preds(trainer, loader):
    """Helper lấy y_true, y_pred cho report"""
    import numpy as np
    trainer.model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(trainer.device)
            attention_mask = batch['attention_mask'].to(trainer.device)
            labels = batch['labels']
            outputs = trainer.model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs['logits'], dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


if __name__ == "__main__":
    main()
