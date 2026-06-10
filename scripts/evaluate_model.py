"""
Evaluate model đã train và export metrics
"""
import sys
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import ToxicSpamPredictor


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    logger.add("logs/evaluate.log")
    logger.info("=" * 50)
    logger.info("STEP 3: MODEL EVALUATION")
    logger.info("=" * 50)

    checkpoint_path = Path(config['paths']['checkpoint_dir']) / "checkpoint_best.pt"

    # Load predictor
    predictor = ToxicSpamPredictor(
        checkpoint_path=str(checkpoint_path),
        model_name=config['model']['name'],
        confidence_threshold=config['inference']['confidence_threshold'],
        toxic_threshold=config['inference']['toxic_threshold'],
        spam_threshold=config['inference']['spam_threshold']
    )

    # Load test set
    test_df = pd.read_csv(
        Path(config['paths']['processed_dir']) / "test.csv"
    )

    texts  = test_df['processed_text'].tolist()
    labels = test_df['label'].tolist()

    # Predict
    logger.info(f"Predicting {len(texts)} test samples...")
    results = predictor.predict_batch(texts, batch_size=32)

    preds = [r['label'] for r in results]

    # Metrics
    from src.training.metrics import MetricsCalculator
    calc = MetricsCalculator(num_classes=3)
    metrics = calc.compute(np.array(labels), np.array(preds))
    calc.print_report(np.array(labels), np.array(preds))

    # Save metrics
    metrics_path = Path(config['paths']['checkpoint_dir']) / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Demo predictions
    demo_texts = [
        "ĐỊt mẹ súc vật",
    ]

    logger.info("\n🧪 Demo predictions:")
    for text in demo_texts:
        result = predictor.predict_single(text)
        logger.info(
            f"  {result['emoji']} [{result['prediction'].upper():6}] "
            f"(conf={result['confidence']:.3f}) | {text[:50]}"
        )


if __name__ == "__main__":
    main()
