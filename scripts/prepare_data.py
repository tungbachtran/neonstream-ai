"""
Script chuẩn bị dữ liệu: collect -> preprocess -> augment -> split -> save
"""
import sys
import yaml
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collector import VietnameseDataCollector
from src.data.preprocessor import VietnameseTextPreprocessor, DataSplitter
from src.data.augmentor import VietnameseDataAugmentor


def main():
    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    logger.add("logs/prepare_data.log", rotation="10 MB")
    logger.info("=" * 50)
    logger.info("STEP 1: DATA PREPARATION")
    logger.info("=" * 50)
    raw_dir = Path(config['paths']['raw_dir'])
    # ── 1. Load collected data (ưu tiên file đã collect) ──
    collected_path = raw_dir / "all_collected.csv"
    if collected_path.exists():
        logger.info(f"\n📦 Loading pre-collected data from {collected_path}...")
        raw_df = pd.read_csv(collected_path)
        logger.info(f"Loaded {len(raw_df)} samples")
    else:
        # Fallback: collect on-the-fly (tương thích code cũ)
        logger.info("\n📦 No pre-collected data found, collecting now...")
        collector = VietnameseDataCollector(
            data_dir=str(raw_dir),
            config=config
        )
        raw_df = collector.collect_all()

    logger.info(f"Raw data: {len(raw_df)} samples")
    logger.info(f"Distribution:\n{raw_df['label_name'].value_counts()}")


    # ── 2. Preprocess ────────────────────────────────────
    logger.info("\n🔧 Preprocessing text...")
    preprocessor = VietnameseTextPreprocessor(
        use_word_segment=config['data']['use_word_segment']
    )

    raw_df['processed_text'] = preprocessor.preprocess_batch(
        raw_df['text'].tolist()
    )

    # Xóa empty sau preprocessing
    before = len(raw_df)
    raw_df = raw_df[raw_df['processed_text'].str.len() > 0].reset_index(drop=True)
    logger.info(f"After preprocessing: {len(raw_df)} samples (removed {before - len(raw_df)} empty)")

    # Save processed
    processed_dir = Path(config['paths']['processed_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(processed_dir / "all_processed.csv", index=False, encoding='utf-8')

    # ── 3. Split ─────────────────────────────────────────
    logger.info("\n✂️  Splitting dataset...")
    splitter = DataSplitter(
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio']
    )
    train_df, val_df, test_df = splitter.split(raw_df)

    # ── 4. Augment train set ──────────────────────────────
    if config['data']['use_augmentation']:
        logger.info("\n🔁 Augmenting training data...")
        augmentor = VietnameseDataAugmentor()
        train_df = augmentor.augment_dataset(
            train_df,
            target_per_class=config['data']['target_per_class']
        )

    # ── 5. Save splits ────────────────────────────────────
    logger.info("\n💾 Saving splits...")
    train_df.to_csv(processed_dir / "train.csv", index=False, encoding='utf-8')
    val_df.to_csv(processed_dir / "val.csv", index=False, encoding='utf-8')
    test_df.to_csv(processed_dir / "test.csv", index=False, encoding='utf-8')

    logger.info("\n✅ Data preparation complete!")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Val:   {len(val_df)} samples")
    logger.info(f"  Test:  {len(test_df)} samples")
    logger.info(f"  Saved to: {processed_dir}")


if __name__ == "__main__":
    main()
