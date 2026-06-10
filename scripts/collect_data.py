"""
BƯỚC 0: Thu thập dữ liệu từ tất cả nguồn
Chạy trước scripts/prepare_data.py
"""
import sys
import yaml
import argparse
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collector import VietnameseDataCollector


def parse_args():
    parser = argparse.ArgumentParser(description="Collect training data")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["huggingface", "vnexpress", "youtube",
                 "telegram", "synthetic", "manual", "all"],
        default=["all"],
        help="Nguồn dữ liệu cần thu thập"
    )
    parser.add_argument(
        "--export-review",
        action="store_true",
        help="Export file Excel để human review"
    )
    parser.add_argument(
        "--review-batch",
        type=str,
        default="batch1",
        help="Tên batch review"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Chỉ in thống kê data hiện có"
    )
    return parser.parse_args()


def print_stats(df: pd.DataFrame):
    """In thống kê chi tiết về dataset"""
    logger.info("\n" + "=" * 60)
    logger.info("📊 DATASET STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"\nBy label:")
    for label_name, count in df['label_name'].value_counts().items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        logger.info(f"  {label_name:8} | {bar:<25} | {count:5d} ({pct:.1f}%)")

    if 'source' in df.columns:
        logger.info(f"\nBy source:")
        for source, count in df['source'].value_counts().items():
            logger.info(f"  {source:30} | {count:5d}")

    logger.info(f"\nText length stats:")
    lengths = df['text'].str.len()
    logger.info(f"  Min:    {lengths.min()}")
    logger.info(f"  Max:    {lengths.max()}")
    logger.info(f"  Mean:   {lengths.mean():.1f}")
    logger.info(f"  Median: {lengths.median():.1f}")
    logger.info("=" * 60)


def main():
    args = parse_args()

    # Load config
    with open("configs/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Setup logging
    log_dir = Path(config['paths'].get('log_dir', 'logs'))
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "collect_data.log", rotation="10 MB")

    logger.info("=" * 60)
    logger.info("STEP 0: DATA COLLECTION")
    logger.info("=" * 60)

    raw_dir = Path(config['paths']['raw_dir'])
    collector = VietnameseDataCollector(
        data_dir=str(raw_dir),
        config=config
    )

    # ── Stats only mode ──────────────────────────────────
    if args.stats_only:
        all_raw = raw_dir / "all_collected.csv"
        if all_raw.exists():
            df = pd.read_csv(all_raw)
            print_stats(df)
        else:
            logger.warning("No collected data found. Run without --stats-only first.")
        return

    # ── Override enabled sources từ CLI ──────────────────
    if "all" not in args.sources:
        config.setdefault('collection', {})['enabled_sources'] = args.sources
        collector.config = config
        collector.collection_cfg = config['collection']

    # ── Collect ──────────────────────────────────────────
    logger.info(f"Enabled sources: {config.get('collection', {}).get('enabled_sources', ['seed'])}")
    df = collector.collect_all()

    # ── Save ─────────────────────────────────────────────
    out_path = raw_dir / "all_collected.csv"
    df.to_csv(out_path, index=False, encoding='utf-8')
    logger.info(f"\n💾 Saved {len(df)} samples to {out_path}")

    # ── Stats ────────────────────────────────────────────
    print_stats(df)

    # ── Export for review ────────────────────────────────
    if args.export_review:
        logger.info(f"\n📋 Exporting for human review (batch: {args.review_batch})...")
        review_path = collector.export_for_review(
            df,
            batch_name=args.review_batch,
            sample_per_label=200
        )
        logger.info(f"Review file: {review_path}")
        logger.info("👉 Mở file Excel, điền cột 'human_label' rồi chạy lại với --sources manual")

    logger.info("\n✅ Collection complete!")
    logger.info("👉 Tiếp theo: python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
