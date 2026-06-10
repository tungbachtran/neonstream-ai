"""
Thu thập data từ HuggingFace Datasets
Nguồn: ViHSD, ViHOS, ViCTSD và các dataset tiếng Việt khác
"""
import pandas as pd
from loguru import logger
from typing import List, Dict, Optional


class HuggingFaceCollector:
    """
    Load các dataset NLP tiếng Việt từ HuggingFace Hub.

    Các dataset được hỗ trợ:
    - tarudesu/ViHSD  : Vietnamese Hate Speech Detection (CLEAN/OFFENSIVE/HATE)
    - tarudesu/ViHOS  : Vietnamese Hate & Offensive Speech
    - uitnlp/...      : Các dataset từ UIT NLP Lab
    """

    def __init__(self, datasets_config: List[Dict]):
        """
        datasets_config: list các dict từ config.yaml
        [
          {name, text_col, label_col, label_remap, split},
          ...
        ]
        """
        self.datasets_config = datasets_config

    def load_single(self, cfg: Dict) -> pd.DataFrame:
        """Load 1 dataset theo config"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        logger.info(f"Loading HuggingFace dataset: {cfg['name']} ...")

        try:
            ds = load_dataset(cfg['name'], split=cfg.get('split', 'train'))
            max_samples = cfg.get('max_samples', None)
            if max_samples and len(ds) > max_samples:
                ds = ds.shuffle(seed=42).select(range(max_samples))
                logger.info(f"  Sampled {max_samples:,} from {len(ds):,} total rows")
            df = ds.to_pandas()

            text_col  = cfg['text_col']
            label_col = cfg['label_col']

            # Kiểm tra columns tồn tại
            if text_col not in df.columns:
                logger.warning(f"Column '{text_col}' not found in {cfg['name']}. "
                               f"Available: {df.columns.tolist()}")
                return pd.DataFrame()

            result = pd.DataFrame()
            result['text'] = df[text_col].astype(str)

            # Remap labels
            label_remap = cfg.get('label_remap', {})
            if label_remap:
                # Key trong yaml là string, cần convert sang int
                label_remap_int = {int(k): int(v) for k, v in label_remap.items()}
                result['label'] = df[label_col].map(label_remap_int)
            else:
                result['label'] = df[label_col]

            # Thêm label_name
            label_name_map = {0: 'clean', 1: 'toxic', 2: 'spam', 3: 'adult'}
            result['label_name'] = result['label'].map(label_name_map)
            result['source'] = cfg['name'].replace('/', '_')

            # Xóa rows có label NaN (do remap không khớp)
            before = len(result)
            result = result.dropna(subset=['label', 'text']).reset_index(drop=True)
            result['label'] = result['label'].astype(int)

            logger.info(f"  Loaded {len(result)} samples "
                        f"(dropped {before - len(result)} unmapped)")
            logger.info(f"  Distribution: {result['label_name'].value_counts().to_dict()}")

            return result

        except Exception as e:
            logger.error(f"Failed to load {cfg['name']}: {e}")
            return pd.DataFrame()

    def collect_all(self) -> pd.DataFrame:
        """Load tất cả datasets trong config"""
        dfs = []
        for cfg in self.datasets_config:
            df = self.load_single(cfg)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            logger.warning("No HuggingFace datasets loaded!")
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"HuggingFace total: {len(combined)} samples")
        return combined
