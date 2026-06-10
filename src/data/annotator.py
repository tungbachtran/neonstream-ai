"""
Auto-labeling và export để human review.
Quan trọng nhất với adult content.
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Tuple
from copy import deepcopy


class AutoLabeler:
    """
    Gán nhãn tự động dựa trên keyword matching.
    Kết quả là WEAK LABELS — cần human verify.
    """

    # Từ khóa theo nhãn — bổ sung thêm theo thực tế
    KEYWORDS: Dict[str, List[str]] = {
        'toxic': [
            'đm', 'vcl', 'đcm', 'vl', 'ngu', 'khốn', 'súc vật',
            'vô dụng', 'rác rưởi', 'cút', 'biến', 'câm mồm',
            'thằng chó', 'con chó', 'đồ điên', 'mày tưởng',
        ],
        'spam': [
            'inbox ngay', 'liên hệ ngay', 'order ngay', 'đặt hàng ngay',
            'giảm giá', 'flash sale', 'free ship', 'kiếm tiền',
            'không cần vốn', 'thu nhập', 'tuyển ctv', 'làm tại nhà',
            'lh:', 'zalo:', 'sdt:', '0909', '0898', '0978',
            'giảm %', 'sale off', 'hàng chính hãng giá rẻ',
        ],
        'adult': [
            'sex', 'chịch', 'địt', 'bú', 'liếm', 'cu', 'lồn', 'chim', 'bướm', 'cặc', 'fuck', 'pussy','làm tình', 'quan hệ', 'quan hệ tình dục', 'chơi sex', 'xem sex', 'phim sex', 'xem phim sex', 'tình dục', 'dâm', 'dâm dục',
            'khiêu dâm', 'nude', '18+', 'porn', 'phim porn', 'gái xinh', 'gái gọi', 'massage', 'escort', 'thú vui','xem ngay', 'xem miễn phí', '18+ miễn phí', 'video sex', 'ảnh sex', 'live sex', 'nứng'
        ],
    }

    def __init__(self, custom_keywords: Optional[Dict[str, List[str]]] = None):
        self.keywords = deepcopy(self.KEYWORDS)
        if custom_keywords:
            for label, kws in custom_keywords.items():
                self.keywords.setdefault(label, []).extend(kws)

    def _check_label(self, text: str, label: str) -> Tuple[bool, List[str]]:
        """Kiểm tra text có thuộc nhãn không, trả về matched keywords"""
        text_lower = text.lower()
        matched = [kw for kw in self.keywords.get(label, []) if kw in text_lower]
        return len(matched) > 0, matched

    def label_single(self, text: str) -> Dict:
        """Auto-label 1 text"""
        results = {}
        for label in ['toxic', 'spam', 'adult']:
            matched, kws = self._check_label(text, label)
            results[label] = {'matched': matched, 'keywords': kws}

        # Priority: adult > toxic > spam > clean
        if results['adult']['matched']:
            return {'auto_label': 3, 'auto_label_name': 'adult',
                    'confidence': 'low', 'matched_kws': results['adult']['keywords']}
        elif results['toxic']['matched']:
            return {'auto_label': 1, 'auto_label_name': 'toxic',
                    'confidence': 'medium', 'matched_kws': results['toxic']['keywords']}
        elif results['spam']['matched']:
            return {'auto_label': 2, 'auto_label_name': 'spam',
                    'confidence': 'medium', 'matched_kws': results['spam']['keywords']}
        else:
            return {'auto_label': 0, 'auto_label_name': 'clean',
                    'confidence': 'high', 'matched_kws': []}

    def label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-label toàn bộ DataFrame"""
        results = df['text'].apply(self.label_single)
        df['auto_label'] = results.apply(lambda x: x['auto_label'])
        df['auto_label_name'] = results.apply(lambda x: x['auto_label_name'])
        df['auto_confidence'] = results.apply(lambda x: x['confidence'])
        df['matched_keywords'] = results.apply(lambda x: str(x['matched_kws']))
        return df


class HumanReviewExporter:
    """
    Export data để human annotator review.
    Tạo file Excel với format dễ dùng.
    """

    def __init__(self, output_dir: str = "data/raw/manual"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_for_review(
        self,
        df: pd.DataFrame,
        batch_name: str = "batch1",
        sample_per_label: int = 200,
        priority_low_confidence: bool = True
    ) -> str:
        """
        Export file Excel để human review.

        Ưu tiên các samples có auto_confidence = 'low'
        vì đây là những cases model không chắc chắn.
        """
        review_dfs = []

        for label in [0, 1, 2, 3]:
            label_df = df[df.get('auto_label', df.get('label')) == label].copy()
            if label_df.empty:
                continue

            # Ưu tiên low confidence
            if priority_low_confidence and 'auto_confidence' in label_df.columns:
                low_conf = label_df[label_df['auto_confidence'] == 'low']
                high_conf = label_df[label_df['auto_confidence'] != 'low']

                n_low = min(len(low_conf), sample_per_label // 2)
                n_high = min(len(high_conf), sample_per_label - n_low)

                sampled = pd.concat([
                    low_conf.sample(n=n_low, random_state=42) if n_low > 0 else pd.DataFrame(),
                    high_conf.sample(n=n_high, random_state=42) if n_high > 0 else pd.DataFrame(),
                ])
            else:
                n = min(len(label_df), sample_per_label)
                sampled = label_df.sample(n=n, random_state=42)

            review_dfs.append(sampled)

        if not review_dfs:
            logger.warning("No data to export for review!")
            return ""

        review_df = pd.concat(review_dfs, ignore_index=True)
        review_df = review_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Thêm cột cho annotator
        review_df['human_label'] = ''       # Annotator điền: 0/1/2/3
        review_df['human_label_name'] = ''  # clean/toxic/spam/adult
        review_df['is_correct'] = ''        # yes/no
        review_df['notes'] = ''

        # Chọn columns hiển thị
        display_cols = ['text', 'auto_label_name', 'auto_confidence',
                        'matched_keywords', 'human_label', 'human_label_name',
                        'is_correct', 'notes']
        display_cols = [c for c in display_cols if c in review_df.columns]

        out_path = self.output_dir / f"review_{batch_name}.xlsx"

        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            review_df[display_cols].to_excel(
                writer, sheet_name='Review', index=True
            )
            # Sheet hướng dẫn
            guide = pd.DataFrame({
                'Label': [0, 1, 2, 3],
                'Name': ['clean', 'toxic', 'spam', 'adult'],
                'Description': [
                    'Bình luận bình thường, không vi phạm',
                    'Ngôn ngữ tục tĩu, xúc phạm, thù địch',
                    'Quảng cáo, spam, rao vặt không phù hợp',
                    'Nội dung khiêu dâm, 18+'
                ]
            })
            guide.to_excel(writer, sheet_name='Guide', index=False)

        logger.info(f"Exported {len(review_df)} samples for review: {out_path}")
        return str(out_path)

    def load_reviewed(self, file_path: str) -> pd.DataFrame:
        """Load file đã được human annotate"""
        df = pd.read_excel(file_path, sheet_name='Review')

        # Lấy human label nếu có, fallback về auto label
        if 'human_label' in df.columns:
            mask = df['human_label'].notna() & (df['human_label'] != '')
            df.loc[mask, 'label'] = df.loc[mask, 'human_label'].astype(int)
            df.loc[~mask, 'label'] = df.loc[~mask, 'auto_label']
        else:
            df['label'] = df['auto_label']

        label_name_map = {0: 'clean', 1: 'toxic', 2: 'spam', 3: 'adult'}
        df['label_name'] = df['label'].map(label_name_map)
        df['source'] = 'manual_reviewed'

        # Chỉ lấy rows có text hợp lệ
        df = df[df['text'].notna() & (df['text'].str.len() > 0)].reset_index(drop=True)

        logger.info(f"Loaded {len(df)} reviewed samples from {file_path}")
        logger.info(f"Distribution: {df['label_name'].value_counts().to_dict()}")
        return df[['text', 'label', 'label_name', 'source']]
