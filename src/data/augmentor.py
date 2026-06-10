"""
Data Augmentation cho tiếng Việt
Tăng cường dữ liệu để model robust hơn
"""
import random
import pandas as pd
import numpy as np
from typing import List, Tuple
from loguru import logger
from copy import deepcopy


class VietnameseDataAugmentor:
    """
    Các kỹ thuật augmentation cho tiếng Việt:
    1. Random deletion - xóa ngẫu nhiên từ
    2. Random swap - hoán đổi vị trí từ
    3. Synonym replacement - thay thế từ đồng nghĩa
    4. Back translation simulation - mô phỏng back translation
    5. Noise injection - thêm nhiễu ký tự
    """

    # Từ đồng nghĩa đơn giản cho tiếng Việt
    SYNONYMS = {
        "đẹp": ["xinh", "dễ thương", "đáng yêu"],
        "xấu": ["tệ", "dở", "kém"],
        "tốt": ["hay", "giỏi", "xuất sắc"],
        "nhanh": ["mau", "lẹ", "tốc độ"],
        "chậm": ["từ từ", "thong thả"],
        "vui": ["hạnh phúc", "phấn khởi", "hớn hở"],
        "buồn": ["đau lòng", "thất vọng", "chán nản"],
        "mua": ["đặt hàng", "order", "lấy"],
        "bán": ["phân phối", "cung cấp"],
        "giúp": ["hỗ trợ", "assist", "giúp đỡ"],
    }

    def __init__(self, aug_prob: float = 0.3):
        self.aug_prob = aug_prob
        random.seed(42)
        np.random.seed(42)

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Xóa ngẫu nhiên các từ với xác suất p"""
        words = text.split()
        if len(words) <= 3:
            return text

        new_words = [w for w in words if random.random() > p]
        if not new_words:
            return random.choice(words)
        return ' '.join(new_words)

    def random_swap(self, text: str, n: int = 1) -> str:
        """Hoán đổi ngẫu nhiên n cặp từ"""
        words = text.split()
        if len(words) < 2:
            return text

        new_words = words.copy()
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        return ' '.join(new_words)

    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Thay thế n từ bằng từ đồng nghĩa"""
        words = text.split()
        new_words = words.copy()
        replaced = 0

        random.shuffle(words)
        for word in words:
            if word in self.SYNONYMS and replaced < n:
                synonym = random.choice(self.SYNONYMS[word])
                # Thay thế tất cả occurrence
                new_words = [synonym if w == word else w for w in new_words]
                replaced += 1

        return ' '.join(new_words)

    def add_noise(self, text: str, p: float = 0.05) -> str:
        """Thêm nhiễu ký tự - mô phỏng lỗi gõ phím"""
        result = []
        for char in text:
            if random.random() < p and char.isalpha():
                # Duplicate hoặc skip ký tự
                action = random.choice(['dup', 'skip', 'keep'])
                if action == 'dup':
                    result.append(char)
                    result.append(char)
                elif action == 'skip':
                    pass
                else:
                    result.append(char)
            else:
                result.append(char)
        return ''.join(result)

    def augment_dataset(
        self,
        df: pd.DataFrame,
        target_per_class: int = 500,
        augment_labels: List[int] = None
    ) -> pd.DataFrame:
        """
        Augment dataset để cân bằng classes.
        Hỗ trợ 4 nhãn: 0=clean, 1=toxic, 2=spam, 3=adult
        """
        # ← SỬA: tự detect số nhãn từ data thay vì hardcode
        if augment_labels is None:
            augment_labels = sorted(df['label'].unique().tolist())

        label_name_map = {0: 'clean', 1: 'toxic', 2: 'spam', 3: 'adult'}
        augmented_records = []

        for label in augment_labels:
            label_df = df[df['label'] == label]
            current_count = len(label_df)

            # Lấy label_name an toàn
            if 'label_name' in label_df.columns and len(label_df) > 0:
                label_name = label_df['label_name'].iloc[0]
            else:
                label_name = label_name_map.get(label, str(label))

            logger.info(
                f"Label {label} ({label_name}): "
                f"{current_count} -> target {target_per_class}"
            )

            if current_count >= target_per_class:
                sampled = label_df.sample(n=target_per_class, random_state=42)
                augmented_records.append(sampled)
                continue

            needed = target_per_class - current_count
            augmented_records.append(label_df)

            texts = label_df['text'].tolist()
            new_samples = []

            while len(new_samples) < needed:
                text = random.choice(texts)
                variants = self.augment_text(text)
                for v in variants:
                    if len(new_samples) >= needed:
                        break
                    new_samples.append({
                        "text": v,
                        "label": label,
                        "label_name": label_name,
                        "augmented": True,
                        "source": "augmented"
                    })

            aug_df = pd.DataFrame(new_samples)
            augmented_records.append(aug_df)
            logger.info(f"  Added {len(new_samples)} augmented samples")

        result = pd.concat(augmented_records, ignore_index=True)
        result = result.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Final augmented dataset: {len(result)} samples")
        return result
