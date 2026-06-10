"""
Tiền xử lý văn bản tiếng Việt
"""
import re
import unicodedata
import pandas as pd
import numpy as np
from typing import List, Optional
from loguru import logger
from underthesea import word_tokenize


class VietnameseTextPreprocessor:
    """
    Tiền xử lý văn bản tiếng Việt:
    - Normalize unicode
    - Xử lý emoji, ký tự đặc biệt
    - Word segmentation với underthesea
    - Chuẩn hóa leet speak (1337) tiếng Việt
    """

    # Leet speak mapping phổ biến trong tiếng Việt
    LEET_MAP = {
        '4': 'a', '@': 'a', '3': 'e', '1': 'i', '!': 'i',
        '0': 'o', '5': 's', '7': 't', '+': 't', '$': 's',
        'đ': 'd', 'Đ': 'd',
    }

    # Từ viết tắt tục phổ biến cần normalize
    ABBREVIATION_MAP = {
        'dm': 'đ*t m*', 'đm': 'đ*t m*', 'vl': 'v*i l*n',
        'vcl': 'v*i c*i l*n', 'cl': 'c*i l*n', 'cc': 'c*c',
        'đcm': 'đ*t c*i m*', 'dcm': 'đ*t c*i m*',
        'lol': 'laugh out loud', 'wtf': 'what the f*ck',
    }

    def __init__(self, use_word_segment: bool = True):
        self.use_word_segment = use_word_segment

    def normalize_unicode(self, text: str) -> str:
        """Chuẩn hóa unicode về NFC"""
        return unicodedata.normalize('NFC', text)

    def remove_extra_spaces(self, text: str) -> str:
        """Xóa khoảng trắng thừa"""
        return re.sub(r'\s+', ' ', text).strip()

    def normalize_repeated_chars(self, text: str) -> str:
        """
        Chuẩn hóa ký tự lặp: 'nggguuu' -> 'ngu', 'đẹpppp' -> 'đẹp'
        """
        return re.sub(r'(.)\1{2,}', r'\1\1', text)

    def remove_urls(self, text: str) -> str:
        """Xóa URL"""
        return re.sub(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            ' [URL] ', text
        )

    def remove_emails(self, text: str) -> str:
        """Xóa email"""
        return re.sub(r'\S+@\S+', ' [EMAIL] ', text)

    def normalize_phone_numbers(self, text: str) -> str:
        """Normalize số điện thoại"""
        return re.sub(r'(\+84|0)[0-9]{8,10}', ' [PHONE] ', text)

    def handle_emoji(self, text: str) -> str:
        """Xử lý emoji - giữ lại nhưng thêm space"""
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F9FF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub(r' [EMOJI] ', text)

    def normalize_leet_speak(self, text: str) -> str:
        """Chuyển đổi leet speak về dạng thường"""
        result = []
        for char in text:
            result.append(self.LEET_MAP.get(char, char))
        return ''.join(result)

    def lowercase_and_normalize(self, text: str) -> str:
        """Lowercase và normalize"""
        return text.lower().strip()

    def word_segment(self, text: str) -> str:
        """
        Tách từ tiếng Việt với underthesea.
        PhoBERT được train với văn bản đã tách từ.
        VD: 'học sinh' -> 'học_sinh'
        """
        try:
            segmented = word_tokenize(text, format="text")
            return segmented
        except Exception as e:
            logger.warning(f"Word segmentation failed: {e}, using original text")
            return text

    def preprocess(self, text: str, for_training: bool = True) -> str:
        """
        Pipeline tiền xử lý đầy đủ
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Normalize unicode
        text = self.normalize_unicode(text)

        # 2. Lowercase
        text = self.lowercase_and_normalize(text)

        # 3. Remove URLs, emails, phones
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.normalize_phone_numbers(text)

        # 4. Handle emoji
        text = self.handle_emoji(text)

        # 5. Normalize leet speak
        text = self.normalize_leet_speak(text)

        # 6. Normalize repeated chars
        text = self.normalize_repeated_chars(text)

        # 7. Remove extra spaces
        text = self.remove_extra_spaces(text)

        # 8. Word segmentation (quan trọng cho PhoBERT)
        if self.use_word_segment:
            text = self.word_segment(text)

        return text

    def preprocess_batch(self, texts: List[str], for_training: bool = True) -> List[str]:
        """Xử lý batch"""
        processed = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.info(f"Preprocessing {i}/{len(texts)}...")
            processed.append(self.preprocess(text, for_training))
        return processed


class DataSplitter:
    """Chia dataset train/val/test"""

    def __init__(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

    def split(self, df: pd.DataFrame) -> tuple:
        """
        Stratified split để đảm bảo phân phối nhãn đều
        """
        from sklearn.model_selection import train_test_split

        # Tách test trước
        train_val, test = train_test_split(
            df,
            test_size=self.test_ratio,
            stratify=df['label'],
            random_state=self.random_state
        )

        # Tách train/val từ phần còn lại
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio_adjusted,
            stratify=train_val['label'],
            random_state=self.random_state
        )

        logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        logger.info(f"Train distribution:\n{train['label_name'].value_counts()}")

        return (
            train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True)
        )
