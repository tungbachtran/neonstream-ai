"""
Inference engine - load model và predict
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
from transformers import AutoTokenizer
from loguru import logger
import time

from src.models.phobert_classifier import PhoBERTClassifier
from src.data.preprocessor import VietnameseTextPreprocessor


class ToxicSpamPredictor:
    """
    Production-ready predictor với:
    - Model loading từ checkpoint
    - Batch inference
    - Confidence thresholds
    - Caching tokenizer
    - Detailed result output
    """

    # LABEL_MAP  = {0: "clean", 1: "toxic", 2: "spam", 3: "adult"}
    # LABEL_EMOJI = {0: "✅", 1: "🚫", 2: "📢", 3: "🔞"} 
    LABEL_MAP  = {0: "clean", 1: "toxic", 2: "spam", }
    LABEL_EMOJI = {0: "✅", 1: "🚫", 2: "📢",} 

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "vinai/phobert-base",
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        toxic_threshold: float = 0.4,
        spam_threshold: float = 0.4,
        adult_threshold: float = 0.4
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.thresholds = {
            0: 1.0 - toxic_threshold - spam_threshold,  # clean
            1: toxic_threshold,
            2: spam_threshold,
            #3: adult_threshold
        }

        # Device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Predictor using device: {self.device}")

        # Load components
        self.preprocessor = VietnameseTextPreprocessor(use_word_segment=True)
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        logger.info("✅ Predictor ready!")

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load PhoBERT tokenizer"""
        logger.info(f"Loading tokenizer from {self.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False  # PhoBERT dùng slow tokenizer
        )
        return tokenizer

    def _load_model(self) -> PhoBERTClassifier:
        """Load model từ checkpoint"""
        logger.info(f"Loading model from {self.checkpoint_path}...")

        # Khởi tạo model architecture
        model = PhoBERTClassifier(
            model_name=self.model_name,
            num_labels=3,
            dropout_rate=0.1,  # Thấp hơn khi inference
            use_attention_pooling=True
        )

        # Load weights
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Best F1: {checkpoint.get('best_val_f1', 'unknown')}")

        return model

    def _tokenize(
        self,
        texts: List[str],
        max_length: int = 256
    ) -> Dict[str, torch.Tensor]:
        """Tokenize batch of texts"""
        encoding = self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False
        )
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }

    @torch.no_grad()
    def predict_single(self, text: str) -> Dict:
        """
        Predict 1 text, trả về kết quả chi tiết
        """
        start_time = time.time()

        # Preprocess
        processed_text = self.preprocessor.preprocess(text)
        if not processed_text:
            return self._empty_result(text, "empty_text")

        # Tokenize
        inputs = self._tokenize([processed_text])

        # Inference
        outputs = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

        logits = outputs['logits'][0]
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        # Determine prediction
        pred_label = int(np.argmax(probs))
        pred_name = self.LABEL_MAP[pred_label]
        confidence = float(probs[pred_label])

        # Apply custom thresholds
        # Nếu toxic/spam score vượt threshold thì flag
        is_flagged = (
            probs[1] >= self.thresholds[1] or   # toxic
            probs[2] >= self.thresholds[2] #or   # spam
           # probs[3] >= self.thresholds[3]       # ← THÊM adult
        )

        inference_time = (time.time() - start_time) * 1000  # ms

        return {
            "text": text,
            "processed_text": processed_text,
            "prediction": pred_name,
            "label": pred_label,
            "confidence": round(confidence, 4),
            "is_flagged": is_flagged,
            "scores": {
                "clean": round(float(probs[0]), 4),
                "toxic": round(float(probs[1]), 4),
                "spam":  round(float(probs[2]), 4),
                #"adult": round(float(probs[3]), 4),  # ← THÊM
            },
            "emoji": self.LABEL_EMOJI[pred_label],
            "inference_time_ms": round(inference_time, 2)
        }

    @torch.no_grad()
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 256
    ) -> List[Dict]:
        """
        Predict batch of texts hiệu quả
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Preprocess
            processed = [self.preprocessor.preprocess(t) for t in batch_texts]

            # Filter empty
            valid_indices = [j for j, t in enumerate(processed) if t]
            if not valid_indices:
                results.extend([self._empty_result(t) for t in batch_texts])
                continue

            valid_texts = [processed[j] for j in valid_indices]

            # Tokenize
            inputs = self._tokenize(valid_texts, max_length)

            # Inference
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            logits = outputs['logits']
            probs = F.softmax(logits, dim=-1).cpu().numpy()

            # Build results
            batch_results = [None] * len(batch_texts)
            for k, j in enumerate(valid_indices):
                p = probs[k]
                pred_label = int(np.argmax(p))
                is_flagged = (
                    p[1] >= self.thresholds[1] or
                    p[2] >= self.thresholds[2]
                )
                batch_results[j] = {
                    "text": batch_texts[j],
                    "prediction": self.LABEL_MAP[pred_label],
                    "label": pred_label,
                    "confidence": round(float(p[pred_label]), 4),
                    "is_flagged": is_flagged,
                    "scores": {
                        "clean": round(float(p[0]), 4),
                        "toxic": round(float(p[1]), 4),
                        "spam": round(float(p[2]), 4)
                    }
                }

            # Fill empty results
            for j in range(len(batch_texts)):
                if batch_results[j] is None:
                    batch_results[j] = self._empty_result(batch_texts[j])

            results.extend(batch_results)

        return results

    def _empty_result(self, text: str, reason: str = "processing_error") -> Dict:
        return {
            "text": text,
            "prediction": "clean",
            "label": 0,
            "confidence": 0.0,
            "is_flagged": False,
            "scores": {"clean": 1.0, "toxic": 0.0, "spam": 0.0},
            "error": reason
        }
