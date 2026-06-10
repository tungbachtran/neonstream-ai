"""
PhoBERT Classifier cho Vietnamese Toxic/Spam Detection

Architecture:
PhoBERT Base → [CLS] token → Dropout → Linear → 3 classes
                           ↘ Multi-head Attention Pooling (optional)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoConfig,
    PreTrainedModel
)
from typing import Optional, Tuple, Dict
from loguru import logger


class AttentionPooling(nn.Module):
    """
    Attention-based pooling thay vì chỉ dùng [CLS] token.
    Giúp model tập trung vào các từ quan trọng hơn.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        hidden_states: (batch, seq_len, hidden_size)
        attention_mask: (batch, seq_len)
        Returns: (batch, hidden_size)
        """
        # Tính attention weights
        weights = self.attention(hidden_states)  # (batch, seq_len, 1)

        # Mask padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        weights = weights * mask

        # Normalize
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted sum
        pooled = (hidden_states * weights).sum(dim=1)  # (batch, hidden_size)
        return pooled


class PhoBERTClassifier(nn.Module):
    """
    PhoBERT-based classifier cho Vietnamese text classification.
    
    Features:
    - PhoBERT base/large backbone
    - Dual pooling: CLS + Attention pooling
    - Multi-sample dropout cho regularization
    - Label smoothing support
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_labels: int = 4,
        dropout_rate: float = 0.3,
        use_attention_pooling: bool = True,
        num_dropout_samples: int = 5,
        freeze_layers: int = 0
    ):
        super().__init__()

        self.num_labels = num_labels
        self.use_attention_pooling = use_attention_pooling
        self.num_dropout_samples = num_dropout_samples

        # Load PhoBERT
        logger.info(f"Loading PhoBERT from: {model_name}")
        self.config = AutoConfig.from_pretrained(model_name)
        self.phobert = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size  # 768 for base, 1024 for large

        # Freeze early layers nếu cần (transfer learning)
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

        # Pooling
        if use_attention_pooling:
            self.attention_pooling = AttentionPooling(hidden_size)
            classifier_input_size = hidden_size * 2  # CLS + Attention pooling
        else:
            classifier_input_size = hidden_size

        # Classifier head
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(classifier_input_size)

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, classifier_input_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(classifier_input_size // 2),
            nn.Linear(classifier_input_size // 2, num_labels)
        )

        # Initialize weights
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model initialized:")
        logger.info(f"  Total params: {total_params:,}")
        logger.info(f"  Trainable params: {trainable_params:,}")
        logger.info(f"  Hidden size: {hidden_size}")

    def _freeze_layers(self, num_layers: int):
        """Freeze n layers đầu của PhoBERT"""
        # Freeze embeddings
        for param in self.phobert.embeddings.parameters():
            param.requires_grad = False

        # Freeze n encoder layers
        for i, layer in enumerate(self.phobert.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        logger.info(f"Frozen {num_layers} encoder layers")

    def _init_weights(self):
        """Khởi tạo weights cho classifier head"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Returns dict với 'loss' (nếu có labels) và 'logits'
        """
        # PhoBERT encoding
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Lấy last hidden state
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # CLS token representation
        cls_output = sequence_output[:, 0, :]  # (batch, hidden)

        # Pooling
        if self.use_attention_pooling:
            attn_output = self.attention_pooling(sequence_output, attention_mask)
            pooled = torch.cat([cls_output, attn_output], dim=-1)
        else:
            pooled = cls_output

        # Layer norm
        pooled = self.layer_norm(pooled)

        # Multi-sample dropout (training only)
        if self.training and self.num_dropout_samples > 1:
            logits = torch.stack([
                self.classifier(self.dropout(pooled))
                for _ in range(self.num_dropout_samples)
            ]).mean(dim=0)
        else:
            logits = self.classifier(self.dropout(pooled))

        result = {"logits": logits}

        # Tính loss nếu có labels
        if labels is not None:
            if label_smoothing > 0:
                loss = self._label_smoothing_loss(logits, labels, label_smoothing)
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
            result["loss"] = loss

        return result

    def _label_smoothing_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        smoothing: float = 0.1
    ) -> torch.Tensor:
        """Label smoothing loss để tránh overconfident predictions"""
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # One-hot với smoothing
        with torch.no_grad():
            smooth_labels = torch.full_like(log_probs, smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)

        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        return loss

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference mode - trả về (predictions, probabilities)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs["logits"]
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs
