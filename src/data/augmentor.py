"""
src/data/augmentor.py
Vietnamese Data Augmentor — hỗ trợ 4 nhãn: Clean, Toxic, Spam, Adult
"""

import random
import re
import pandas as pd
from loguru import logger
from typing import List


class VietnameseDataAugmentor:
    """
    Augment dữ liệu tiếng Việt bằng các kỹ thuật:
    - Random word deletion
    - Random word swap
    - Synonym replacement
    - Noise injection (typo simulation)
    - Sentence shuffle (cho câu dài)
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

        # ── Từ điển đồng nghĩa tiếng Việt ──────────────────────────
        self.synonym_dict = {
            # Tích cực
            "đẹp":      ["xinh", "dễ thương", "đáng yêu", "lung linh"],
            "tốt":      ["hay", "ổn", "được", "tuyệt"],
            "vui":      ["vui vẻ", "hạnh phúc", "phấn khởi", "hào hứng"],
            "buồn":     ["u sầu", "chán", "thất vọng", "nản"],
            "nhanh":    ["mau", "lẹ", "tốc độ", "vội"],
            "chậm":     ["từ từ", "thong thả", "lề mề"],
            "nhiều":    ["đông", "dồi dào", "phong phú", "lắm"],
            "ít":       ["hiếm", "khan hiếm", "thiếu"],
            "lớn":      ["to", "khổng lồ", "đồ sộ", "rộng"],
            "nhỏ":      ["bé", "tí hon", "nhỏ bé", "mini"],
            "thích":    ["yêu", "mê", "ưa", "khoái"],
            "ghét":     ["không thích", "chán ghét", "khinh"],
            "biết":     ["hiểu", "nắm", "rõ"],
            "nói":      ["phát biểu", "chia sẻ", "kể", "trình bày"],
            "làm":      ["thực hiện", "tiến hành", "xử lý"],
            "đi":       ["di chuyển", "bước", "chạy"],
            "xem":      ["nhìn", "quan sát", "theo dõi"],
            "ăn":       ["dùng bữa", "thưởng thức", "nhâm nhi"],
            "học":      ["nghiên cứu", "tìm hiểu", "ôn"],
            "chơi":     ["giải trí", "vui chơi", "thư giãn"],
            # Toxic synonyms (để augment toxic data)
            "ngu":      ["đần", "khờ", "dốt", "ngốc"],
            "xấu":      ["tệ", "dở", "kém", "tồi"],
            "chán":     ["nhàm", "tẻ", "vô vị"],
            # Spam synonyms
            "kiếm":     ["thu", "nhận", "có"],
            "tiền":     ["thu nhập", "lợi nhuận", "doanh thu"],
            "mua":      ["đặt hàng", "sở hữu", "lấy"],
            "bán":      ["cung cấp", "phân phối", "giao"],
        }

        # ── Stop words — không xóa/swap những từ này ────────────────
        self.stop_words = {
            "và", "hoặc", "nhưng", "vì", "nên", "thì", "mà",
            "là", "có", "không", "được", "cho", "với", "của",
            "trong", "ngoài", "trên", "dưới", "tôi", "bạn",
            "anh", "chị", "em", "họ", "chúng", "các", "những",
        }

    # ════════════════════════════════════════════════════════════════
    # PUBLIC: augment_text — nhận 1 câu, trả về list các biến thể
    # ════════════════════════════════════════════════════════════════
    def augment_text(self, text: str, num_variants: int = 3) -> List[str]:
        """
        Tạo ra `num_variants` biến thể augmented từ 1 câu gốc.
        Mỗi biến thể dùng 1 kỹ thuật khác nhau.
        """
        if not text or len(text.strip()) == 0:
            return []

        variants = []
        techniques = [
            self._random_deletion,
            self._random_swap,
            self._synonym_replacement,
            self._noise_injection,
        ]

        # Shuffle để đa dạng kỹ thuật
        random.shuffle(techniques)

        for i, technique in enumerate(techniques):
            if len(variants) >= num_variants:
                break
            try:
                result = technique(text)
                # Chỉ thêm nếu khác câu gốc và không rỗng
                if result and result.strip() != text.strip() and len(result.strip()) > 3:
                    variants.append(result.strip())
            except Exception:
                continue

        # Nếu không đủ variant, duplicate có chỉnh sửa nhỏ
        while len(variants) < num_variants and len(variants) > 0:
            variants.append(variants[0])

        return variants[:num_variants]

    # ════════════════════════════════════════════════════════════════
    # PUBLIC: augment_dataset — augment toàn bộ DataFrame
    # ════════════════════════════════════════════════════════════════
    def augment_dataset(
        self,
        df: pd.DataFrame,
        target_per_class: int = 1000,
        text_col: str = "processed_text",
        label_col: str = "label",
        label_name_col: str = "label_name",
    ) -> pd.DataFrame:
        """
        Cân bằng dataset về target_per_class mẫu mỗi nhãn.
        - Nhãn nhiều hơn target → undersample (lấy ngẫu nhiên)
        - Nhãn ít hơn target   → augment thêm cho đủ
        """
        all_dfs = []
        label_names = {0: "clean", 1: "toxic", 2: "spam", 3: "adult"}

        unique_labels = sorted(df[label_col].unique())

        for label_id in unique_labels:
            label_df = df[df[label_col] == label_id].copy()
            label_name = label_names.get(label_id, str(label_id))
            current_count = len(label_df)

            logger.info(
                f"Label {label_id} ({label_name}): "
                f"{current_count} -> target {target_per_class}"
            )

            if current_count >= target_per_class:
                # ── Undersample ──────────────────────────────────
                sampled = label_df.sample(
                    n=target_per_class, random_state=42
                )
                all_dfs.append(sampled)
                logger.info(
                    f"  → Undersampled: {current_count} → {target_per_class}"
                )

            else:
                # ── Augment để đủ target ─────────────────────────
                needed = target_per_class - current_count
                augmented_rows = []

                texts = label_df[text_col].tolist()

                # Lặp qua data gốc nhiều vòng cho đến khi đủ
                idx = 0
                while len(augmented_rows) < needed:
                    text = texts[idx % len(texts)]
                    idx += 1

                    variants = self.augment_text(text, num_variants=2)
                    for variant in variants:
                        if len(augmented_rows) >= needed:
                            break
                        augmented_rows.append({
                            text_col:      variant,
                            label_col:     label_id,
                            label_name_col: label_name,
                            "source":      "augmented",
                        })

                aug_df = pd.DataFrame(augmented_rows)

                # Đảm bảo aug_df có đủ cột như label_df
                for col in label_df.columns:
                    if col not in aug_df.columns:
                        aug_df[col] = None

                combined = pd.concat(
                    [label_df, aug_df[label_df.columns]],
                    ignore_index=True
                )
                all_dfs.append(combined)
                logger.info(
                    f"  → Augmented: {current_count} + {len(augmented_rows)} "
                    f"= {len(combined)}"
                )

        result_df = pd.concat(all_dfs, ignore_index=True)

        # Shuffle toàn bộ
        result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(
            f"\n✅ Augmentation done: {len(df)} → {len(result_df)} samples"
        )
        logger.info(
            f"Distribution:\n{result_df[label_name_col].value_counts()}"
        )

        return result_df

    # ════════════════════════════════════════════════════════════════
    # PRIVATE: Các kỹ thuật augmentation
    # ════════════════════════════════════════════════════════════════

    def _random_deletion(self, text: str, p: float = 0.15) -> str:
        """Xóa ngẫu nhiên một số từ với xác suất p."""
        words = text.split()
        if len(words) <= 3:
            return text  # Câu quá ngắn, không xóa

        kept = [w for w in words if w in self.stop_words or random.random() > p]

        # Đảm bảo còn ít nhất 2 từ
        if len(kept) < 2:
            kept = words[:2]

        return " ".join(kept)

    def _random_swap(self, text: str, n: int = 1) -> str:
        """Hoán đổi vị trí ngẫu nhiên n cặp từ."""
        words = text.split()
        if len(words) < 4:
            return text

        new_words = words.copy()
        for _ in range(n):
            # Chọn 2 vị trí không phải stop word
            candidates = [
                i for i, w in enumerate(new_words)
                if w not in self.stop_words
            ]
            if len(candidates) < 2:
                break
            i, j = random.sample(candidates, 2)
            new_words[i], new_words[j] = new_words[j], new_words[i]

        return " ".join(new_words)

    def _synonym_replacement(self, text: str, n: int = 2) -> str:
        """Thay thế n từ bằng từ đồng nghĩa."""
        words = text.split()
        new_words = words.copy()
        replaced = 0

        # Shuffle để không luôn thay từ đầu
        indices = list(range(len(words)))
        random.shuffle(indices)

        for i in indices:
            if replaced >= n:
                break
            word = words[i].lower()
            # Tìm trong synonym dict
            if word in self.synonym_dict:
                synonym = random.choice(self.synonym_dict[word])
                new_words[i] = synonym
                replaced += 1

        return " ".join(new_words)

    def _noise_injection(self, text: str) -> str:
        """
        Giả lập lỗi gõ phím:
        - Nhân đôi ký tự ngẫu nhiên (vd: "ngu" → "nguu")
        - Bỏ ký tự ngẫu nhiên (vd: "ngu" → "nu")
        Chỉ áp dụng cho 1 từ ngẫu nhiên trong câu.
        """
        words = text.split()
        if not words:
            return text

        # Chọn 1 từ không phải stop word để inject noise
        candidates = [
            i for i, w in enumerate(words)
            if w not in self.stop_words and len(w) > 2
        ]
        if not candidates:
            return text

        idx = random.choice(candidates)
        word = words[idx]

        noise_type = random.choice(["duplicate", "delete"])

        if noise_type == "duplicate" and len(word) > 1:
            # Nhân đôi 1 ký tự ngẫu nhiên
            pos = random.randint(0, len(word) - 1)
            word = word[:pos] + word[pos] + word[pos:]
        elif noise_type == "delete" and len(word) > 3:
            # Xóa 1 ký tự ngẫu nhiên (không phải đầu/cuối)
            pos = random.randint(1, len(word) - 2)
            word = word[:pos] + word[pos + 1:]

        words[idx] = word
        return " ".join(words)
