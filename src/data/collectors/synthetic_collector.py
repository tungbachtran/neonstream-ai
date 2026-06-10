"""
Sinh data tổng hợp bằng templates và rules.
Không cần API, không cần crawl — chạy offline hoàn toàn.
Đặc biệt hữu ích cho nhãn 'adult' vì khó crawl.
"""
import random
import itertools
import pandas as pd
from loguru import logger
from typing import List, Dict, Tuple
from copy import deepcopy


class VietnameseSyntheticCollector:
    """
    Sinh data tổng hợp theo template cho 4 nhãn.

    Chiến lược:
    - Clean:  Câu bình thường từ template + từ điển
    - Toxic:  Kết hợp từ tục + cấu trúc câu xúc phạm
    - Spam:   Template quảng cáo + từ khóa spam
    - Adult:  Template + từ khóa nhạy cảm (dùng placeholder)
    """

    # ── CLEAN templates ──────────────────────────────────
    CLEAN_TEMPLATES = [
        "Hôm nay {weather}, mình {activity} thật {adj}",
        "Bạn có thể {action} cho mình không?",
        "Cảm ơn {subject} đã {action} nhiệt tình",
        "Mình vừa {activity}, {adj} lắm",
        "Chúc {target} {wish} nhé",
        "Ai biết {topic} không, chia sẻ với mình với",
        "Sản phẩm {quality}, mình {reaction}",
        "Mình đang tìm {item}, bạn biết chỗ nào không",
        "{subject} thật sự rất {adj} và {adj2}",
        "Hôm nay mình học được {topic}, thú vị lắm",
    ]

    CLEAN_SLOTS = {
        "weather":  ["đẹp quá", "mát mẻ", "nắng đẹp", "se lạnh"],
        "activity": ["đi chơi", "nấu ăn", "học bài", "đọc sách", "xem phim"],
        "adj":      ["vui", "thú vị", "tuyệt vời", "hạnh phúc", "bổ ích"],
        "adj2":     ["hữu ích", "chất lượng", "đáng tin", "chuyên nghiệp"],
        "action":   ["giúp đỡ", "hỗ trợ", "tư vấn", "chia sẻ", "giải thích"],
        "subject":  ["admin", "bạn", "mọi người", "shop", "tác giả"],
        "target":   ["bạn", "mọi người", "gia đình", "anh chị"],
        "wish":     ["sức khỏe", "may mắn", "thành công", "vui vẻ"],
        "topic":    ["lập trình", "tiếng Anh", "nấu ăn", "lịch sử", "khoa học"],
        "quality":  ["chất lượng tốt", "giao hàng nhanh", "đúng mô tả"],
        "reaction": ["rất hài lòng", "sẽ mua lại", "giới thiệu cho bạn bè"],
        "item":     ["tài liệu học", "sản phẩm này", "địa chỉ shop", "thông tin"],
    }

    # ── TOXIC templates ──────────────────────────────────
    TOXIC_TEMPLATES = [
        "Mày {insult}, {action} đi",
        "Đồ {insult2}, {consequence}",
        "{exclaim} mày, {accusation}",
        "Nói chuyện với mày {waste}, {insult3}",
        "Mày là {insult4}, không {ability}",
        "Thứ như mày {judgment}",
        "Câm {part} lại, {reason}",
        "Mày {action2} vl, {insult5}",
    ]

    TOXIC_SLOTS = {
        "insult":    ["ngu vcl", "vô dụng thật sự", "hết thuốc chữa", "não cá vàng"],
        "insult2":   ["vô học", "rác rưởi", "súc vật", "khốn nạn"],
        "insult3":   ["đồ vô dụng", "thứ rác", "kẻ thất bại"],
        "insult4":   ["kẻ hèn nhát", "đứa phản bội", "thứ ăn hại"],
        "insult5":   ["biến đi cho khuất mắt", "đừng có nói chuyện nữa"],
        "action":    ["biến", "cút", "im miệng", "xéo"],
        "action2":   ["nói chuyện", "làm việc", "suy nghĩ"],
        "exclaim":   ["Đm", "Vcl", "Đcm"],
        "accusation":["chỉ biết phá hoại", "không làm được gì ra hồn", "toàn nói bậy"],
        "consequence":["không ai thèm chơi", "biến đi cho rồi", "xấu hổ chưa"],
        "waste":     ["phí thời gian vl", "mệt mỏi lắm", "chán lắm rồi"],
        "judgment":  ["không xứng đáng", "chỉ đáng bị khinh", "thật đáng thương"],
        "ability":   ["làm được gì", "hiểu gì cả", "biết gì hết"],
        "reason":    ["không ai hỏi mày", "mày biết gì", "nói chuyện phí hơi"],
        "part":      ["mồm", "miệng"],
    }

    # ── SPAM templates ───────────────────────────────────
    SPAM_TEMPLATES = [
        "{cta} {product} {discount} {urgency}",
        "KIẾM {amount} MỖI NGÀY {method} {contact}",
        "Tuyển {position} {salary} {requirement}",
        "{product2} {claim} {contact2}",
        "Giảm {pct}% {product3} {urgency2} {contact3}",
        "Bí quyết {benefit} trong {timeframe} {guarantee}",
        "Vay {loan_amount} {loan_condition} {contact4}",
        "FREE {freebie} khi {condition} {contact5}",
    ]

    SPAM_SLOTS = {
        "cta":          ["MUA NGAY", "ĐẶT HÀNG NGAY", "INBOX NGAY", "ORDER NGAY"],
        "product":      ["áo thun", "mỹ phẩm Hàn", "giày Nike rep", "túi xách"],
        "product2":     ["Thuốc giảm cân", "Kem trắng da", "Thực phẩm chức năng"],
        "product3":     ["toàn bộ sản phẩm", "hàng mới về", "hàng thanh lý"],
        "discount":     ["giảm 50%", "giảm 70%", "giá sỉ cực rẻ"],
        "urgency":      ["hôm nay thôi", "số lượng có hạn", "chỉ còn 10 suất"],
        "urgency2":     ["flash sale 2 tiếng", "hôm nay duy nhất", "kết thúc lúc 12h"],
        "amount":       ["500k", "1-2 triệu", "5-10 triệu"],
        "method":       ["không cần vốn", "chỉ cần điện thoại", "làm tại nhà"],
        "contact":      ["LH: 0909xxxxxx", "inbox ngay", "zalo: 0909xxxxxx"],
        "contact2":     ["inbox để biết giá", "LH ngay hôm nay", "comment SĐT"],
        "contact3":     ["đặt hàng ngay", "inbox shop", "LH: 0909xxxxxx"],
        "contact4":     ["LH ngay", "gọi ngay 0909xxxxxx", "zalo tư vấn miễn phí"],
        "contact5":     ["đăng ký ngay", "inbox nhận quà", "comment để nhận"],
        "position":     ["CTV bán hàng online", "nhân viên làm việc tại nhà", "đại lý"],
        "salary":       ["thu nhập 20-50tr/tháng", "lương 15-30 triệu", "hoa hồng 40%"],
        "requirement":  ["không cần kinh nghiệm", "không cần bằng cấp", "ai cũng làm được"],
        "claim":        ["giảm 10kg trong 7 ngày", "trắng da sau 3 ngày", "chữa bách bệnh"],
        "pct":          ["50", "60", "70", "80", "90"],
        "benefit":      ["giảm cân 10kg", "trắng da", "tăng chiều cao 10cm", "kiếm tiền"],
        "timeframe":    ["7 ngày", "1 tháng", "2 tuần"],
        "guarantee":    ["đảm bảo 100%", "cam kết hiệu quả", "hoàn tiền nếu không hiệu quả"],
        "loan_amount":  ["10-50 triệu", "5-100 triệu", "đến 500 triệu"],
        "loan_condition":["không cần thế chấp", "lãi suất 0% tháng đầu", "giải ngân 30 phút"],
        "freebie":      ["SHIP toàn quốc", "quà tặng 500k", "voucher 200k", "tư vấn miễn phí"],
        "condition":    ["đặt hàng hôm nay", "đăng ký ngay", "mua từ 500k"],
    }

    # ── ADULT templates ──────────────────────────────────
    ADULT_TEMPLATES = [
        "{adult_adj} {adult_noun} {adult_action}",
        "Xem {adult_content} {adult_platform} {adult_cta}",
        "{adult_offer} {adult_contact}",
        "Tìm {adult_partner} {adult_condition}",
        "{adult_exclaim} {adult_desc}",
        "{adult_exclaim} muốn {adult_action} {adult_noun} ngay",
        "Inbox {adult_contact} xem {adult_content} {adult_condition}",
        "{adult_partner} {adult_condition} {adult_cta}",
        "Muốn {adult_action} {adult_adj} {adult_noun} không?",
        "{adult_offer} {adult_action} {adult_platform} {adult_cta}",
        "Chỉ cần {adult_contact} là có {adult_content} {adult_adj}",
        "{adult_exclaim} {adult_noun} đang {adult_desc}, {adult_cta}",
        "Tìm {adult_partner} để {adult_action} {adult_condition}",
        "{adult_adj} {adult_content} {adult_platform} miễn phí {adult_cta}",
        "Gái {adult_adj} sẵn sàng {adult_action} {adult_noun} {adult_contact}",
    ]


    ADULT_SLOTS = {
    "adult_adj": [
        "nóng", "dâm đãng", "sướng", "mát xa", "cực mạnh", 
        "không che", "full HD", "mới nhất", "cao cấp"
    ],
    "adult_noun": [
        "lồn", "cu", "chim", "cặc", "vú", "mông", "bướm", 
        "sex", "clip sex", "phim sex", "ảnh nude", "sex toy"
    ],
    "adult_action": [
        "chịch", "địt", "bú", "liếm", "cưỡi", "doggy", "69", 
        "thủ dâm", "show hàng", "chơi sex", "làm tình"
    ],
    "adult_content": [
        "phim sex", "clip sex", "ảnh nude", "video 18+", 
        "live sex", "sex show", "chịch nhau"
    ],
    "adult_platform": [
        "zalo", "telegram", "inbox", "web 18+", "link ngay", 
        "group kín", "chat 1-1"
    ],
    "adult_cta": [
        "inbox ngay", "liên hệ ngay", "xem miễn phí", 
        "gọi ngay", "nhắn zalo", "đặt lịch"
    ],
    "adult_offer": [
        "gái gọi", "bạn tình 1 đêm", "massage yoni", 
        "sex toy", "clip tự quay", "gái xinh show hàng"
    ],
    "adult_contact": [
        "LH 0909xxx", "zalo 0978xxx", "inbox ngay", 
        "sdt: 0898xxx", "telegram @xxx"
    ],
    "adult_partner": [
        "bạn tình", "gái xinh", "em gái", "anh lớn tuổi", 
        "bạn chịch", "gái gọi cao cấp"
    ],
    "adult_condition": [
        "không ràng buộc", "sướng là được", "giá rẻ", 
        "tại nhà", "full service", "không che"
    ],
    "adult_exclaim": [
        "Sướng quá", "Cứng quá", "Muốn chịch ngay", 
        "Nóng quá trời", "Đang rất ướt", "Cực khoái"
    ],
    "adult_desc": [
        "em đang nóng", "anh đang cứng", "lồn em ướt", 
        "muốn được địt mạnh", "sướng kinh khủng", "thật sướng"
    ],
}


    def __init__(self, seed: int = 42):
        random.seed(seed)

    def _fill_template(self, template: str, slots: Dict) -> str:
        """Điền slot vào template"""
        result = template
        for slot_name, options in slots.items():
            placeholder = "{" + slot_name + "}"
            if placeholder in result:
                result = result.replace(placeholder, random.choice(options))
        return result

    def generate_samples(
        self,
        label_name: str,
        count: int
    ) -> List[str]:
        """Sinh `count` mẫu cho 1 nhãn"""
        template_map = {
            'clean': (self.CLEAN_TEMPLATES, self.CLEAN_SLOTS),
            'toxic': (self.TOXIC_TEMPLATES, self.TOXIC_SLOTS),
            'spam':  (self.SPAM_TEMPLATES,  self.SPAM_SLOTS),
            'adult': (self.ADULT_TEMPLATES, self.ADULT_SLOTS),
        }

        if label_name not in template_map:
            raise ValueError(f"Unknown label: {label_name}")

        templates, slots = template_map[label_name]
        samples = []

        # Tạo tất cả combinations trước, sau đó sample
        for _ in range(count * 3):  # Sinh dư để dedup
            template = random.choice(templates)
            text = self._fill_template(template, slots)
            if text not in samples:
                samples.append(text)
            if len(samples) >= count:
                break

        return samples[:count]

    def collect(
        self,
        target_per_class: int = 500,
        enable_adult: bool = True
    ) -> pd.DataFrame:
        """Sinh data tổng hợp cho tất cả nhãn"""
        label_map = {'clean': 0, 'toxic': 1, 'spam': 2, 'adult': 3}
        labels_to_generate = ['clean', 'toxic', 'spam']
        if enable_adult:
            labels_to_generate.append('adult')

        records = []
        for label_name in labels_to_generate:
            label = label_map[label_name]
            samples = self.generate_samples(label_name, target_per_class)
            for text in samples:
                records.append({
                    'text': text,
                    'label': label,
                    'label_name': label_name,
                    'source': 'synthetic'
                })
            logger.info(f"Generated {len(samples)} synthetic {label_name} samples")

        df = pd.DataFrame(records)
        logger.info(f"Synthetic total: {len(df)} samples")
        return df
