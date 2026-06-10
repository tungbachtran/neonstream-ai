"""
Master collector — gộp tất cả nguồn dữ liệu
Thay thế hoàn toàn collector.py cũ
"""
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import List, Dict, Optional

from src.data.collectors.huggingface_collector import HuggingFaceCollector
from src.data.collectors.youtube_collector import YouTubeCommentCrawler
from src.data.collectors.vnexpress_collector import VnExpressCollector
from src.data.collectors.telegram_collector import TelegramChannelCollector
from src.data.collectors.synthetic_collector import VietnameseSyntheticCollector
from src.data.annotator import AutoLabeler, HumanReviewExporter


class VietnameseDataCollector:
    """
    Master collector tích hợp tất cả nguồn.
    Tương thích ngược với code cũ — vẫn có method collect_all().
    """

    LABEL_MAP = {0: 'clean', 1: 'toxic', 2: 'spam', 3: 'adult'}

    def __init__(
        self,
        data_dir: str = "data/raw",
        config: Optional[Dict] = None
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.collection_cfg = config.get('collection', {}) if config else {}
        self.auto_labeler = AutoLabeler()
        self.review_exporter = HumanReviewExporter(
            output_dir=str(self.data_dir / "manual")
        )

    # ── Seed data (giữ lại từ code cũ, mở rộng 4 nhãn) ──
    def create_seed_dataset(self) -> pd.DataFrame:
        """Dataset seed mở rộng với 4 nhãn"""
        samples = {
            "clean": [
                "Hôm nay thời tiết đẹp quá, mình đi chơi nhé",
                "Bạn có thể giúp mình giải bài toán này không?",
                "Cảm ơn bạn đã hỗ trợ nhiệt tình",
                "Mình vừa xem bộ phim hay lắm, bạn thử xem đi",
                "Chúc mừng sinh nhật bạn nhé, chúc bạn nhiều sức khỏe",
                "Hôm nay mình học được nhiều điều mới thú vị",
                "Bạn có biết nhà hàng nào ngon ở khu vực này không?",
                "Mình cần tư vấn về sản phẩm này",
                "Cảm ơn admin đã tạo ra group hữu ích như vậy",
                "Mọi người ơi cho mình hỏi về vấn đề này với",
                "Sản phẩm chất lượng tốt, giao hàng nhanh, mình hài lòng",
                "Bài viết rất hay và bổ ích, cảm ơn tác giả",
                "Mình đang tìm kiếm tài liệu học tiếng Anh",
                "Hôm nay công việc bận rộn nhưng vui",
                "Gia đình mình vừa đi du lịch về, tuyệt vời lắm",
            ],
            "toxic": [
                "Mày ngu vl, làm cái gì cũng hỏng",
                "Đồ vô dụng, không làm được trò trống gì",
                "Câm mồm đi, không ai hỏi mày",
                "Mày là thứ rác rưởi của xã hội",
                "Đm, nói chuyện với mày mệt vl",
                "Thằng ngu, biến đi cho khuất mắt",
                "Mày có não không vậy, nói chuyện như con bò",
                "Đồ khốn nạn, ai cho mày vào đây",
                "Mày tưởng mày giỏi lắm à, rác thôi",
                "Cút đi, không ai muốn nhìn mặt mày",
                "Nói chuyện với đứa như mày phí thời gian vl",
                "Mày là đồ vô học, không biết gì hết",
                "Thứ như mày không xứng đáng sống trong xã hội",
                "Mày điên à, nói chuyện không có não",
                "Đồ súc vật, biết gì mà nói",
            ],
            "spam": [
                "KIẾM TIỀN ONLINE 5-10 TRIỆU/NGÀY không cần vốn LH NGAY 0909xxx",
                "Bán hàng giảm giá 90% hôm nay thôi, inbox ngay kẻo hết",
                "Cơ hội vàng làm giàu nhanh chóng, đầu tư 1 triệu lãi 10 triệu",
                "FREE SHIP toàn quốc, giảm 50% tất cả sản phẩm hôm nay",
                "Tuyển CTV bán hàng online thu nhập 20-50tr/tháng không cần kinh nghiệm",
                "Bí quyết giảm cân 10kg trong 7 ngày, đảm bảo hiệu quả 100%",
                "Vay tiền nhanh không cần thế chấp, giải ngân trong 30 phút",
                "Mua 1 tặng 1 hôm nay duy nhất, số lượng có hạn inbox nhanh",
                "Kiếm 500k mỗi ngày chỉ cần ngồi nhà bấm điện thoại",
                "Sản phẩm thần kỳ chữa bách bệnh, dùng 3 ngày là khỏi",
                "CLICK NGAY để nhận quà tặng trị giá 5 triệu đồng MIỄN PHÍ",
                "Đầu tư crypto lãi 300% mỗi tháng, uy tín 100% không lừa đảo",
                "Thuốc tăng chiều cao 15cm trong 1 tháng, hiệu quả đảm bảo",
                "Tuyển nhân viên làm việc tại nhà lương 30tr không cần bằng",
                "Tuyển nhân viên làm việc tại nhà lương 30tr không cần bằng cấp",
                "Giảm giá sốc 70% chỉ hôm nay, đặt hàng ngay kẻo hết",
                "Bán tài khoản Netflix giá rẻ, bảo hành trọn đời",
                "Hack não học tiếng Anh trong 30 ngày, cam kết giao tiếp được",
                "Cơ hội đầu tư bất động sản sinh lời 200% không rủi ro",
                "Mỹ phẩm Hàn Quốc chính hãng giá sỉ, inbox để biết giá",
                "Tặng ngay 100k vào tài khoản khi đăng ký, không điều kiện",
            ],
            "adult": [
                "Muốn chịch em không, em đang rất nóng",
                "Bú cu anh đi em, anh đang cứng quá",
                "Lồn em ướt hết rồi, muốn địt ngay bây giờ",
                "Xem phim sex không che full HD, inbox zalo ngay",
                "Tìm bạn tình 1 đêm, gái xinh, sướng lắm",
                "Chịch doggy sướng quá trời, ai muốn thử không",
                "Bán clip sex tự quay, giá rẻ, inbox để xem mẫu",
                "Gái gọi cao cấp, phục vụ tại nhà, LH 0909xxx",
                "Muốn xem em bú liếm không, live 18+ ngay",
                "Sex toy hàng Nhật, rung mạnh, sướng kinh khủng",
                "Tìm anh lớn tuổi, em muốn được chịch mạnh",
                "Phim sex Việt Nam mới nhất, xem miễn phí",
                "69 với em sướng không tả nổi, thử đi anh",
                "Massage yoni + sex, giá chỉ 1 triệu, inbox ngay",
                "Gái xinh show hàng, chat sex 1-1, zalo 0978xxx",
                "Muốn cưỡi ngựa không em, anh đang rất cứng",
                "Bán ảnh nude + clip sex, giá sỉ, ai cần liên hệ",
                "Tìm bạn chịch không ràng buộc, sướng là được",
                "Live sex show, xem em thủ dâm, cực mạnh",
                "Địt nhau sướng quá, em muốn được bắn đầy vào",
            ],

        }

        records = []
        label_map = {"clean": 0, "toxic": 1, "spam": 2, "adult": 3}
        for label_name, texts in samples.items():
            label = label_map[label_name]
            for text in texts:
                records.append({
                    "text": text,
                    "label": label,
                    "label_name": label_name,
                    "source": "seed"
                })

        df = pd.DataFrame(records)
        logger.info(f"Seed dataset: {len(df)} samples")
        logger.info(f"Distribution:\n{df['label_name'].value_counts()}")
        return df

    def load_from_files(self) -> pd.DataFrame:
        """Load thêm data từ file txt nếu có (giữ tương thích code cũ)"""
        dfs = []
        for label, filename in [
            (0, "clean.txt"), (1, "toxic.txt"),
            (2, "spam.txt"),  (3, "adult.txt")
        ]:
            filepath = self.data_dir / filename
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                label_name = self.LABEL_MAP[label]
                df = pd.DataFrame({
                    "text": lines,
                    "label": label,
                    "label_name": label_name,
                    "source": f"file_{filename}"
                })
                dfs.append(df)
                logger.info(f"Loaded {len(df)} samples from {filename}")

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def collect_from_huggingface(self) -> pd.DataFrame:
        """Thu thập từ HuggingFace theo config"""
        hf_cfg = self.collection_cfg.get('huggingface', {})
        datasets_cfg = hf_cfg.get('datasets', [])
        if not datasets_cfg:
            return pd.DataFrame()

        collector = HuggingFaceCollector(datasets_cfg)
        return collector.collect_all()

    def collect_from_vnexpress(self) -> pd.DataFrame:
        """Thu thập clean data từ VnExpress"""
        vne_cfg = self.collection_cfg.get('vnexpress', {})
        collector = VnExpressCollector(
            max_articles=vne_cfg.get('max_articles', 100),
            max_comments_per_article=vne_cfg.get('max_comments_per_article', 50)
        )
        category_urls = vne_cfg.get('categories', None)
        return collector.collect(category_urls)

    def collect_from_youtube(self) -> pd.DataFrame:
        """Thu thập từ YouTube"""
        yt_cfg = self.collection_cfg.get('youtube', {})
        api_key = yt_cfg.get('api_key', '')
        if not api_key:
            logger.warning("YouTube API key not set, skipping.")
            return pd.DataFrame()

        crawler = YouTubeCommentCrawler(
            api_key=api_key,
            max_comments_per_video=yt_cfg.get('max_comments_per_video', 300)
        )
        video_targets = yt_cfg.get('video_targets', {})
        return crawler.crawl_by_label(video_targets)

    def collect_from_telegram(self) -> pd.DataFrame:
        """Thu thập từ Telegram"""
        tg_cfg = self.collection_cfg.get('telegram', {})
        api_id   = tg_cfg.get('api_id', '')
        api_hash = tg_cfg.get('api_hash', '')
        phone    = tg_cfg.get('phone', '')

        if not all([api_id, api_hash, phone]):
            logger.warning("Telegram credentials not set, skipping.")
            return pd.DataFrame()

        collector = TelegramChannelCollector(api_id, api_hash, phone)
        channels  = tg_cfg.get('channels', {})
        return collector.collect(channels)

    def collect_from_synthetic(self) -> pd.DataFrame:
        """Sinh data tổng hợp"""
        syn_cfg = self.collection_cfg.get('synthetic', {})
        collector = VietnameseSyntheticCollector()
        return collector.collect(
            target_per_class=syn_cfg.get('target_per_class', 500),
            enable_adult=syn_cfg.get('enable_adult', True)
        )

    def collect_from_manual(self) -> pd.DataFrame:
        """Load file đã được human annotate"""
        manual_cfg = self.collection_cfg.get('manual', {})
        files = manual_cfg.get('files', [])
        if not files:
            return pd.DataFrame()

        dfs = []
        for file_path in files:
            if Path(file_path).exists():
                df = self.review_exporter.load_reviewed(file_path)
                dfs.append(df)
            else:
                logger.warning(f"Manual file not found: {file_path}")

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def collect_all(self) -> pd.DataFrame:
        """
        Entry point chính — gộp tất cả nguồn.
        Tương thích ngược với scripts/prepare_data.py cũ.
        """
        enabled = self.collection_cfg.get('enabled_sources', ['seed', 'synthetic'])
        all_dfs = []

        # Luôn load seed + file
        logger.info("📦 Loading seed dataset...")
        all_dfs.append(self.create_seed_dataset())

        file_df = self.load_from_files()
        if not file_df.empty:
            all_dfs.append(file_df)

        # Load theo enabled_sources
        source_map = {
            'huggingface': self.collect_from_huggingface,
            'vnexpress':   self.collect_from_vnexpress,
            'youtube':     self.collect_from_youtube,
            'telegram':    self.collect_from_telegram,
            'synthetic':   self.collect_from_synthetic,
            'manual':      self.collect_from_manual,
        }

        for source in enabled:
            if source in source_map:
                logger.info(f"📦 Collecting from: {source}...")
                try:
                    df = source_map[source]()
                    if not df.empty:
                        # Lưu cache từng nguồn
                        cache_path = self.data_dir / f"{source}_raw.csv"
                        df.to_csv(cache_path, index=False, encoding='utf-8')
                        logger.info(f"  Cached to {cache_path}")
                        all_dfs.append(df)
                except Exception as e:
                    logger.error(f"  Failed to collect from {source}: {e}")

        # Gộp tất cả
        combined = pd.concat(all_dfs, ignore_index=True)

        # Dedup
        before = len(combined)
        combined = combined.drop_duplicates(subset=['text']).reset_index(drop=True)
        logger.info(f"Dedup: {before} -> {len(combined)} samples")

        # Đảm bảo có đủ columns
        if 'label_name' not in combined.columns:
            combined['label_name'] = combined['label'].map(self.LABEL_MAP)

        logger.info(f"\n✅ Total collected: {len(combined)} unique samples")
        logger.info(f"Distribution:\n{combined['label_name'].value_counts()}")

        return combined

    def export_for_review(
        self,
        df: pd.DataFrame,
        batch_name: str = "batch1",
        sample_per_label: int = 200
    ) -> str:
        """Export để human review — wrapper tiện dụng"""
        # Auto-label trước khi export
        df = self.auto_labeler.label_dataframe(df.copy())
        return self.review_exporter.export_for_review(
            df, batch_name, sample_per_label
        )
