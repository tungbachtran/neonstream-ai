"""
Crawl messages từ Telegram public channels
Cần: pip install telethon
Lấy API credentials: my.telegram.org -> API development tools
"""
import asyncio
import pandas as pd
from loguru import logger
from typing import List, Dict, Optional
from pathlib import Path


class TelegramChannelCollector:
    """
    Crawl messages từ Telegram public channels.

    Cách lấy API credentials:
    1. Vào my.telegram.org
    2. Đăng nhập bằng số điện thoại
    3. "API development tools"
    4. Tạo app -> lấy api_id và api_hash

    Channels tốt cho từng nhãn:
    - spam:  Nhóm mua bán, rao vặt, quảng cáo
    - clean: Kênh tin tức, học tập
    - adult: Kênh 18+ public (nếu cần nhãn adult)
    - toxic: Kênh tranh luận, drama
    """

    LABEL_MAP = {'clean': 0, 'toxic': 1, 'spam': 2, 'adult': 3}

    def __init__(
        self,
        api_id: str,
        api_hash: str,
        phone: str,
        session_name: str = "tg_collector"
    ):
        self.api_id = int(api_id)
        self.api_hash = api_hash
        self.phone = phone
        self.session_name = session_name
        self.client = None

    async def _init_client(self):
        """Khởi tạo Telethon client"""
        try:
            from telethon import TelegramClient
            self.client = TelegramClient(
                self.session_name,
                self.api_id,
                self.api_hash
            )
            await self.client.start(phone=self.phone)
            logger.info("Telegram client connected!")
        except ImportError:
            raise ImportError("pip install telethon")

    async def _crawl_channel(
        self,
        channel: str,
        limit: int = 1000
    ) -> List[str]:
        """Crawl messages từ 1 channel"""
        messages = []
        try:
            async for message in self.client.iter_messages(
                channel,
                limit=limit,
                filter=None
            ):
                if message.text and len(message.text.strip()) >= 5:
                    messages.append(message.text.strip())
        except Exception as e:
            logger.warning(f"Failed to crawl {channel}: {e}")
        return messages

    async def _collect_async(
        self,
        channels: Dict[str, List[str]],
        limit_per_channel: int = 1000
    ) -> pd.DataFrame:
        """Async collection từ nhiều channels"""
        await self._init_client()
        all_records = []

        for label_name, channel_list in channels.items():
            label = self.LABEL_MAP.get(label_name, 0)
            logger.info(f"Crawling {len(channel_list)} channels for: {label_name}")

            for channel in channel_list:
                messages = await self._crawl_channel(channel, limit_per_channel)
                for msg in messages:
                    all_records.append({
                        'text': msg,
                        'label': label,
                        'label_name': label_name,
                        'source': f'telegram_{channel}'
                    })
                logger.info(f"  ✅ @{channel}: {len(messages)} messages")
                await asyncio.sleep(1)  # Rate limit

        await self.client.disconnect()

        df = pd.DataFrame(all_records)
        logger.info(f"Telegram total: {len(df)} messages")
        return df

    def collect(
        self,
        channels: Dict[str, List[str]],
        limit_per_channel: int = 1000
    ) -> pd.DataFrame:
        """Sync wrapper cho async collection"""
        return asyncio.run(
            self._collect_async(channels, limit_per_channel)
        )
