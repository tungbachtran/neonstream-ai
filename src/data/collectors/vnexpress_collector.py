"""
Crawl comments từ VnExpress — nguồn clean data tốt nhất
VnExpress có API comment bán công khai, không cần auth
"""
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from typing import List, Dict, Optional
from urllib.parse import urljoin


class VnExpressCollector:
    """
    Crawl comments từ VnExpress.net.

    VnExpress dùng hệ thống comment riêng với API endpoint:
    https://usi-saas.vnexpress.net/index/get?objectid=ARTICLE_ID&objecttype=1

    Comments ở đây thường CLEAN vì có moderation.
    Dùng làm nguồn clean data chính.
    """

    BASE_URL = "https://vnexpress.net"
    COMMENT_API = "https://usi-saas.vnexpress.net/index/get"

    CATEGORY_URLS = {
        "thoi-su":   "https://vnexpress.net/thoi-su",
        "giao-duc":  "https://vnexpress.net/giao-duc",
        "kinh-doanh":"https://vnexpress.net/kinh-doanh",
        "giai-tri":  "https://vnexpress.net/giai-tri",
        "the-thao":  "https://vnexpress.net/the-thao",
        "khoa-hoc":  "https://vnexpress.net/khoa-hoc",
    }

    def __init__(self, max_articles: int = 100, max_comments_per_article: int = 50):
        self.max_articles = max_articles
        self.max_comments = max_comments_per_article
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        })

    def _extract_article_id(self, url: str) -> Optional[str]:
        """Extract article ID từ URL VnExpress"""
        match = re.search(r'-(\d{7,})\.html', url)
        return match.group(1) if match else None

    def get_article_urls(self, category_url: str, limit: int = 50) -> List[str]:
        """Lấy danh sách URL bài viết từ 1 category"""
        urls = []
        try:
            resp = self.session.get(category_url, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')

            # VnExpress article links
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if re.search(r'-\d{7,}\.html$', href):
                    full_url = href if href.startswith('http') else urljoin(self.BASE_URL, href)
                    if full_url not in urls:
                        urls.append(full_url)
                if len(urls) >= limit:
                    break

        except Exception as e:
            logger.warning(f"Failed to get articles from {category_url}: {e}")

        return urls

    def get_article_comments(self, article_url: str) -> List[Dict]:
        """Lấy comments của 1 bài viết qua API"""
        article_id = self._extract_article_id(article_url)
        if not article_id:
            return []

        comments = []
        offset = 0
        limit = 100

        while len(comments) < self.max_comments:
            try:
                params = {
                    'objectid': article_id,
                    'objecttype': 1,
                    'offset': offset,
                    'limit': min(limit, self.max_comments - len(comments)),
                    'sort': 'like'
                }
                resp = self.session.get(
                    self.COMMENT_API,
                    params=params,
                    timeout=10
                )
                data = resp.json()

                items = data.get('data', {}).get('items', [])
                if not items:
                    break

                for item in items:
                    content = item.get('content', '').strip()
                    # Lọc comment quá ngắn hoặc chỉ có emoji
                    if content and len(content) >= 10:
                        comments.append({
                            'text': content,
                            'likes': item.get('userlike', 0),
                            'article_id': article_id
                        })

                if len(items) < limit:
                    break

                offset += limit
                time.sleep(0.2)

            except Exception as e:
                logger.warning(f"Comment API error for {article_id}: {e}")
                break

        return comments

    def collect(self, category_urls: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Thu thập comments từ nhiều categories.
        Tất cả đều được gán nhãn 'clean' (label=0).
        """
        if category_urls is None:
            category_urls = list(self.CATEGORY_URLS.values())

        all_comments = []
        articles_per_cat = max(1, self.max_articles // len(category_urls))

        for cat_url in category_urls:
            cat_name = cat_url.split('/')[-1]
            logger.info(f"Crawling category: {cat_name}")

            article_urls = self.get_article_urls(cat_url, limit=articles_per_cat)
            logger.info(f"  Found {len(article_urls)} articles")

            for art_url in article_urls:
                comments = self.get_article_comments(art_url)
                all_comments.extend(comments)
                time.sleep(0.5)

            logger.info(f"  Total so far: {len(all_comments)} comments")

        if not all_comments:
            return pd.DataFrame()

        df = pd.DataFrame(all_comments)
        df['label'] = 0
        df['label_name'] = 'clean'
        df['source'] = 'vnexpress'

        logger.info(f"VnExpress total: {len(df)} clean comments")
        return df[['text', 'label', 'label_name', 'source']]
