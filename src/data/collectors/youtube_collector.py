"""
Crawl comments từ YouTube
Cần: pip install google-api-python-client
Lấy API key: console.cloud.google.com -> YouTube Data API v3
Quota miễn phí: 10,000 units/ngày (~100 video requests)
"""
import time
import pandas as pd
from loguru import logger
from typing import List, Dict, Optional
from pathlib import Path
import json


class YouTubeCommentCrawler:
    """
    Crawl YouTube comments theo video ID.

    Cách lấy API key:
    1. Vào console.cloud.google.com
    2. Tạo project mới
    3. Enable "YouTube Data API v3"
    4. Credentials -> Create API Key
    """

    # Mapping label theo loại video target
    LABEL_MAP = {'clean': 0, 'toxic': 1, 'spam': 2, 'adult': 3}

    def __init__(self, api_key: str, max_comments_per_video: int = 300):
        self.api_key = api_key
        self.max_comments = max_comments_per_video
        self._service = None

    def _get_service(self):
        """Lazy init YouTube service"""
        if self._service is None:
            try:
                from googleapiclient.discovery import build
                self._service = build(
                    'youtube', 'v3',
                    developerKey=self.api_key,
                    cache_discovery=False
                )
            except ImportError:
                raise ImportError("pip install google-api-python-client")
        return self._service

    def get_video_comments(
        self,
        video_id: str,
        max_results: Optional[int] = None
    ) -> List[Dict]:
        """Lấy comments của 1 video"""
        service = self._get_service()
        max_results = max_results or self.max_comments
        comments = []
        next_page_token = None

        while len(comments) < max_results:
            try:
                request = service.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(100, max_results - len(comments)),
                    pageToken=next_page_token,
                    textFormat="plainText",
                    order="relevance"
                )
                response = request.execute()

                for item in response.get("items", []):
                    top = item["snippet"]["topLevelComment"]["snippet"]
                    text = top["textDisplay"].strip()
                    if text and len(text) >= 5:
                        comments.append({
                            "text": text,
                            "like_count": top.get("likeCount", 0),
                            "reply_count": item["snippet"].get("totalReplyCount", 0),
                            "video_id": video_id
                        })

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

                time.sleep(0.3)

            except Exception as e:
                logger.warning(f"Error fetching comments for {video_id}: {e}")
                break

        return comments

    def crawl_by_label(
        self,
        video_targets: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Crawl nhiều video, gán nhãn sơ bộ theo loại video.

        video_targets = {
            'clean':  ['video_id_1', 'video_id_2'],
            'toxic':  ['video_id_3'],
            'spam':   ['video_id_4'],
            'adult':  ['video_id_5'],
        }

        LƯU Ý: Đây là nhãn sơ bộ (weak label).
        Comments từ video toxic KHÔNG có nghĩa là tất cả đều toxic.
        Cần human review hoặc keyword filter sau.
        """
        all_records = []

        for label_name, video_ids in video_targets.items():
            label = self.LABEL_MAP.get(label_name, 0)
            logger.info(f"Crawling {len(video_ids)} videos for label: {label_name}")

            for vid_id in video_ids:
                try:
                    comments = self.get_video_comments(vid_id)
                    for c in comments:
                        c['label'] = label
                        c['label_name'] = label_name
                        c['source'] = 'youtube'
                    all_records.extend(comments)
                    logger.info(f"  ✅ {vid_id}: {len(comments)} comments")
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"  ❌ {vid_id}: {e}")

        df = pd.DataFrame(all_records)
        if df.empty:
            return df

        logger.info(f"YouTube total: {len(df)} comments")
        logger.info(f"Distribution: {df['label_name'].value_counts().to_dict()}")
        return df[['text', 'label', 'label_name', 'source']]

    def search_and_crawl(
        self,
        queries: Dict[str, List[str]],
        videos_per_query: int = 5
    ) -> pd.DataFrame:
        """
        Tìm kiếm video theo query rồi crawl comments.
        Không cần biết video ID trước.

        queries = {
            'toxic': ['drama việt nam', 'tranh luận gay gắt'],
            'spam':  ['mua bán online', 'kiếm tiền online'],
            'clean': ['học tiếng việt', 'tin tức hôm nay'],
            'adult': ['phim 18+ review', 'nội dung người lớn'],
        }
        """
        service = self._get_service()
        all_video_ids: Dict[str, List[str]] = {k: [] for k in queries}

        for label_name, query_list in queries.items():
            for query in query_list:
                try:
                    request = service.search().list(
                        part="snippet",
                        q=query,
                        type="video",
                        maxResults=videos_per_query,
                        relevanceLanguage="vi",
                        regionCode="VN"
                    )
                    response = request.execute()
                    for item in response.get("items", []):
                        vid_id = item["id"]["videoId"]
                        all_video_ids[label_name].append(vid_id)
                    time.sleep(0.5)
                except Exception as e:
                    logger.warning(f"Search failed for '{query}': {e}")

        return self.crawl_by_label(all_video_ids)
