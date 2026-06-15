"""Database and HTTP configuration."""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://alpha:alpha123@localhost:5432/sentiment_alpha")
TGB_COOKIE = os.getenv("TGB_COOKIE", "")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

BASE_URL = "https://www.tgb.cn"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Cookie": TGB_COOKIE,
}

API_HEADERS = {
    **HEADERS,
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}

# Rate limiting
LIST_DELAY = (1.5, 3.0)  # seconds between list page requests
DETAIL_DELAY = (0.8, 1.5)  # seconds between detail page requests
MAX_RETRIES = 3
