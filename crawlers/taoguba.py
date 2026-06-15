"""
淘股吧 (tgb.cn) 爬虫 — 按股票搜索帖子并入库。

用法：
    python -m crawlers.taoguba

依赖：
    - .env 中设置 TGB_COOKIE（浏览器登录 tgb.cn 后复制）
    - PostgreSQL 已通过 docker compose 启动
"""

import logging
import random
import re
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import BASE_URL, DETAIL_DELAY, HEADERS, LIST_DELAY, MAX_RETRIES, engine

logger = logging.getLogger(__name__)

# ── helpers ────────────────────────────────────────────────────


def _safe_get(url: str, delay_range: tuple[float, float] = DETAIL_DELAY) -> Optional[requests.Response]:
    """GET with retries and polite delay."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(random.uniform(*delay_range))
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                logger.warning("non-200: %d", resp.status_code)
                continue
            # Detect login redirect page
            first_chunk = resp.text[:3000]
            if "var isLogin = 0" in first_chunk or "请登录" in first_chunk:
                logger.warning("Login wall — cookie may be expired")
                return None
            return resp
        except requests.RequestException as e:
            logger.warning("request attempt %d/%d failed: %s", attempt, MAX_RETRIES, e)
    return None


def _parse_int(text: str) -> int:
    """Extract integer from text like '1.2万' or '3,456'."""
    text = text.strip().replace(",", "").replace("，", "")
    if not text:
        return 0
    if "万" in text:
        return int(float(text.replace("万", "")) * 10000)
    try:
        return int(float(text))
    except ValueError:
        return 0


# ── stock list ──────────────────────────────────────────────────


def get_stock_list() -> pd.DataFrame:
    """Use AkShare to get CSI 300 constituents."""
    try:
        import akshare as ak
    except ImportError:
        logger.warning("akshare not available; using hardcoded top stocks")
        return _top_stocks_fallback()

    try:
        df = ak.index_stock_cons_weight_csindex("000300")
        df = df.rename(columns={"成分券代码": "code", "成分券名称": "name"})
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df[["code", "name"]]
    except Exception:
        logger.warning("AkShare CSI300 failed, using hardcoded list")
        return _top_stocks_fallback()


def _top_stocks_fallback() -> pd.DataFrame:
    """Top 100 CSI 300 stocks by weight as fallback."""
    codes = (
        "600519,600036,601318,000858,600276,000333,601166,600900,601012,002415,"
        "600030,601888,002714,000651,600887,000568,603259,601398,600809,000725,"
        "300750,002475,601288,600585,000002,601899,600309,002142,002304,000001,"
        "600031,002027,300059,600050,601668,000792,600570,601857,600436,002230,"
        "300124,002460,000063,601919,600048,600016,600104,002594,603288,300015,"
        "000538,600809,601328,601088,688981,002241,688111,300274,600019,688169,"
        "002352,601238,600028,600690,601390,300408,002459,002493,000100,300498,"
        "601818,600029,600015,601006,600000,002601,688012,000625,000776,688036,"
        "601688,600760,300760,002812,300413,002466,688599,002001,601111,002353,"
        "002410,600150,002129,000977,601615,600893,600346,600188,601066,688271"
    ).replace("\n", "")
    return pd.DataFrame([(c, f"stock_{c}") for c in codes.split(",") if c], columns=["code", "name"])


# ── search ──────────────────────────────────────────────────────


def search_stock(code: str, max_pages: int = 25) -> list[str]:
    """Search tgb.cn for a stock code and return list of detail-page URLs.

    The search/quote page embeds post links like:
        <a href='https://www.tgb.cn/a/2sARbemNtv3#T贵州茅台'>post title</a>
    We also paginate if the page supports it.
    """
    urls: set[str] = set()
    base = f"{BASE_URL}/search/search?searchContent={code}&type=2"

    for page in range(1, max_pages + 1):
        url = f"{base}&page={page}" if page > 1 else base
        logger.debug("search page %d: %s", page, url)
        resp = _safe_get(url, LIST_DELAY)
        if resp is None:
            logger.warning("search page %d failed for %s", page, code)
            continue

        # Extract all /a/{id} URLs from the page (the quote page embeds post links)
        raw_hrefs: list[str] = re.findall(r'href="(https?://[^"]*?/a/[^"]+?)"', resp.text)
        found = 0
        for raw in raw_hrefs:
            # Clean off anchor fragments (#...)
            clean = re.sub(r"#[^#]*$", "", raw)
            # Normalize /a/{topicID}/{replyID} → /a/{topicID} (same post page)
            clean = re.sub(r"^(https?://[^/]+/a/[^/]+)/\d+$", r"\1", clean)
            if clean not in urls:
                urls.add(clean)
                found += 1

        logger.debug("page %d: found %d /a/ links", page, found)
        if found == 0:
            break

    return list(urls)


# ── detail ──────────────────────────────────────────────────────


def parse_post(url: str) -> Optional[dict]:
    """Fetch and parse a single tgb.cn post page.

    Returns dict with keys matching the `posts` table schema,
    or None if the page could not be parsed.
    """
    resp = _safe_get(url)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # title
    title_el = soup.select_one("#stockTitle") or soup.select_one(".article-tittle")
    title = title_el.text.strip() if title_el else ""

    # content — main post body
    content_el = soup.select_one("#first") or soup.select_one(".article-text.p_coten")
    content = content_el.get_text(separator="\n", strip=True) if content_el else ""

    # author
    author_el = soup.select_one(".right-data-user a") or soup.select_one(".middle-list-user a")
    author = author_el.text.strip() if author_el else ""

    # post time — use regex to find YYYY-MM-DD HH:MM pattern in the page
    post_time = None
    date_matches = re.findall(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})', resp.text)
    if date_matches:
        try:
            post_time = datetime.strptime(date_matches[0], "%Y-%m-%d %H:%M")
        except ValueError:
            pass

    # reads & replies — sidebar or list cells
    read_count = 0
    reply_count = 0
    # Try comment-data sections count posts/replies
    comments = soup.select(".comment-data")
    reply_count = len(comments)

    # Read count sometimes in sidebar
    talk_el = soup.select_one(".middle-list-talk")
    if talk_el:
        parts = re.split(r"\s*/\s*", talk_el.text.strip())
        if len(parts) >= 2:
            try:
                reply_count = max(reply_count, _parse_int(parts[0]))
                read_count = _parse_int(parts[1])
            except (ValueError, IndexError):
                pass

    # Try #userFans, #likes for additional stats — ignore for now; not needed

    return {
        "platform": "taoguba",
        "stock_code": "",  # filled by caller
        "stock_name": "",
        "title": title,
        "content": content,
        "author": author,
        "post_time": post_time,
        "read_count": read_count,
        "reply_count": reply_count,
        "url": url,
    }


# ── crawl loop ──────────────────────────────────────────────────


def crawl_stock(code: str, name: str, max_search_pages: int = 25) -> list[dict]:
    """Crawl all posts for one stock."""
    logger.info("searching %s (%s)", code, name)
    urls = search_stock(code, max_pages=max_search_pages)
    logger.info("found %d post URLs for %s", len(urls), code)

    posts = []
    for i, url in enumerate(urls):
        logger.debug("[%d/%d] %s", i + 1, len(urls), url)
        post = parse_post(url)
        if post:
            post["stock_code"] = code
            post["stock_name"] = name
            posts.append(post)
        time.sleep(random.uniform(*DETAIL_DELAY))

    return posts


def save_posts(posts: list[dict]) -> int:
    """Bulk-insert posts into PostgreSQL. Returns number of new rows."""
    if not posts:
        return 0

    df = pd.DataFrame(posts)

    # Convert post_time to datetime if not already
    if "post_time" in df.columns:
        df["post_time"] = pd.to_datetime(df["post_time"], errors="coerce")

    with engine.begin() as conn:
        # Upsert: ignore duplicates by URL
        from sqlalchemy import text

        for _, row in df.iterrows():
            d = row.to_dict()
            if pd.isna(d.get("post_time")):
                d["post_time"] = None
            else:
                d["post_time"] = d["post_time"].isoformat() if d["post_time"] is not None else None
            stmt = text("""
                INSERT INTO posts (platform, stock_code, stock_name, title, content,
                                   author, post_time, read_count, reply_count, url)
                VALUES (:platform, :stock_code, :stock_name, :title, :content,
                        :author, :post_time, :read_count, :reply_count, :url)
                ON CONFLICT (url) DO NOTHING
            """)
            conn.execute(stmt, d)

    return len(df)


# ── main ────────────────────────────────────────────────────────


def main(
    max_stocks: int = 50,
    max_search_pages: int = 25,
    start_from: int = 0,
):
    """Crawl 淘股吧 posts for CSI 300 stocks.

    Args:
        max_stocks:  最多爬几只股票（默认50）
        max_search_pages: 每只股票搜几页（默认3）
        start_from:  从第几只股票开始（断点续跑）
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not HEADERS.get("Cookie"):
        logger.error("TGB_COOKIE is empty — set it in .env first!")
        logger.error("1. 浏览器登录 https://www.tgb.cn/")
        logger.error("2. F12 → Application → Cookies → 复制整段 Cookie 到 .env 的 TGB_COOKIE= 后")
        return

    stocks = get_stock_list()
    logger.info("stock pool: %d stocks", len(stocks))

    total_posts = 0
    for idx, (_, row) in enumerate(stocks.iterrows()):
        if idx < start_from:
            continue
        if idx >= start_from + max_stocks:
            break

        code, name = row["code"], row["name"]
        logger.info("── [%d/%d] %s %s ──", idx + 1, min(start_from + max_stocks, len(stocks)), code, name)

        try:
            posts = crawl_stock(code, name, max_search_pages=max_search_pages)
            n = save_posts(posts)
            total_posts += n
            logger.info("saved %d new posts for %s (total: %d)", n, code, total_posts)
        except Exception:
            logger.exception("unexpected error crawling %s", code)
            time.sleep(5)  # pause on error

    logger.info("DONE — total new posts: %d", total_posts)


if __name__ == "__main__":
    main()
