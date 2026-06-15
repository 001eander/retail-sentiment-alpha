"""
东方财富股吧 (guba.eastmoney.com) 爬虫 — 按股票遍历帖子并入库。

用法：
    python -m crawlers.eastmoney

覆盖 CSI 300 成分股，遍历个股列表页（每页80帖），提取元数据后抓取详情页。

Features
--------
- 无需 cookie / 登录
- 自动处理 GB2312/GBK 编码
- 请求间随机延时，避免 IP 屏蔽
- ON CONFLICT (url) DO NOTHING 去重入库
"""

import logging
import random
import re
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

from .config import engine
# TODO(tech-debt): extract get_stock_list to config.py or a shared
# utils module to break the eastmoney → taoguba dependency.
from .taoguba import get_stock_list

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────

BASE_URL = "https://guba.eastmoney.com"
LIST_URL_TPL = BASE_URL + "/list,{code},f_{page}.html"
LIST_URL_TPL_P1 = BASE_URL + "/list,{code}.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://guba.eastmoney.com/",
}

LIST_DELAY = (2.0, 3.0)  # seconds between listing page requests
DETAIL_DELAY = (1.0, 1.5)  # seconds between detail page requests
STOCK_DELAY = 5.0  # seconds between switching stocks
MAX_RETRIES = 2

# ── helpers ──────────────────────────────────────────────────────


def _safe_get(
    url: str,
    delay_range: tuple[float, float] = DETAIL_DELAY,
    session: Optional[requests.Session] = None,
) -> Optional[requests.Response]:
    """GET with retries, encoding detection, and polite delay.

    Args:
        url: Target URL.
        delay_range: (min, max) seconds to sleep before request.
        session: Reusable ``requests.Session`` (or ``None`` for ad-hoc).

    Returns:
        ``requests.Response`` with status 200, or ``None`` on failure.
    """
    sess = session or requests
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt == 1:
                time.sleep(random.uniform(*delay_range))
            resp = sess.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                logger.warning("HTTP %d for %s", resp.status_code, url)
                continue

            # EastMoney often serves GB2312 / GBK.  Use the more permissive
            # ``apparent_encoding`` (chardet) when it detects a Chinese charset.
            if resp.apparent_encoding:
                enc = resp.apparent_encoding.lower()
                if enc in ("gb2312", "gbk", "gb18030"):
                    resp.encoding = "gbk"  # gbk is a superset of gb2312
                elif enc not in ("ascii", "iso-8859-1"):
                    resp.encoding = enc

            return resp
        except requests.RequestException as e:
            logger.warning("request attempt %d/%d failed: %s", attempt, MAX_RETRIES, e)
    return None


def _parse_int(text: str) -> int:
    """Extract integer from an EastMoney-style number string.

    Handles patterns like ``'1234'``, ``'1.2万'``, ``'--'`` (returns 0).
    """
    text = text.strip().replace(",", "").replace("，", "")
    if not text or text == "--":
        return 0
    if "万" in text:
        return int(float(text.replace("万", "")) * 10000)
    try:
        return int(float(text))
    except ValueError:
        return 0


# ── date parsing ─────────────────────────────────────────────────


def parse_date(date_str: str, fallback_year: Optional[int] = None) -> Optional[datetime]:
    """Parse an EastMoney date string into a ``datetime``.

    Handles two common formats:
    - ``"2025-06-15 14:30"`` (full date with year)
    - ``"06-15 14:30"`` (year-less — uses *fallback_year*)

    Args:
        date_str: Date string from the page.
        fallback_year: Year to use for year-less dates.

    Returns:
        ``datetime`` or ``None`` if parsing fails.
    """
    if not date_str or not date_str.strip():
        return None
    date_str = date_str.strip()
    if fallback_year is None:
        fallback_year = datetime.now().year

    # Full format: YYYY-MM-DD HH:MM
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    except ValueError:
        pass

    # Year-less format: MM-DD HH:MM
    try:
        dt = datetime.strptime(date_str, "%m-%d %H:%M")
        return dt.replace(year=fallback_year)
    except ValueError:
        pass

    # Date-only variants (rare on eastmoney, but defensive)
    for fmt in ("%Y-%m-%d", "%m-%d"):
        try:
            dt = datetime.strptime(date_str, fmt)
            if fmt.startswith("%m-"):
                dt = dt.replace(year=fallback_year)
            return dt
        except ValueError:
            continue

    logger.debug("unable to parse date string: %s", date_str)
    return None


# ── listing page ─────────────────────────────────────────────────


def crawl_stock_list(code: str, max_pages: int = 25, session: Optional[requests.Session] = None) -> list[dict]:
    """Crawl listing pages for one stock and return post metadata.

    Iterates through ``list,{code},f_{page}.html`` (80 posts/page),
    extracting post_id, title, url, author, read_count, reply_count,
    and the raw time string from each ``div.articleh.normal_post``.

    Stops early when a page returns 0 posts (past the last page).

    Args:
        code: 6-digit stock code (e.g. ``'600519'``).
        max_pages: Maximum listing pages to crawl.
        session: Reusable ``requests.Session`` (or ``None`` for ad-hoc).

    Returns:
        List of dicts with keys:
            post_id, title, url, author, read_count, reply_count, post_time_str
    """
    posts: list[dict] = []
    own_session = session is None
    if own_session:
        session = requests.Session()

    for page in range(1, max_pages + 1):
        url = LIST_URL_TPL_P1.format(code=code) if page == 1 else LIST_URL_TPL.format(code=code, page=page)

        logger.debug("listing page %d: %s", page, url)
        resp = _safe_get(url, LIST_DELAY, session)
        if resp is None:
            logger.warning("listing page %d for %s failed", page, code)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        # Post containers are table rows; skip header rows
        rows = soup.select("tr.listitem")

        page_count = 0
        for row in rows:
            try:
                post = _parse_listing_row(row, code)
                if post:
                    posts.append(post)
                    page_count += 1
            except Exception:
                logger.exception("failed to parse a listing row on page %d", page)
                continue

        logger.debug("page %d: %d posts extracted", page, page_count)

        # Stop early when a page has no posts
        if page_count == 0:
            logger.info("page %d returned 0 posts — stopping listing crawl for %s", page, code)
            break

    if own_session:
        session.close()
    return posts


def _parse_listing_row(row, code: str) -> Optional[dict]:
    """Parse a single ``tr.listitem`` element.
    
    Current page structure (2026):
        <tr class="listitem">
          <td><div class="read">N</div></td>
          <td><div class="reply">N</div></td>
          <td><div class="title"><a data-postid="..." href="/news,...">title</a></div></td>
          <td><div class="author"><a href="...">author</a></div></td>
          <td><div class="update">MM-DD HH:MM</div></td>
        </tr>
    """
    # ── read count ──
    read_el = row.select_one("div.read")
    read_count = _parse_int(read_el.text) if read_el else 0

    # ── reply count ──
    reply_el = row.select_one("div.reply")
    reply_count = _parse_int(reply_el.text) if reply_el else 0

    # ── title & URL ──
    link_el = row.select_one("div.title > a")
    if not link_el:
        return None

    title = (link_el.get("title") or link_el.get_text()).strip()
    href = link_el.get("href", "")

    # Build absolute URL, handling protocol-relative and path-relative forms
    if href.startswith("//"):
        full_url = "https:" + href
    elif href.startswith("http"):
        full_url = href
    else:
        full_url = BASE_URL + href

    # Skip non-guba posts (caifuhao, external links)
    if "/news," not in full_url:
        return None

    # Extract post_id from data-postid attribute or URL fallback
    post_id = link_el.get("data-postid") or _extract_post_id(href, code)

    # ── author ──
    author_el = row.select_one("div.author > a")
    author = author_el.text.strip() if author_el else ""

    # ── post time (year-less: "MM-DD HH:MM") ──
    time_el = row.select_one("div.update")
    post_time_str = time_el.text.strip() if time_el else ""

    return {
        "post_id": post_id,
        "title": title,
        "url": full_url,
        "author": author,
        "read_count": read_count,
        "reply_count": reply_count,
        "post_time_str": post_time_str,
    }


def _extract_post_id(href: str, code: str) -> str:
    """Extract the numeric post ID from an EastMoney detail URL."""
    # Standard: /news,{code},{post_id}.html
    pattern = r"/news," + re.escape(code) + r",(\d+)\.html"
    m = re.search(pattern, href)
    if m:
        return m.group(1)
    # Fallback: last numeric segment before .html
    m = re.search(r"/(\d+)\.html", href)
    if m:
        return m.group(1)
    return ""  # empty → caller knows extraction failed


# ── detail page ──────────────────────────────────────────────────


def fetch_post_detail(url: str) -> Optional[dict]:
    """Fetch and parse an EastMoney post detail page.

    .. warning::

        **BROKEN (2026-06-15)** — 当前页面结构已不兼容，三个问题：

        1. **编码**: 页面返回 GBK 内容但 ``_safe_get`` 未强制设置 ``resp.encoding``，
           导致中文乱码，BS4 解析出的选择器内容全部为乱码。
        2. **正文不在静态 DOM 中**: ``.xeditor_content`` 里的正文由 JS 动态填充，
           静态 HTTP 请求只能拿到 ``<p><br></p>`` 空壳。
        3. **真实数据源**: 正文嵌入在 ``<script>var post_article = {...}`` 的 JSON
           中，字段为 ``post_content``（HTML 片段）。修复方案：
           用正则提取 ``var post_article`` JSON → ``json.loads`` →
           对 ``post_content`` 做 BS4 去标签取纯文本。

        在此之前请使用快速模式（``fetch_details=False``，仅抓列表页标题）。

    Args:
        url: Full URL of the post (``/news,{code},{post_id}.html``).

    Returns:
        Dict with keys ``title``, ``content``, ``post_time`` (datetime or None),
        or ``None`` on failure.
    """
    resp = _safe_get(url, DETAIL_DELAY)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # ── title ──
    title = ""
    # Primary: extract from <title> tag (format: "title_股票代码_东方财富网股吧")
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.text.split("_")[0].strip()
    # Fallback: h1/h2 in content area
    if not title:
        title_el = soup.select_one("#zwcontt h2") or soup.select_one(".newsauthor h2")
        if title_el:
            title = title_el.text.strip()

    # ── content ──
    content = ""
    content_el = (
        soup.select_one(".xeditor_content")
        or soup.select_one(".app_h5_article")
        or soup.select_one("#zwconbody > div")
        or soup.select_one(".article-body")
    )
    if content_el:
        content = content_el.get_text(separator="\n", strip=True)

    # ── post time ──
    post_time: Optional[datetime] = None
    time_el = soup.select_one("div.time") or soup.select_one(".newsauthor .time")
    if time_el:
        time_text = time_el.text.strip()
        post_time = parse_date(time_text)

    return {
        "title": title,
        "content": content,
        "post_time": post_time,
    }


# ── full pipeline ────────────────────────────────────────────────


def crawl_stock(code: str, name: str, max_list_pages: int = 25, fetch_details: bool = False) -> list[dict]:
    """Full crawl pipeline for one stock: listing → (optional) detail → assemble.

    When ``fetch_details=False`` (listing-only / fast mode), uses the listing
    title as content and infers the date from the listing time string.  This
    is ~10× faster and yields ~80 posts per page — adequate for bulk sentiment.

    .. warning::

        ``fetch_details=True`` is currently **BROKEN** (see :func:`fetch_post_detail`).
        Do not use until the ``var post_article`` JSON extraction approach is implemented.

    Args:
        code: 6-digit stock code.
        name: Human-readable stock name.
        max_list_pages: Max listing pages to scan (80 posts/page).
        fetch_details: If False, skip detail-page fetches.

    Returns:
        List of post dicts ready for ``save_posts()``.
    """
    logger.info("crawling %s (%s) %s", code, name, "(listing-only)" if not fetch_details else "")

    with requests.Session() as session:
        listing_posts = crawl_stock_list(code, max_pages=max_list_pages, session=session)
    logger.info("listing for %s: %d posts found", code, len(listing_posts))

    results: list[dict] = []

    if not fetch_details:
        # Fast mode: treat listing data as the post content
        for lp in listing_posts:
            post_time = parse_date(lp["post_time_str"])
            results.append({
                "platform": "eastmoney",
                "stock_code": code,
                "stock_name": name,
                "title": lp["title"],
                "content": lp["title"],  # fallback: use title as content
                "author": lp["author"],
                "post_time": post_time,
                "read_count": lp["read_count"],
                "reply_count": lp["reply_count"],
                "url": lp["url"],
            })
        return results

    # Full mode: fetch detail pages for richer content
    for i, lp in enumerate(listing_posts):
        logger.debug("[%d/%d] detail: %s", i + 1, len(listing_posts), lp["url"])
        try:
            detail = fetch_post_detail(lp["url"])
            if detail is None:
                continue

            # Prefer the detail page's full date; fall back to the listing time
            post_time = detail["post_time"]
            if post_time is None and lp.get("post_time_str"):
                post_time = parse_date(lp["post_time_str"])

            results.append({
                "platform": "eastmoney",
                "stock_code": code,
                "stock_name": name,
                "title": detail["title"] or lp["title"],
                "content": detail["content"] or lp["title"],
                "author": lp["author"],
                "post_time": post_time,
                "read_count": lp["read_count"],
                "reply_count": lp["reply_count"],
                "url": lp["url"],
            })
        except Exception:
            logger.exception("failed to process detail for %s", lp.get("url", ""))
            continue

    logger.info("detail fetch for %s: %d/%d succeeded", code, len(results), len(listing_posts))
    return results


# ── database ──────────────────────────────────────────────────────


def save_posts(posts: list[dict]) -> int:
    """Bulk-insert posts into PostgreSQL.  Returns number of new rows.

    Uses ``ON CONFLICT (url) DO NOTHING`` to skip duplicates,
    matching the ``posts`` table's unique constraint on ``url``.
    """
    if not posts:
        return 0

    df = pd.DataFrame(posts)

    if "post_time" in df.columns:
        df["post_time"] = pd.to_datetime(df["post_time"], errors="coerce")

    # Serialize post_time for the SQL text statement
    records: list[dict] = []
    for _, row in df.iterrows():
        d = row.to_dict()
        pt = d.get("post_time")
        if pt is None or (isinstance(pt, pd.Timestamp) and pd.isna(pt)):
            d["post_time"] = None
        elif isinstance(pt, pd.Timestamp):
            d["post_time"] = pt.to_pydatetime().isoformat()
        elif isinstance(pt, datetime):
            d["post_time"] = pt.isoformat()
        else:
            # covers pd.NaT / float("nan") / invalid types
            d["post_time"] = None
        records.append(d)

    stmt = text("""
        INSERT INTO posts (platform, stock_code, stock_name, title, content,
                           author, post_time, read_count, reply_count, url)
        VALUES (:platform, :stock_code, :stock_name, :title, :content,
                :author, :post_time, :read_count, :reply_count, :url)
        ON CONFLICT (url) DO NOTHING
    """)

    with engine.begin() as conn:
        # Count existing URLs before insert for accurate inserted count
        urls = [r["url"] for r in records]
        existing_count = conn.execute(
            text("SELECT COUNT(*) FROM posts WHERE url = ANY(:urls)"),
            {"urls": urls},
        ).scalar()

        # Single bulk INSERT — SQLAlchemy will execute with executemany
        conn.execute(stmt, records)

    return len(records) - existing_count


# ── main ─────────────────────────────────────────────────────────


def main(
    max_stocks: int = 300,
    max_list_pages: int = 25,
    start_from: int = 0,
    fetch_details: bool = False,
):
    """Crawl 东方财富股吧 posts for CSI 300 stocks.

    By default runs in **listing-only** (fast) mode: ~2.5s per listing page ×
    80 posts = ~200 posts/min.  Set ``fetch_details=True`` for richer content
    (~10× slower).

    Args:
        max_stocks: Maximum number of stocks to crawl (default 300 ≈ full index).
        max_list_pages: Listing pages per stock (80 posts/page).
        start_from: Resume from this index in the stock list (0-based).
        fetch_details: If True, fetch each post's detail page for full content.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    stocks = get_stock_list()
    logger.info("stock pool: %d stocks, fetch_details=%s", len(stocks), fetch_details)

    total_posts = 0
    for idx, (_, row) in enumerate(stocks.iterrows()):
        if idx < start_from:
            continue
        if idx >= start_from + max_stocks:
            break

        code, name = row["code"], row["name"]
        logger.info(
            "── [%d/%d] %s %s ──",
            idx + 1,
            min(start_from + max_stocks, len(stocks)),
            code,
            name,
        )

        try:
            posts = crawl_stock(code, name, max_list_pages=max_list_pages, fetch_details=fetch_details)
            n = save_posts(posts)
            total_posts += n
            logger.info("saved %d new posts for %s (running total: %d)", n, code, total_posts)
        except Exception:
            logger.exception("unexpected error crawling %s", code)
            time.sleep(5)

        # Rate limit between different stocks
        time.sleep(STOCK_DELAY)

    logger.info("DONE — total new posts saved: %d", total_posts)


if __name__ == "__main__":
    main()
