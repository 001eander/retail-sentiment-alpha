"""
淘股吧 (tgb.cn) BBS 爬虫 — 遍历最新帖子并匹配 CSI 300 股票。

用法：
    python -m crawlers.taoguba_bbs

从 /bbs/{page}/{sort} 获取全量最新帖子，再从帖子内容中提取股票代码，
实现远高于按股票代码搜索的覆盖率。
"""

import logging
import random
import re
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from .config import BASE_URL, DETAIL_DELAY, HEADERS, LIST_DELAY, MAX_RETRIES
from .taoguba import _parse_int, _safe_get, parse_post, save_posts

logger = logging.getLogger(__name__)

# ── compiled regexes ────────────────────────────────────────────

_STOCK_CODE_RE = re.compile(r"\b(0[0-9]{5}|3[0-9]{5}|6[0-9]{5})\b")
_PAGINATION_RE = re.compile(r"共\s*\d+/(\d+)页")
_REPLY_SPAN_RE = re.compile(r"\((\d+)\)")


# ── CSI 300 list ────────────────────────────────────────────────


def get_csi300_list() -> list[tuple[str, str]]:
    """Return list of (code, name) tuples for CSI 300 constituents.

    Tries akshare first; falls back to a hardcoded list of heavyweight stocks.
    """
    try:
        import akshare as ak

        df = ak.index_stock_cons_weight_csindex("000300")
        df = df.rename(columns={"成分券代码": "code", "成分券名称": "name"})
        df["code"] = df["code"].astype(str).str.zfill(6)
        result = list(zip(df["code"], df["name"]))
        logger.info("loaded %d CSI 300 constituents via akshare", len(result))
        return result
    except ImportError:
        logger.warning("akshare not available; using hardcoded list")
    except Exception:
        logger.warning("akshare CSI 300 lookup failed; using hardcoded list")

    return _top_stocks_fallback()


def _top_stocks_fallback() -> list[tuple[str, str]]:
    """Top CSI 300 stocks by weight as fallback list."""
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
    )
    return [(c, f"stock_{c}") for c in codes.split(",") if c]


# ── BBS listing page ────────────────────────────────────────────


def fetch_bbs_list_page(page: int, sort: int = 1) -> list[dict]:
    """Fetch one BBS listing page and return parsed post rows.

    Args:
        page: Page number (1-based).
        sort: 1 = post date (newest first), 0 = last reply date.

    Returns:
        List of dicts with keys:
            topic_id, title, url, author, post_date, reply_count, read_count
    """
    url = f"{BASE_URL}/bbs/{page}/{sort}"
    resp = _safe_get(url, LIST_DELAY)
    if resp is None:
        logger.warning("bbs page %d returned no response", page)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select(".Nbbs-tiezi-lists")
    if not rows:
        logger.debug("bbs page %d has no post rows", page)
        return []

    results = []
    for row in rows:
        try:
            post = _parse_bbs_row(row)
            if post:
                results.append(post)
        except Exception:
            logger.exception("failed to parse a bbs row on page %d", page)
            continue

    return results


def _parse_bbs_row(row) -> Optional[dict]:
    """Parse a single ``.Nbbs-tiezi-lists`` div into a post dict."""
    # ── title & link ──
    link_el = row.select_one("a.overhide.mw300")
    if not link_el:
        return None

    title = (link_el.get("title") or link_el.text).strip()
    href = link_el.get("href", "")
    # Relative URL: "a/{topicID}" — extract topic_id
    topic_id = href.rstrip("/").split("/")[-1] if "/" in href else href
    full_url = urljoin(BASE_URL, href)

    # ── reply count (from <span> adjacent to link) ──
    reply_count = 0
    span_el = row.select_one("a.overhide.mw300 + span")
    if span_el:
        m = _REPLY_SPAN_RE.search(span_el.get_text())
        if m:
            reply_count = int(m.group(1))

    # ── replies / views from talk cell ──
    talk_el = row.select_one(".middle-list-talk")
    read_count = 0
    if talk_el:
        parts = re.split(r"\s*/\s*", talk_el.text.strip())
        if len(parts) >= 2:
            read_count = _parse_int(parts[1])
            if reply_count == 0:
                reply_count = _parse_int(parts[0])

    # ── post date ──
    post_date = ""
    date_el = row.select_one(".middle-list-post")
    if date_el:
        post_date = date_el.text.strip()

    # ── author ──
    author = ""
    author_el = row.select_one(".middle-list-user a")
    if author_el:
        author = author_el.text.strip()

    return {
        "topic_id": topic_id,
        "title": title,
        "url": full_url,
        "author": author,
        "post_date": post_date,
        "reply_count": reply_count,
        "read_count": read_count,
    }


# ── pagination ──────────────────────────────────────────────────


def get_bbs_total_pages(sort: int = 1) -> int:
    """Fetch BBS page 1 and extract total page count from pagination bar.

    Returns total pages, or a safe high estimate (5000) if extraction fails.
    """
    url = f"{BASE_URL}/bbs/1/{sort}"
    resp = _safe_get(url, LIST_DELAY)
    if resp is None:
        logger.warning("could not fetch bbs page 1 to determine total pages")
        return 5000

    m = _PAGINATION_RE.search(resp.text)
    if m:
        total = int(m.group(1))
        logger.info("bbs has %d pages (sort=%d)", total, sort)
        return total

    logger.warning("could not find pagination bar, using default max")
    return 5000


# ── stock code extraction ───────────────────────────────────────


def extract_stock_codes(text: str, stock_list: list[tuple[str, str]]) -> set[str]:
    """Extract matching CSI 300 stock codes from text.

    Strategy (priority order):
    1. Regex for 6-digit codes (0xxxxx, 3xxxxx, 6xxxxx), keep only those in
       stock_list.
    2. Substring-match stock names from stock_list against the text.

    Returns a set of matching 6-digit codes (empty if none found).
    """
    if not text or not stock_list:
        return set()

    # Build lookup structures once
    code_set = {code for code, _ in stock_list}
    name_to_code = {name: code for code, name in stock_list}

    matched: set[str] = set()

    # Strategy 1: regex 6-digit codes, intersect with known codes
    for m in _STOCK_CODE_RE.finditer(text):
        code = m.group(0)
        if code in code_set:
            matched.add(code)

    # Strategy 2: substring match of stock names
    for name, code in name_to_code.items():
        if name in text:
            matched.add(code)

    return matched


# ── date inference ──────────────────────────────────────────────


def _infer_date(mmdd_hhmm: str) -> Optional[datetime]:
    """Infer a full datetime from a ``MM-DD HH:MM`` BBS listing string.

    Since we crawl newest-first (reverse chronological), if the parsed month
    is *greater* than the current month, the post must be from the *previous*
    year.  Otherwise it's from the current year.

    Returns ``None`` if the string cannot be parsed.
    """
    if not mmdd_hhmm or not mmdd_hhmm.strip():
        return None

    now = datetime.now()
    try:
        dt = datetime.strptime(mmdd_hhmm.strip(), "%m-%d %H:%M")
    except ValueError:
        return None

    year = now.year
    if dt.month > now.month:
        year -= 1

    return dt.replace(year=year)


# ── main crawl ──────────────────────────────────────────────────


def _title_mentions_stock(title: str, stock_list: list[tuple[str, str]]) -> bool:
    """Quick check: does the title contain any stock code or stock name?

    This is the *first-pass filter* that avoids a detail-page HTTP request for
    posts unlikely to discuss a tracked stock.
    """
    if not title:
        return False
    if _STOCK_CODE_RE.search(title):
        return True
    for _, name in stock_list:
        if not name:
            continue
        if name in title:
            return True
    return False


def crawl_bbs(max_pages: int = 500, stock_list: Optional[list[tuple[str, str]]] = None) -> int:
    """Main BBS crawl loop.

    Workflow
    --------
    1. Determine total BBS pages (clamped to *max_pages*).
    2. Iterate pages newest-first (page 1 = newest).
    3. For each listing page:
       a. Parse all post rows.
       b. Title-only filter — skip posts whose title lacks any stock code/name.
       c. Fetch detail page for remaining candidates.
       d. Extract stock codes from full text (title + content).
       e. Save matched posts to database.

    Args:
        max_pages:  Maximum listing pages to crawl.
        stock_list: (code, name) pairs. Auto-fetched if ``None``.

    Returns:
        Number of newly inserted posts.
    """
    if stock_list is None:
        stock_list = get_csi300_list()

    logger.info("CSI 300 stock pool: %d stocks", len(stock_list))

    total_pages = get_bbs_total_pages(sort=1)
    total_pages = min(total_pages, max_pages)
    logger.info("will crawl %d BBS pages (newest-first)", total_pages)

    total_scanned = 0
    total_matched = 0
    total_saved = 0

    for page in range(1, total_pages + 1):
        logger.info("── page %d/%d ──", page, total_pages)

        # ── fetch listing ──
        posts = fetch_bbs_list_page(page, sort=1)
        if not posts:
            logger.info("page %d is empty — stopping", page)
            break

        total_scanned += len(posts)
        logger.debug("page %d: %d posts total", page, len(posts))

        # ── title-only filter ──
        candidates = [p for p in posts if _title_mentions_stock(p["title"], stock_list)]
        logger.debug("page %d: %d candidates after title filter", page, len(candidates))

        # ── detail fetch + stock-code extraction for candidates ──
        to_save = []
        for post in candidates:
            try:
                detail = parse_post(post["url"])
                if not detail:
                    continue

                combined = f"{detail['title']} {detail['content']}"
                codes = extract_stock_codes(combined, stock_list)

                if codes:
                    total_matched += 1
                    for code in codes:
                        record = dict(detail)  # shallow copy
                        record["stock_code"] = code
                        # look up stock name
                        for sc, sn in stock_list:
                            if sc == code:
                                record["stock_name"] = sn
                                break
                        to_save.append(record)

                # rate-limit between detail requests
                time.sleep(random.uniform(*DETAIL_DELAY))

            except Exception:
                logger.exception("failed to process %s", post.get("url", ""))
                continue

        # ── save batch ──
        if to_save:
            n = save_posts(to_save)
            total_saved += n
            logger.info(
                "page %d: saved %d new (scanned %d, candidates %d, total saved %d)",
                page,
                n,
                len(posts),
                len(candidates),
                total_saved,
            )
        else:
            logger.info(
                "page %d: no matches (scanned %d, candidates %d)",
                page,
                len(posts),
                len(candidates),
            )

    logger.info(
        "DONE — scanned: %d, matched: %d, total saved: %d",
        total_scanned,
        total_matched,
        total_saved,
    )
    return total_saved


# ── entry point ─────────────────────────────────────────────────


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not HEADERS.get("Cookie"):
        logger.error("TGB_COOKIE is empty — set it in .env first!")
        logger.error("1. 浏览器登录 https://www.tgb.cn/")
        logger.error("2. F12 → Application → Cookies → 复制整段 Cookie 到 .env 的 TGB_COOKIE= 后")
        return

    crawl_bbs(max_pages=500)


if __name__ == "__main__":
    main()
