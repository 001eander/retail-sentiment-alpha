"""
A-share market data fetcher using baostock.

Fetches daily OHLCV data for CSI 300 constituents and the CSI 300 index,
then stores results in PostgreSQL via SQLAlchemy.

Usage
-----
    python -m data.market_data

This will download the last ~1 year of data for all CSI 300 stocks and the
CSI 300 index, inserting into the ``market_daily`` and ``index_daily`` tables.

.. note::
   Previously used AkShare (东方财富 push2his API), but that endpoint is
   blocked/unreachable from this network.  Switched to baostock which uses
   its own data API and works reliably.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any

import baostock as bs
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://alpha:alpha123@localhost:5432/sentiment_alpha",
)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def _bs_symbol(code: str) -> str:
    """Convert 6-digit code to baostock symbol: ``sh.600519`` or ``sz.000001``."""
    code = str(code).zfill(6)
    return f"sh.{code}" if code.startswith("6") else f"sz.{code}"

# ---------------------------------------------------------------------------
# Stock list helpers
# ---------------------------------------------------------------------------


def get_stock_list() -> list[tuple[str, str]]:
    """Fetch CSI 300 constituent stocks via AkShare.

    Returns
    -------
    list[tuple[str, str]]
        ``(stock_code, stock_name)`` for each constituent.  Falls back to a
        hardcoded top-100 list when AkShare is unavailable.
    """
    try:
        import akshare as ak

        df = ak.index_stock_cons_weight_csindex("000300")
        df = df.rename(columns={"成分券代码": "code", "成分券名称": "name"})
        df["code"] = df["code"].astype(str).str.zfill(6)
        return list(df[["code", "name"]].itertuples(index=False, name=None))
    except Exception:
        logger.warning("AkShare CSI300 failed, using hardcoded fallback list")
        return _fallback_stocks()


def _fallback_stocks() -> list[tuple[str, str]]:
    """Hardcoded top 100 CSI 300 stocks (by approximate weight)."""
    codes_str = (
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
    codes = [c.strip() for c in codes_str.split(",") if c.strip()]
    names = [f"stock_{c}" for c in codes]
    return list(zip(codes, names))


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def fetch_stock_daily(
    code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily OHLCV for a single A-share stock via baostock.

    Parameters
    ----------
    code : str
        6-digit stock code, e.g. ``"600519"``.
    start_date : str
        Start date in ``YYYY-MM-DD`` format (baostock format).
    end_date : str
        End date in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DataFrame
        Columns: ``stock_code, trade_date, open, high, low, close,
        pre_close, volume, amount, turnover``.
        Returns an empty DataFrame on error.
    """
    try:
        symbol = _bs_symbol(code)
        rs = bs.query_history_k_data_plus(
            symbol,
            "date,open,high,low,close,preclose,volume,amount,turn",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",  # 不复权
        )

        if rs.error_code != "0":
            logger.warning("baostock query error for %s: %s", code, rs.error_msg)
            return _empty_market_df()

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            logger.warning("No data returned for %s", code)
            return _empty_market_df()

        df = pd.DataFrame(rows, columns=rs.fields)

        # Rename baostock columns → our schema
        df = df.rename(columns={
            "date": "trade_date",
            "preclose": "pre_close",
            "turn": "turnover",
        })

        # Type conversions
        numeric_cols = ["open", "high", "low", "close", "pre_close", "volume", "amount", "turnover"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["volume"] = df["volume"].astype("int64")
        df["stock_code"] = code

        return df[
            [
                "stock_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "volume",
                "amount",
                "turnover",
            ]
        ]

    except Exception:
        logger.exception("Failed to fetch data for %s", code)
        return _empty_market_df()


def _empty_market_df() -> pd.DataFrame:
    """Return an empty DataFrame with the expected market_daily column set."""
    return pd.DataFrame(
        columns=[
            "stock_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "volume",
            "amount",
            "turnover",
        ]
    )


def fetch_index_daily(
    symbol: str = "000300",
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    """Fetch daily close for a Chinese index via baostock.

    Parameters
    ----------
    symbol : str
        Index code, e.g. ``"000300"`` (CSI 300).
    start_date : str
        Start date in ``YYYY-MM-DD`` format.
    end_date : str
        End date in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DataFrame
        Columns: ``idx_code, trade_date, close``.
    """
    try:
        bs_sym = f"sh.{symbol}"
        rs = bs.query_history_k_data_plus(
            bs_sym,
            "date,close",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",
        )

        if rs.error_code != "0":
            logger.warning("baostock index query error: %s", rs.error_msg)
            return _empty_index_df()

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            logger.warning("No index data returned for %s", symbol)
            return _empty_index_df()

        df = pd.DataFrame(rows, columns=rs.fields)
        df = df.rename(columns={"date": "trade_date"})
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["idx_code"] = symbol

        return df[["idx_code", "trade_date", "close"]]

    except Exception:
        logger.exception("Failed to fetch index data for %s", symbol)
        return _empty_index_df()


def _empty_index_df() -> pd.DataFrame:
    """Return an empty DataFrame with the expected index_daily column set."""
    return pd.DataFrame(columns=["idx_code", "trade_date", "close"])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_market_data(df: pd.DataFrame) -> int:
    """Bulk-insert stock daily data into the ``market_daily`` table.

    Uses ``ON CONFLICT (stock_code, trade_date) DO NOTHING`` so that
    duplicate rows are silently ignored.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns matching the ``market_daily`` schema.

    Returns
    -------
    int
        Number of rows actually inserted.
    """
    if df.empty:
        logger.warning("save_market_data called with empty DataFrame")
        return 0

    required = {
        "stock_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "volume",
        "amount",
        "turnover",
    }
    missing = required - set(df.columns)
    if missing:
        logger.error("Missing columns in market data: %s", missing)
        return 0

    rows: list[dict[str, Any]] = df.to_dict("records")

    insert_sql = text("""
        INSERT INTO market_daily
            (stock_code, trade_date, open, high, low, close,
             pre_close, volume, amount, turnover)
        VALUES
            (:stock_code, :trade_date, :open, :high, :low, :close,
             :pre_close, :volume, :amount, :turnover)
        ON CONFLICT (stock_code, trade_date) DO NOTHING
    """)

    with engine.begin() as conn:
        result = conn.execute(insert_sql, rows)

    logger.info("Inserted %d rows into market_daily", result.rowcount)
    return result.rowcount


def save_index_data(df: pd.DataFrame) -> int:
    """Bulk-insert index daily data into the ``index_daily`` table.

    Uses ``ON CONFLICT (idx_code, trade_date) DO NOTHING`` so that
    duplicate rows are silently ignored.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``idx_code``, ``trade_date``, ``close``.

    Returns
    -------
    int
        Number of rows actually inserted.
    """
    if df.empty:
        logger.warning("save_index_data called with empty DataFrame")
        return 0

    required = {"idx_code", "trade_date", "close"}
    missing = required - set(df.columns)
    if missing:
        logger.error("Missing columns in index data: %s", missing)
        return 0

    rows: list[dict[str, Any]] = df.to_dict("records")

    insert_sql = text("""
        INSERT INTO index_daily (idx_code, trade_date, close)
        VALUES (:idx_code, :trade_date, :close)
        ON CONFLICT (idx_code, trade_date) DO NOTHING
    """)

    with engine.begin() as conn:
        result = conn.execute(insert_sql, rows)

    logger.info("Inserted %d rows into index_daily", result.rowcount)
    return result.rowcount


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def download_all(
    start_date: str = "2024-06-01",
    end_date: str = "2026-06-15",
    max_stocks: int = 300,
    rate_limit: float = 0.2,
) -> None:
    """Download market data for CSI 300 stocks and the CSI 300 index.

    Parameters
    ----------
    start_date : str
        Start date in ``YYYY-MM-DD`` format (baostock format).
    end_date : str
        End date in ``YYYY-MM-DD`` format.
    max_stocks : int
        Maximum number of stocks to process.
    rate_limit : float
        Seconds to ``sleep()`` between consecutive stock requests.
    """
    # ── baostock login ──
    lg = bs.login()
    if lg.error_code != "0":
        logger.error("baostock login failed: %s", lg.error_msg)
        return
    logger.info("baostock login ok")

    try:
        _do_download(start_date, end_date, max_stocks, rate_limit)
    finally:
        bs.logout()
        logger.info("baostock logout")


def _do_download(
    start_date: str,
    end_date: str,
    max_stocks: int,
    rate_limit: float,
) -> None:
    """Inner download logic (baostock session already active)."""
    stocks = get_stock_list()
    logger.info("Got %d stocks from CSI 300 constituent list", len(stocks))

    total = min(len(stocks), max_stocks)
    logger.info("Will process up to %d stocks", total)

    # ---- download index data first ----
    logger.info("Downloading CSI 300 index (%s – %s)", start_date, end_date)
    try:
        idx_df = fetch_index_daily(
            symbol="000300", start_date=start_date, end_date=end_date
        )
        if idx_df.empty:
            logger.warning("Index data empty, skipping save")
        else:
            save_index_data(idx_df)
    except Exception as exc:
        logger.error("Failed to fetch / save index data: %s", exc)

    # ---- download per-stock data ----
    success = 0
    skipped = 0

    for i, (code, name) in enumerate(stocks[:max_stocks]):
        logger.info("[%d/%d] Processing %s (%s)", i + 1, total, code, name)
        try:
            df = fetch_stock_daily(
                code=code,
                start_date=start_date,
                end_date=end_date,
            )
            if df.empty:
                logger.warning("No data for %s (%s), skipping", code, name)
                skipped += 1
            else:
                n = save_market_data(df)
                success += 1
                logger.debug("Inserted %d rows for %s", n, code)
        except Exception as exc:
            logger.error("Failed to process %s (%s): %s", code, name, exc)
            skipped += 1

        if i < total - 1:
            time.sleep(rate_limit)

    logger.info(
        "Done. Successfully processed %d stocks, skipped %d.",
        success,
        skipped,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    logger.info("Starting market data download from %s to %s", start_date, end_date)
    download_all(start_date=start_date, end_date=end_date)
