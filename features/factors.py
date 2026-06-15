"""
Daily sentiment factor computation from social-media posts.

Aggregates post-level sentiment data into daily stock-level factors,
stores them in the ``daily_factors`` table, and provides a helper to
build a modeling dataset joined with market data.

Functions
---------
compute_daily_factors
    Aggregate raw posts into daily sentiment factors.
save_factors
    Upsert factor records into the ``daily_factors`` table.
build_model_dataset
    Join factors with market data and add control / target variables.
run_pipeline
    Run the full compute → save → report workflow.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
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

# ---------------------------------------------------------------------------
# Core factor computation
# ---------------------------------------------------------------------------


def compute_daily_factors() -> pd.DataFrame:
    """Aggregate posts with non-null sentiment into daily stock-level factors.

    Reads from ``posts`` where ``sentiment IS NOT NULL``, groups by
    ``(stock_code, DATE(post_time))``, and computes:

    * **post_volume_anomaly** — daily post count divided by the 20-day rolling
      mean post count for that stock.  When the rolling mean is NaN (fewer than
      20 days of history) the stock's own overall mean is used as the
      denominator.  If that too is zero/NaN, the anomaly is set to 0.
    * **sentiment_score** — mean of ``sentiment`` for the stock+date.
    * **sentiment_divergence** — standard deviation of ``sentiment`` for the
      stock+date (0 when only one post exists).
    * **interaction_intensity** — mean of ``reply_count``; if all replies are
      zero for that day, falls back to the mean of ``read_count``.

    Returns
    -------
    pd.DataFrame
        Columns: ``stock_code``, ``trade_date``, ``post_volume_anomaly``,
        ``sentiment_score``, ``sentiment_divergence``, ``interaction_intensity``.
    """
    logger.info("Querying posts with non-null sentiment …")

    sql = text("""
        SELECT
            stock_code,
            DATE(post_time)                                AS trade_date,
            COUNT(*)::int                                  AS daily_count,
            AVG(sentiment)::double precision               AS sentiment_score,
            STDDEV_SAMP(sentiment)::double precision       AS sentiment_divergence,
            AVG(reply_count)::double precision             AS avg_reply,
            AVG(read_count)::double precision              AS avg_read
        FROM posts
        WHERE sentiment IS NOT NULL
        GROUP BY stock_code, DATE(post_time)
        ORDER BY stock_code, trade_date
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()

    if not rows:
        logger.warning("No posts with sentiment found — returning empty DataFrame.")
        return _empty_factors_df()

    agg = pd.DataFrame(
        rows,
        columns=[
            "stock_code",
            "trade_date",
            "daily_count",
            "sentiment_score",
            "sentiment_divergence",
            "avg_reply",
            "avg_read",
        ],
    )
    agg["trade_date"] = pd.to_datetime(agg["trade_date"])

    # ── post_volume_anomaly ─────────────────────────────────────────────
    _compute_volume_anomaly(agg)

    # ── interaction_intensity — reply_count primary, read_count fallback ─
    _compute_interaction_intensity(agg)

    # ── sentiment_score / divergence — carry through as-is ──────────────
    #   Replace NaN in stddev (single-post days) with 0.
    agg["sentiment_divergence"] = agg["sentiment_divergence"].fillna(0.0)

    result = agg[
        [
            "stock_code",
            "trade_date",
            "post_volume_anomaly",
            "sentiment_score",
            "sentiment_divergence",
            "interaction_intensity",
        ]
    ].copy()

    logger.info(
        "Computed daily factors: %d stock-date rows.",
        len(result),
    )
    return result


def _empty_factors_df() -> pd.DataFrame:
    """Return an empty DataFrame with the expected factor columns."""
    return pd.DataFrame(
        columns=[
            "stock_code",
            "trade_date",
            "post_volume_anomaly",
            "sentiment_score",
            "sentiment_divergence",
            "interaction_intensity",
        ]
    )


def _compute_volume_anomaly(agg: pd.DataFrame) -> None:
    """Compute ``post_volume_anomaly`` in-place on *agg*.

    Strategy
    --------
    1. 20-day rolling mean per stock of ``daily_count``.
    2. For each stock, compute its overall mean as fallback.
    3. anomaly = daily_count / rolling_mean (falling back to overall_mean).
    4. If denominator is still 0 / NaN, set anomaly = 0.
    """
    # Sort so rolling works correctly
    agg.sort_values(["stock_code", "trade_date"], inplace=True)
    agg.reset_index(drop=True, inplace=True)

    # 20-day rolling mean per stock (closed=right so it uses up to today)
    rolling_mean = (
        agg.groupby("stock_code")["daily_count"]
        .transform(lambda s: s.rolling(20, min_periods=1).mean())
    )

    # Overall mean per stock (fallback for stocks with < 20 days)
    overall_mean = agg.groupby("stock_code")["daily_count"].transform("mean")

    # Use rolling mean where available, otherwise overall mean
    denominator = rolling_mean.where(rolling_mean.notna(), overall_mean)

    # Guard division by zero
    safe = denominator > 0
    anomaly = np.where(safe, agg["daily_count"] / denominator, 0.0)

    agg["post_volume_anomaly"] = np.where(
        np.isfinite(anomaly), anomaly, 0.0
    )


def _compute_interaction_intensity(agg: pd.DataFrame) -> None:
    """Compute ``interaction_intensity`` in-place on *agg*.

    Uses ``avg_reply`` as the primary metric.  Falls back to ``avg_read``
    when the reply mean is zero (i.e. all replies are zero for that day).
    """
    # Where avg_reply is 0 (all replies zero), fall back to avg_read
    agg["interaction_intensity"] = np.where(
        agg["avg_reply"] > 0,
        agg["avg_reply"],
        agg["avg_read"],
    )
    # If read_count also unavailable/NaN, fill with 0
    agg["interaction_intensity"] = agg["interaction_intensity"].fillna(0.0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_factors(df: pd.DataFrame) -> int:
    """Upsert daily factor records into the ``daily_factors`` table.

    Uses ``ON CONFLICT (stock_code, trade_date) DO UPDATE SET`` so that
    re-running the pipeline refreshes existing rows.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``stock_code``, ``trade_date``,
        ``post_volume_anomaly``, ``sentiment_score``,
        ``sentiment_divergence``, ``interaction_intensity``.

    Returns
    -------
    int
        Number of rows affected (inserted + updated).
    """
    if df.empty:
        logger.warning("save_factors called with empty DataFrame")
        return 0

    required = {
        "stock_code",
        "trade_date",
        "post_volume_anomaly",
        "sentiment_score",
        "sentiment_divergence",
        "interaction_intensity",
    }
    missing = required - set(df.columns)
    if missing:
        logger.error("Missing columns in factors DataFrame: %s", missing)
        return 0

    # Convert trade_date to date (in case it's datetime)
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    rows: list[dict[str, Any]] = df.to_dict("records")

    upsert_sql = text("""
        INSERT INTO daily_factors
            (stock_code, trade_date,
             post_volume_anomaly, sentiment_score,
             sentiment_divergence, interaction_intensity)
        VALUES
            (:stock_code, :trade_date,
             :post_volume_anomaly, :sentiment_score,
             :sentiment_divergence, :interaction_intensity)
        ON CONFLICT (stock_code, trade_date) DO UPDATE SET
            post_volume_anomaly    = EXCLUDED.post_volume_anomaly,
            sentiment_score        = EXCLUDED.sentiment_score,
            sentiment_divergence   = EXCLUDED.sentiment_divergence,
            interaction_intensity  = EXCLUDED.interaction_intensity
    """)

    with engine.begin() as conn:
        result = conn.execute(upsert_sql, rows)

    logger.info("Upserted %d rows into daily_factors", result.rowcount)
    return result.rowcount


# ---------------------------------------------------------------------------
# Modeling dataset
# ---------------------------------------------------------------------------


def build_model_dataset(
    start_date: str = "2025-06-15",
    end_date: str = "2026-06-15",
    feature_groups: list[str] | None = None,
) -> pd.DataFrame:
    """Build a scikit-learn-ready dataset by joining factors with market data.

    Steps
    -----
    1. Read ``daily_factors`` and ``market_daily`` for the date range.
    2. Inner-join on ``(stock_code, trade_date)``.
    3. Add control variables:
       - ``ret_1d``: today's return *(close - pre_close) / pre_close*.
       - ``fwd_ret_1d``: next day's return (shifted -1 per stock).  **Target**.
       - ``volatility_20d``: 20-day rolling std of daily returns.
       - ``volume_anomaly_20d``: today's volume / 20-day rolling mean volume.
       - ``turnover_20d``: 20-day rolling mean of turnover.
    4. Join with ``index_daily`` (CSI 300) for ``mkt_ret_1d``.
    5. Drop rows where ``fwd_ret_1d`` is NaN (boundary days).

    Parameters
    ----------
    start_date : str
        Start date inclusive (``YYYY-MM-DD``).
    end_date : str
        End date inclusive (``YYYY-MM-DD``).
    feature_groups : list[str] or None
        Which factor groups to include in the output.  Valid values are
        ``"sentiment"`` and ``"traditional"``.  When ``None`` (default) all
        columns are kept (backward compatible).  The five control variables
        (``ret_1d``, ``volatility_20d``, ``volume_anomaly_20d``,
        ``turnover_20d``, ``mkt_ret_1d``) and raw price columns are always
        included regardless of this setting.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame ready for scikit-learn.  Missing market-return values
        are filled with 0 (no index return information).
    """
    logger.info("Building model dataset: %s → %s", start_date, end_date)

    # ── 1. Load factors ─────────────────────────────────────────────────
    factors_sql = text("""
        SELECT *
        FROM daily_factors
        WHERE trade_date BETWEEN :start_date AND :end_date
        ORDER BY stock_code, trade_date
    """)

    # ── 2. Load market data ─────────────────────────────────────────────
    market_sql = text("""
        SELECT *
        FROM market_daily
        WHERE trade_date BETWEEN :start_date AND :end_date
        ORDER BY stock_code, trade_date
    """)

    # ── 3. Load index data ──────────────────────────────────────────────
    index_sql = text("""
        SELECT trade_date, close
        FROM index_daily
        WHERE trade_date BETWEEN :start_date AND :end_date
        ORDER BY trade_date
    """)

    with engine.connect() as conn:
        factors_df = pd.read_sql(factors_sql, conn, params={
            "start_date": start_date,
            "end_date": end_date,
        })
        market_df = pd.read_sql(market_sql, conn, params={
            "start_date": start_date,
            "end_date": end_date,
        })
        index_df = pd.read_sql(index_sql, conn, params={
            "start_date": start_date,
            "end_date": end_date,
        })

    if factors_df.empty:
        logger.warning("No factors found in date range — returning empty DataFrame.")
        return pd.DataFrame()

    if market_df.empty:
        logger.warning("No market data found in date range — returning empty DataFrame.")
        return pd.DataFrame()

    # Normalise date columns
    for df_ in (factors_df, market_df, index_df):
        df_["trade_date"] = pd.to_datetime(df_["trade_date"])

    # ── 4. Inner join factors + market ──────────────────────────────────
    df = pd.merge(
        factors_df,
        market_df,
        on=["stock_code", "trade_date"],
        how="inner",
        suffixes=("", "_mkt"),
    )

    # Sort for window functions
    df.sort_values(["stock_code", "trade_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 5. Control variables ────────────────────────────────────────────

    # ret_1d: today's return
    df["ret_1d"] = np.where(
        df["pre_close"].notna() & (df["pre_close"] != 0),
        (df["close"] - df["pre_close"]) / df["pre_close"],
        0.0,
    )

    # fwd_ret_1d: next day's return (shift per stock) — THE TARGET
    df["fwd_ret_1d"] = df.groupby("stock_code")["ret_1d"].transform(
        lambda s: s.shift(-1)
    )

    # volatility_20d: rolling 20-day std of daily returns
    df["volatility_20d"] = (
        df.groupby("stock_code")["ret_1d"]
        .transform(lambda s: s.rolling(20, min_periods=1).std())
    )

    # volume_anomaly_20d: volume / 20-day rolling mean volume
    vol_rolling_mean = (
        df.groupby("stock_code")["volume"]
        .transform(lambda s: s.rolling(20, min_periods=1).mean())
    )
    safe_vol = vol_rolling_mean > 0
    df["volume_anomaly_20d"] = np.where(
        safe_vol,
        df["volume"] / vol_rolling_mean,
        0.0,
    )
    df["volume_anomaly_20d"] = np.where(
        np.isfinite(df["volume_anomaly_20d"]),
        df["volume_anomaly_20d"],
        0.0,
    )

    # turnover_20d: rolling 20-day mean of turnover
    df["turnover_20d"] = (
        df.groupby("stock_code")["turnover"]
        .transform(lambda s: s.rolling(20, min_periods=1).mean())
    )

    # ── 6. Index market return ──────────────────────────────────────────
    index_df.rename(columns={"close": "idx_close"}, inplace=True)
    index_df["idx_ret_1d"] = index_df["idx_close"].pct_change().fillna(0.0)

    # Keep only date + market return
    index_returns = index_df[["trade_date", "idx_ret_1d"]].copy()

    df = pd.merge(df, index_returns, on="trade_date", how="left")
    # Fill missing market-return days (e.g. weekends / holidays) with 0
    df["mkt_ret_1d"] = df["idx_ret_1d"].fillna(0.0)

    # ── 7. Traditional price-volume factors ──────────────────────────────
    df = _compute_traditional_factors(df)

    # ── 8. Clean up ─────────────────────────────────────────────────────

    # Drop the last day of each stock (no fwd_ret_1d)
    before = len(df)
    df = df.dropna(subset=["fwd_ret_1d"])
    after = len(df)
    if before - after:
        logger.info("Dropped %d boundary rows with no forward return.", before - after)

    # Drop helper columns
    drop_cols = [c for c in ["idx_close", "idx_ret_1d"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Drop rows with infinite values (safety)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=numeric_cols, how="any")

    # ── 9. Filter by feature groups (ablation support) ─────────────────────
    if feature_groups is not None:
        always_keep = [
            "stock_code",
            "trade_date",
            "fwd_ret_1d",
            # controls
            "ret_1d",
            "volatility_20d",
            "volume_anomaly_20d",
            "turnover_20d",
            "mkt_ret_1d",
            # raw price data
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "volume",
            "amount",
            "turnover",
        ]
        keep: list[str] = [c for c in always_keep if c in df.columns]

        if "sentiment" in feature_groups:
            keep += [
                "post_volume_anomaly",
                "sentiment_score",
                "sentiment_divergence",
                "interaction_intensity",
            ]
        if "traditional" in feature_groups:
            keep += [
                "ret_5d",
                "ret_20d",
                "max_ret_5d",
                "min_ret_5d",
                "illiquidity_20d",
                "skewness_20d",
                "rsi_14d",
                "volume_trend_5d",
            ]

        df = df[[c for c in keep if c in df.columns]]
        logger.info(
            "Feature groups %s applied — %d columns retained.",
            feature_groups,
            len(df.columns),
        )

    logger.info(
        "Dataset ready: %d rows, %d features.",
        len(df),
        len([c for c in df.columns if c not in ("stock_code", "trade_date")]),
    )
    return df


def _compute_traditional_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add 8 traditional price-volume factor columns to the modeling dataset.

    Operates on a DataFrame that already has stock_code, close, pre_close,
    volume, turnover, and ret_1d.  All rolling/computed values use per-stock
    grouping to avoid cross-stock leakage.

    Added columns
    -------------
    ret_5d, ret_20d, max_ret_5d, min_ret_5d, illiquidity_20d,
    skewness_20d, rsi_14d, volume_trend_5d.
    """
    # Ensure sorted per stock before window operations
    df = df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)

    # 1. ret_5d — 5-day cumulative return
    df["ret_5d"] = (
        df.groupby("stock_code")["close"]
        .transform(lambda s: s / s.shift(5) - 1)
    )

    # 2. ret_20d — 20-day cumulative return
    df["ret_20d"] = (
        df.groupby("stock_code")["close"]
        .transform(lambda s: s / s.shift(20) - 1)
    )

    # 3. max_ret_5d — max ret_1d over past 5 days
    df["max_ret_5d"] = (
        df.groupby("stock_code")["ret_1d"]
        .transform(lambda s: s.rolling(5, min_periods=3).max())
    )

    # 4. min_ret_5d — min ret_1d over past 5 days
    df["min_ret_5d"] = (
        df.groupby("stock_code")["ret_1d"]
        .transform(lambda s: s.rolling(5, min_periods=3).min())
    )

    # 5. illiquidity_20d — Amihud illiquidity
    #    daily_illiq = |ret_1d| / (volume * 100)  (volume is in 手)
    daily_illiq = df["ret_1d"].abs() / (df["volume"] * 100.0)
    daily_illiq = daily_illiq.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["illiquidity_20d"] = (
        daily_illiq.groupby(df["stock_code"])
        .transform(lambda s: s.rolling(20, min_periods=5).mean())
    )

    # 6. skewness_20d — rolling 20-day skewness of ret_1d
    df["skewness_20d"] = (
        df.groupby("stock_code")["ret_1d"]
        .rolling(20, min_periods=10)
        .skew()
        .reset_index(level=0, drop=True)
    )

    # 7. rsi_14d — RSI(14)
    delta = df.groupby("stock_code")["close"].transform(pd.Series.diff)
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = (
        gain.groupby(df["stock_code"])
        .transform(lambda s: s.rolling(14, min_periods=10).mean())
    )
    avg_loss = (
        loss.groupby(df["stock_code"])
        .transform(lambda s: s.rolling(14, min_periods=10).mean())
    )

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    df["rsi_14d"] = rsi.fillna(50.0)

    # 8. volume_trend_5d — 5d avg volume / 20d avg volume
    vol_ma5 = (
        df.groupby("stock_code")["volume"]
        .transform(lambda s: s.rolling(5, min_periods=1).mean())
    )
    vol_ma20 = (
        df.groupby("stock_code")["volume"]
        .transform(lambda s: s.rolling(20, min_periods=1).mean())
    )
    df["volume_trend_5d"] = np.where(
        vol_ma20 > 0,
        vol_ma5 / vol_ma20,
        1.0,
    )
    df["volume_trend_5d"] = np.where(
        np.isfinite(df["volume_trend_5d"]),
        df["volume_trend_5d"],
        1.0,
    )

    return df


def run_pipeline(
    compute_only: bool = False,
    build_dataset: bool = True,
    start_date: str = "2025-06-15",
    end_date: str = "2026-06-15",
) -> dict[str, Any]:
    """Run the full factor pipeline: compute → save → (optionally) build dataset.

    Parameters
    ----------
    compute_only : bool
        If ``True``, only compute and save factors — skip the modeling dataset.
    build_dataset : bool
        If ``True`` (default), also build and return the ML training dataset.
    start_date : str
        Start date for the modeling dataset (``YYYY-MM-DD``).
    end_date : str
        End date for the modeling dataset (``YYYY-MM-DD``).

    Returns
    -------
    dict
        Summary statistics including row counts and (optionally) the dataset
        shape.
    """
    logger.info("=" * 60)
    logger.info("FACTOR PIPELINE STARTED")
    logger.info("=" * 60)

    # Step 1: Compute
    factors = compute_daily_factors()
    logger.info("Computed %d factor rows.", len(factors))

    # Step 2: Save
    written = save_factors(factors)
    logger.info("Written %d rows to daily_factors table.", written)

    stats: dict[str, Any] = {
        "factors_computed": len(factors),
        "factors_written": written,
    }

    # Step 3: Build dataset
    if build_dataset and not compute_only:
        dataset = build_model_dataset(start_date=start_date, end_date=end_date)
        stats["dataset_shape"] = dataset.shape
        logger.info("Dataset shape: %s", dataset.shape)
        if not dataset.empty:
            stats["dataset_columns"] = list(dataset.columns)
            stats["dataset_target_mean"] = float(dataset["fwd_ret_1d"].mean())
            stats["dataset_target_std"] = float(dataset["fwd_ret_1d"].std())

    logger.info("=" * 60)
    logger.info("FACTOR PIPELINE COMPLETE")
    logger.info("=" * 60)

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    stats = run_pipeline()
    print("\nPipeline summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
