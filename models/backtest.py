"""
Backtesting module for retail-sentiment-alpha.

Portfolio construction by predicted-return deciles, cumulative-return
comparison, and long-short evaluation.

Usage
-----
::

    uv run python -m models.backtest
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Generate predictions
# ---------------------------------------------------------------------------


def generate_predictions(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    stock_codes: pd.Series,
    test_dates: pd.Series,
) -> pd.DataFrame:
    """Generate out-of-sample predictions from a trained model.

    Parameters
    ----------
    model : estimator
        Fitted model with a ``predict`` method.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True test returns (``fwd_ret_1d``).
    stock_codes : pd.Series
        Stock-code label for each row of *X_test*.
    test_dates : pd.Series
        Trade date for each row of *X_test*.

    Returns
    -------
    pd.DataFrame
        Columns: ``stock_code``, ``trade_date``, ``pred_ret``, ``actual_ret``.
    """
    y_pred = model.predict(X_test)

    result = pd.DataFrame(
        {
            "stock_code": np.asarray(stock_codes),
            "trade_date": np.asarray(test_dates),
            "pred_ret": y_pred,
            "actual_ret": np.asarray(y_test),
        }
    )

    logger.info(
        "Generated %d predictions (date range: %s → %s).",
        len(result),
        result["trade_date"].min(),
        result["trade_date"].max(),
    )
    return result


# ---------------------------------------------------------------------------
# 2. Form decile portfolios
# ---------------------------------------------------------------------------


def form_portfolios(
    pred_df: pd.DataFrame,
    n_groups: int = 10,
) -> pd.DataFrame:
    """Sort stocks by predicted return and assign decile groups.

    For each trade date, stocks are ranked by ``pred_ret`` descending
    and assigned to *n_groups* equal-weighted portfolios via
    ``pd.qcut``.  Group **1** = highest predicted returns;
    group **n_groups** = lowest predicted returns.

    When there are fewer stocks than *n_groups* on a given date, the
    number of groups is reduced to the number of available stocks
    (minimum 3).  Dates with fewer than 3 stocks are skipped with a
    warning.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Must contain columns ``trade_date`` and ``pred_ret``.
    n_groups : int
        Target number of portfolios (default 10).

    Returns
    -------
    pd.DataFrame
        Input *pred_df* with an added ``group`` column (``int``,
        1 … *n_groups*).

    Raises
    ------
    ValueError
        If *pred_df* is empty.
    """
    if pred_df.empty:
        raise ValueError("pred_df is empty — cannot form portfolios.")

    logger.info("Forming %d portfolios from %d predictions …", n_groups, len(pred_df))

    # Iterate over each date manually so we always have the date value
    # (groupby().apply() drops the group-key column from the passed frame).
    assigned_dfs: list[pd.DataFrame] = []
    for date, grp in pred_df.groupby("trade_date"):
        n = len(grp)
        effective = min(n_groups, n)
        if effective < 3:
            logger.warning(
                "Only %d stock(s) on %s — minimum 3 groups required.  Skipping.",
                n,
                date,
            )
            continue

        # rank(method="first") breaks ties so qcut never fails on duplicates
        labels: np.ndarray = pd.qcut(
            grp["pred_ret"].rank(method="first"),
            q=effective,
            labels=False,
        )
        # Group 1 = highest predicted return
        grp = grp.copy()
        grp["group"] = (effective - labels).astype(int)
        logger.debug("  %s → %d group(s) from %d stock(s)", date, effective, n)
        assigned_dfs.append(grp)

    if not assigned_dfs:
        logger.warning("No portfolios could be formed — check data availability.")
        return pd.DataFrame()

    result = pd.concat(assigned_dfs, ignore_index=True)
    result["group"] = result["group"].astype(int)

    # ── Log group-size distribution (first few dates as sample) ──────────
    sizes = result.groupby(["trade_date", "group"]).size()
    logger.info(
        "Portfolios formed: %d unique dates, %d total assignments.",
        result["trade_date"].nunique(),
        len(result),
    )
    # Show a compact per-date group-size sample (first 3 dates)
    sample_dates = result["trade_date"].unique()[:3]
    for d in sample_dates:
        chunk = sizes.loc[d]
        logger.info("  %s  group sizes: %s", d, dict(chunk))

    return result


# ---------------------------------------------------------------------------
# 3. Compute portfolio returns
# ---------------------------------------------------------------------------


def compute_portfolio_returns(port_df: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted mean return per group per date + long-short spread.

    The long-short return is the top-decile return minus the bottom-decile
    return for each date.

    Parameters
    ----------
    port_df : pd.DataFrame
        Output of :func:`form_portfolios` (must contain ``trade_date``,
        ``group``, and ``actual_ret`` columns).

    Returns
    -------
    pd.DataFrame
        Columns: ``trade_date``, ``group`` (``str`` — decile label or
        ``"long_short"``), ``portfolio_ret``.
    """
    # Equal-weighted mean per (trade_date, group)
    port_ret: pd.DataFrame = (
        port_df.groupby(["trade_date", "group"], as_index=False)["actual_ret"]
        .mean()
        .rename(columns={"actual_ret": "portfolio_ret"})
    )
    port_ret["group"] = port_ret["group"].astype(str)

    # Long-short: top group − bottom group for each date
    ls_records: list[dict[str, Any]] = []
    for date, grp in port_ret.groupby("trade_date"):
        top = grp.loc[grp["group"] == "1", "portfolio_ret"]
        if top.empty:
            continue
        max_group = str(grp["group"].astype(int).max())
        bottom = grp.loc[grp["group"] == max_group, "portfolio_ret"]
        if bottom.empty:
            continue
        ls_records.append(
            {
                "trade_date": date,
                "group": "long_short",
                "portfolio_ret": top.iloc[0] - bottom.iloc[0],
            }
        )

    ls_df = (
        pd.DataFrame(ls_records)
        if ls_records
        else pd.DataFrame(columns=["trade_date", "group", "portfolio_ret"])
    )

    result = (
        pd.concat([port_ret, ls_df], ignore_index=True)
        .sort_values(["group", "trade_date"])
        .reset_index(drop=True)
    )

    logger.info(
        "Computed portfolio returns: %d group-date observations (incl. long-short).",
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# 4. Cumulative returns
# ---------------------------------------------------------------------------


def compute_cumulative_returns(port_ret_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative returns for each portfolio group.

    Uses ``(1 + ret).cumprod() - 1`` per group.

    Parameters
    ----------
    port_ret_df : pd.DataFrame
        Output of :func:`compute_portfolio_returns`.

    Returns
    -------
    pd.DataFrame
        Input *port_ret_df* with an added ``cum_ret`` column.
    """
    if port_ret_df.empty:
        logger.warning("Empty portfolio returns — returning empty DataFrame.")
        result = port_ret_df.copy()
        result["cum_ret"] = np.nan
        return result

    # Check for all-NaN returns
    if port_ret_df["portfolio_ret"].isna().all():
        logger.warning(
            "All portfolio returns are NaN — cumulative computation skipped."
        )
        result = port_ret_df.copy()
        result["cum_ret"] = np.nan
        return result

    result = port_ret_df.sort_values(["group", "trade_date"]).copy()

    result["cum_ret"] = result.groupby("group")["portfolio_ret"].transform(
        lambda x: (1 + x).cumprod() - 1,
    )

    n_groups = result["group"].nunique()
    logger.info("Computed cumulative returns for %d group(s).", n_groups)
    return result


# ---------------------------------------------------------------------------
# 5. Performance metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    port_ret_df: pd.DataFrame,
    annual_factor: int = 252,
) -> dict[str, dict[str, float]]:
    """Compute annualised return, volatility, Sharpe, max drawdown, win rate.

    Metrics are calculated for each portfolio group (including
    ``"long_short"``).

    Parameters
    ----------
    port_ret_df : pd.DataFrame
        Output of :func:`compute_portfolio_returns` (must contain
        ``group`` and ``portfolio_ret`` columns).
    annual_factor : int
        Number of trading periods per year (default 252 for daily data).

    Returns
    -------
    dict
        ``{group_name: {annualized_return, annualized_volatility,
        sharpe_ratio, max_drawdown, win_rate}}``.
    """
    metrics: dict[str, dict[str, float]] = {}

    for group_name, grp in port_ret_df.groupby("group"):
        rets = grp["portfolio_ret"].dropna()
        if rets.empty:
            logger.warning("No non-NaN returns for group '%s'.", group_name)
            continue

        ann_ret = float(rets.mean() * annual_factor)
        ann_vol = float(rets.std() * np.sqrt(annual_factor))
        sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0

        # Maximum drawdown
        cum = (1 + rets).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        max_dd = float(dd.min())

        win_rate = float((rets > 0).mean())

        metrics[str(group_name)] = {
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
        }

    return metrics


# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------


def plot_backtest(
    cum_ret_df: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
    output_dir: str = "outputs",
) -> None:
    """Generate and save backtest diagnostic plots.

    Plots
    -----
    * ``backtest_cumulative.png`` — cumulative returns for deciles 1, 5,
      10 and the long-short spread.
    * ``backtest_decile_returns.png`` — bar chart of annualised returns
      per decile.
    * ``backtest_metrics_table.png`` — formatted table of key metrics.

    Parameters
    ----------
    cum_ret_df : pd.DataFrame
        Output of :func:`compute_cumulative_returns`.
    metrics : dict
        Output of :func:`compute_metrics`.
    output_dir : str
        Directory to save images (created if missing).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _plot_cumulative(cum_ret_df, out)
    _plot_decile_returns(metrics, out)
    _plot_metrics_table(metrics, out)


def _plot_cumulative(cum_ret_df: pd.DataFrame, out: Path) -> None:
    """Cumulative-returns line plot (deciles 1, 5, 10, and long-short)."""
    if cum_ret_df.empty or cum_ret_df["cum_ret"].isna().all():
        logger.warning("No valid cumulative returns — skipping cumulative plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5.5))

    available = set(cum_ret_df["group"].unique())
    highlight = {"1", "5", "10"} & available
    others = {g for g in available - {"long_short"} if g.isdigit()} - highlight

    # Faint background lines for all other deciles
    for grp in sorted(others, key=int):
        data = cum_ret_df[cum_ret_df["group"] == grp].sort_values("trade_date")
        ax.plot(
            data["trade_date"],
            data["cum_ret"] * 100,
            color="gray",
            alpha=0.25,
            linewidth=0.6,
        )

    # Highlighted deciles
    palette = {"1": "#2ecc71", "5": "#3498db", "10": "#e74c3c"}
    for grp in sorted(highlight, key=int):
        data = cum_ret_df[cum_ret_df["group"] == grp].sort_values("trade_date")
        ax.plot(
            data["trade_date"],
            data["cum_ret"] * 100,
            color=palette.get(grp, "#333"),
            linewidth=1.6,
            label=f"Decile {grp}",
        )

    # Long-short (dashed, thicker)
    ls = cum_ret_df[cum_ret_df["group"] == "long_short"].sort_values("trade_date")
    if not ls.empty:
        ax.plot(
            ls["trade_date"],
            ls["cum_ret"] * 100,
            color="#9b59b6",
            linewidth=2.8,
            linestyle="--",
            label="Long–Short",
        )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Trade Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Backtest — Cumulative Returns by Decile Portfolio")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "backtest_cumulative.png", dpi=150)
    plt.close(fig)
    logger.info("Saved backtest_cumulative.png")


def _plot_decile_returns(
    metrics: dict[str, dict[str, float]],
    out: Path,
) -> None:
    """Bar chart of annualised return per decile."""
    numeric = {k: v for k, v in metrics.items() if k.isdigit()}
    if not numeric:
        logger.warning("No numeric-group metrics — skipping decile-returns plot.")
        return

    groups = sorted(numeric, key=int)
    ann_rets = [numeric[g]["annualized_return"] * 100 for g in groups]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in ann_rets]
    bars = ax.bar(groups, ann_rets, color=colors, width=0.7, edgecolor="white")

    for bar, val in zip(bars, ann_rets):
        y_pos = bar.get_height() + (0.4 if val >= 0 else -2.0)
        va: str = "bottom" if val >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{val:.2f}%",
            ha="center",
            va=va,
            fontsize=8,
        )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Decile Group")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("Annualized Return by Decile Portfolio")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "backtest_decile_returns.png", dpi=150)
    plt.close(fig)
    logger.info("Saved backtest_decile_returns.png")


def _plot_metrics_table(
    metrics: dict[str, dict[str, float]],
    out: Path,
) -> None:
    """Formatted table of performance metrics."""
    if not metrics:
        logger.warning("No metrics — skipping metrics table.")
        return

    # Sort: numeric groups first (by integer key), then long_short
    sorted_keys = sorted(
        [k for k in metrics if k != "long_short"],
        key=lambda x: int(x),
    )
    if "long_short" in metrics:
        sorted_keys.append("long_short")

    rows = []
    for key in sorted_keys:
        m = metrics[key]
        rows.append(
            [
                key,
                f"{m['annualized_return'] * 100:.2f}%",
                f"{m['annualized_volatility'] * 100:.2f}%",
                f"{m['sharpe_ratio']:.2f}",
                f"{m['max_drawdown'] * 100:.2f}%",
                f"{m['win_rate'] * 100:.1f}%",
            ]
        )

    col_labels = [
        "Group",
        "Ann. Return",
        "Ann. Vol",
        "Sharpe",
        "Max DD",
        "Win Rate",
    ]

    fig_height = 0.5 * len(rows) + 2.5
    fig, ax = plt.subplots(figsize=(10, max(3.0, fig_height)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Header styling
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", weight="bold")

    # Highlight the long_short row
    for i, row in enumerate(rows):
        if row[0] == "long_short":
            for j in range(len(row)):
                table[i + 1, j].set_facecolor("#f3e8ff")

    ax.set_title("Backtest Performance Metrics", fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(out / "backtest_metrics_table.png", dpi=150)
    plt.close(fig)
    logger.info("Saved backtest_metrics_table.png")


# ---------------------------------------------------------------------------
# 7. Orchestration
# ---------------------------------------------------------------------------


def run_backtest(
    models_dict: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    stock_codes: pd.Series,
    test_dates: pd.Series,
    output_dir: str = "outputs",
) -> dict[str, Any]:
    """Run the full backtest workflow for all models in *models_dict*.

    For each model (Ridge, LASSO, LightGBM):

    1. Generate out-of-sample predictions
    2. Form decile portfolios
    3. Compute portfolio + cumulative returns
    4. Compute performance metrics
    5. Generate diagnostic plots

    Parameters
    ----------
    models_dict : dict
        Output of ``models.regression.run_modeling_pipeline()``.  Must
        contain keys ``"ridge"``, ``"lasso"``, and ``"lightgbm"``, each
        with a ``"model"`` entry.
    X_test : pd.DataFrame
        Test feature matrix (aligned with *stock_codes* and *test_dates*).
    y_test : pd.Series
        True test returns.
    stock_codes : pd.Series
        Stock code for each test row.
    test_dates : pd.Series
        Trade date for each test row.
    output_dir : str
        Directory for diagnostic plots (default ``"outputs"``).

    Returns
    -------
    dict
        ``{model_name: {predictions, portfolios, returns,
        cumulative_returns, metrics}}``.

    Raises
    ------
    ValueError
        If no predictions can be generated for any model.
    """
    logger.info("=" * 60)
    logger.info("BACKTEST STARTED")
    logger.info("=" * 60)

    model_keys = [
        ("Ridge", "ridge"),
        ("LASSO", "lasso"),
        ("LightGBM", "lightgbm"),
    ]

    results: dict[str, Any] = {}

    for display_name, key in model_keys:
        if key not in models_dict:
            logger.warning("'%s' not found in models_dict — skipping.", key)
            continue

        model_entry = models_dict[key]
        model = model_entry.get("model")
        if model is None:
            logger.warning("No model object for '%s' — skipping.", display_name)
            continue

        logger.info("-" * 40)
        logger.info("Backtesting %s …", display_name)

        # 1. Generate predictions
        pred_df = generate_predictions(
            model, X_test, y_test, stock_codes, test_dates
        )
        if pred_df.empty:
            raise ValueError(f"No predictions generated for {display_name}.")

        # 2. Form portfolios
        port_df = form_portfolios(pred_df)
        if port_df.empty:
            logger.warning("No portfolios formed for %s.", display_name)
            continue

        # 3. Compute portfolio returns
        port_ret_df = compute_portfolio_returns(port_df)
        if port_ret_df.empty:
            logger.warning("No portfolio returns for %s.", display_name)
            continue

        # 4. Compute cumulative returns
        cum_ret_df = compute_cumulative_returns(port_ret_df)

        # 5. Compute metrics
        metrics = compute_metrics(port_ret_df)

        # 6. Generate plots
        plot_backtest(cum_ret_df, metrics, output_dir=output_dir)

        model_result = {
            "predictions": pred_df,
            "portfolios": port_df,
            "returns": port_ret_df,
            "cumulative_returns": cum_ret_df,
            "metrics": metrics,
        }
        results[display_name] = model_result

        ls_metrics = metrics.get("long_short", {})
        ls_sharpe = ls_metrics.get("sharpe_ratio", float("nan"))
        ls_ann_ret = ls_metrics.get("annualized_return", 0.0)
        logger.info(
            "%s — Long–Short Sharpe: %.4f, Ann. Return: %.2f%%",
            display_name,
            ls_sharpe,
            ls_ann_ret * 100,
        )

    logger.info("=" * 60)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 60)

    if not results:
        raise ValueError("No backtest results produced — check model availability.")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=" * 60)
    logger.info("BACKTEST ENTRY POINT — loading dataset …")
    logger.info("=" * 60)

    # ── 1. Load dataset (with stock-code metadata) ──────────────────────
    from features.factors import build_model_dataset

    dataset = build_model_dataset()
    if dataset.empty:
        logger.error(
            "Empty dataset — cannot run backtest.\n\n"
            "Run the factor pipeline first:\n\n"
            "    uv run python -m features.factors"
        )
        raise SystemExit(1)

    stock_codes_orig = dataset["stock_code"]
    dates_orig = dataset["trade_date"]
    y_orig = dataset["fwd_ret_1d"]
    X_orig = dataset.drop(columns=["stock_code", "trade_date", "fwd_ret_1d"])

    # Apply same NaN filtering as ``regression.load_data``
    nan_mask = X_orig.isna().any(axis=1) | y_orig.isna()
    n_nan = nan_mask.sum()
    if n_nan:
        logger.warning("Dropping %d row(s) with NaN values.", n_nan)
        X_orig = X_orig.loc[~nan_mask]
        y_orig = y_orig.loc[~nan_mask]
        stock_codes_orig = stock_codes_orig.loc[~nan_mask]
        dates_orig = dates_orig.loc[~nan_mask]

    logger.info(
        "Dataset ready: %d rows, %d feature(s), %d stock(s), date range %s → %s.",
        len(X_orig),
        X_orig.shape[1],
        stock_codes_orig.nunique(),
        dates_orig.min(),
        dates_orig.max(),
    )

    # ── 2. Chronological train / test split ────────────────────────────
    from models.regression import train_test_split_by_time

    X_train, X_test, y_train, y_test, train_idx, test_idx, scaler = (
        train_test_split_by_time(X_orig, y_orig, dates_orig)
    )

    # Subset stock-code / date metadata to test rows
    test_stock_codes = stock_codes_orig.iloc[test_idx]
    test_dates = dates_orig.iloc[test_idx]

    # ── 3. Train models ──────────────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Training models …")

    from models.regression import train_lasso, train_lightgbm, train_ridge

    ridge_result = train_ridge(X_train, y_train)
    lasso_result = train_lasso(X_train, y_train)
    lgbm_result = train_lightgbm(X_train, y_train, X_test, y_test)

    models_dict = {
        "ridge": ridge_result,
        "lasso": lasso_result,
        "lightgbm": lgbm_result,
    }

    # ── 4. Run backtest ──────────────────────────────────────────────────
    backtest_results = run_backtest(
        models_dict=models_dict,
        X_test=X_test,
        y_test=y_test,
        stock_codes=test_stock_codes,
        test_dates=test_dates,
        output_dir="outputs",
    )

    # ── 5. Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    for model_name in ["Ridge", "LASSO", "LightGBM"]:
        if model_name in backtest_results:
            m = backtest_results[model_name]["metrics"]
            ls = m.get("long_short", {})
            ls_sharpe = ls.get("sharpe_ratio", float("nan"))
            ls_ann = ls.get("annualized_return", 0.0) * 100
            ls_vol = ls.get("annualized_volatility", 0.0) * 100
            ls_wr = ls.get("win_rate", 0.0) * 100
            print(
                f"  {model_name:10s}  "
                f"LS Sharpe: {ls_sharpe:.4f}  "
                f"Return: {ls_ann:+.2f}%  "
                f"Vol: {ls_vol:.2f}%  "
                f"Win: {ls_wr:.1f}%"
            )
    print("=" * 60)
    print("Plots saved to: outputs/")
    print("Done.")
