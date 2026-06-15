"""Generate 4 publication-quality PNG figures for the retail-sentiment report.

Usage: python report/visualizations.py
"""

from __future__ import annotations
import logging, os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from scipy.stats import skew as compute_skew
from sqlalchemy import create_engine, text

load_dotenv()
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://alpha:alpha123@localhost:5432/sentiment_alpha")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})
PALETTE = sns.color_palette("Set2")


def _save_fig(name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / name)
    plt.close()
    logger.info("Saved figure: %s", name)


def _no_data_figure(name: str) -> None:
    logger.warning("No data for %s — saving placeholder.", name)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, "No Data", ha="center", va="center",
            fontsize=18, color="gray", transform=ax.transAxes)
    ax.set_title("No Data Available")
    _save_fig(name)


def plot_sentiment_distribution() -> str | None:
    name = "sentiment_distribution.png"
    sql = text("SELECT sentiment FROM posts WHERE sentiment IS NOT NULL")
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    if not rows:
        _no_data_figure(name); return None

    scores = np.array([r[0] for r in rows], dtype=np.float64)
    n = len(scores)
    mean, std, sk = float(np.mean(scores)), float(np.std(scores, ddof=1)), float(compute_skew(scores, bias=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=30, density=True, alpha=0.6, color=PALETTE[0], label="Frequency")
    try:
        sns.kdeplot(x=scores, ax=ax, color=PALETTE[1], linewidth=2, label="KDE")
    except Exception:
        logger.warning("KDE estimation failed.")
    ax.axvline(mean, color=PALETTE[2], linestyle="--", linewidth=1.5, label=f"Mean = {mean:.3f}")
    ax.text(0.97, 0.97, f"N = {n:,}\nMean = {mean:.4f}\nStd = {std:.4f}\nSkew = {sk:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.85))
    ax.set_xlabel("Sentiment Score [-1, 1]")
    ax.set_ylabel("Frequency")
    ax.set_title("Retail Sentiment Score Distribution")
    ax.legend(fontsize=9)
    _save_fig(name); return name


def plot_factor_correlation() -> str | None:
    name = "factor_correlation.png"
    factor_cols = ["post_volume_anomaly", "sentiment_score", "sentiment_divergence", "interaction_intensity"]
    sql = text(f"SELECT {', '.join(factor_cols)} FROM daily_factors")
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    if not rows:
        _no_data_figure(name); return None

    df = pd.DataFrame(rows, columns=factor_cols).dropna(how="all")
    if df.empty:
        _no_data_figure(name); return None

    corr = df.corr(method="pearson")
    cols_list = list(corr.columns)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                xticklabels=cols_list, yticklabels=cols_list, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Correlation"})
    ax.set_title("Factor Correlation Matrix")
    plt.yticks(rotation=0)
    _save_fig(name); return name


def plot_posts_over_time() -> str | None:
    name = "posts_over_time.png"
    sql = text("SELECT DATE(post_time) AS dt, COUNT(*) AS cnt FROM posts "
               "WHERE post_time IS NOT NULL GROUP BY dt ORDER BY dt")
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    if not rows:
        _no_data_figure(name); return None

    df = pd.DataFrame(rows, columns=["dt", "cnt"])
    df["dt"] = pd.to_datetime(df["dt"])
    df["rolling_7d"] = df["cnt"].rolling(window=7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["dt"], df["cnt"], width=0.8, color=PALETTE[0], alpha=0.6, label="Daily Posts")
    ax.plot(df["dt"], df["rolling_7d"], color=PALETTE[1], linewidth=2, label="7-Day Rolling Avg")
    ax.set_xlabel("Date"); ax.set_ylabel("Post Count")
    ax.set_title("Daily Post Volume"); ax.legend(fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    _save_fig(name); return name


def plot_platform_comparison() -> str | None:
    name = "platform_comparison.png"
    sql = text("SELECT platform, COUNT(*) AS cnt FROM posts GROUP BY platform ORDER BY cnt DESC")
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    if not rows:
        _no_data_figure(name); return None

    platforms = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(platforms, counts, color=PALETTE[0], alpha=0.7)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Platform"); ax.set_ylabel("Post Count")
    ax.set_title("Post Count by Platform")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    _save_fig(name); return name


def generate_all() -> dict[str, str | None]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Saving figures to: %s", OUTPUT_DIR.resolve())
    makers = [
        ("sentiment_distribution.png", plot_sentiment_distribution),
        ("factor_correlation.png", plot_factor_correlation),
        ("posts_over_time.png", plot_posts_over_time),
        ("platform_comparison.png", plot_platform_comparison),
    ]
    results: dict[str, str | None] = {}
    for i, (name, func) in enumerate(makers, 1):
        logger.info("─" * 48)
        logger.info("Figure %d/4 — %s", i, name)
        results[name] = func()

    ok = sum(1 for v in results.values() if v is not None)
    logger.info("─" * 48)
    logger.info("Done — %d/%d generated.", ok, len(makers))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    generate_all()
