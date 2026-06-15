# AGENTS.md — retail-sentiment-alpha

散户情绪另类因子 Alpha 策略，期末大作业。爬取股吧帖子 → FinBERT 情绪打分 → 因子计算 → ML 建模 → 回测。

## Quick commands

```bash
# Run everything with uv (only package manager used)
uv run python -m <module>
uv run python run_pipeline.py          # full NLP → factors → ablation → LSTM → backtest
uv run python web/app.py               # dashboard at http://localhost:8000

# Database
docker compose up -d                   # PostgreSQL 16 (auto-creates schema from sql/schema.sql)
```

## First-time setup

```bash
uv sync                                # install dependencies (Python 3.11+)
docker compose up -d                   # start PostgreSQL (schema auto-created)
# Ensure .env exists with:
#   DATABASE_URL=postgresql://alpha:alpha123@localhost:5432/sentiment_alpha
#   TGB_COOKIE=<淘股吧 cookie>    (optional, only needed for taoguba crawlers)
```

## Execution order

Steps 1–2 are prerequisites — run them before `run_pipeline.py` (which covers steps 3–6):

1. **Crawlers** → populates `posts` table
2. **Market data** → populates `market_daily` / `index_daily` tables
3. **NLP** → scores `posts.sentiment` via FinBERT  ← `run_pipeline.py` starts here
4. **Factors** → aggregates into `daily_factors`, joins with market data → modeling dataset
5. **Models** (regression / LSTM) → trains on factor + market dataset
6. **Backtest** → portfolio simulation on model predictions

## Module entrypoints

| Module | Command | Notes |
|---|---|---|
| `crawlers.eastmoney` | `uv run python -m crawlers.eastmoney` | 东方财富股吧，**no cookie needed**，listing-only by default (fast). Crawls CSI 300 stocks. |
| `crawlers.taoguba` | `uv run python -m crawlers.taoguba` | 淘股吧 stock search. **Needs `TGB_COOKIE` in `.env`.** |
| `crawlers.taoguba_bbs` | `uv run python -m crawlers.taoguba_bbs` | 淘股吧 BBS latest-posts crawl. **Needs `TGB_COOKIE` in `.env`.** |
| `data.market_data` | `uv run python -m data.market_data` | Downloads ~1 year CSI 300 OHLCV via baostock. |
| `nlp.sentiment` | `uv run python -m nlp.sentiment` | FinBERT-Chinese scoring. **First run downloads ~410MB model.** |
| `features.factors` | `uv run python -m features.factors` | Compute daily sentiment factors from scored posts. |
| `models.regression` | `uv run python -m models.regression` | Ridge / LASSO / LightGBM + ablation. |
| `models.lstm` | `uv run python -m models.lstm` | PyTorch LSTM (auto-detects MPS/CUDA/CPU). |
| `models.backtest` | `uv run python -m models.backtest` | Decile portfolio + long-short evaluation. |

## Gotchas

- **`crawlers.eastmoney` imports `get_stock_list` from `crawlers.taoguba`.** Despite the name, the eastmoney crawler depends on the taoguba module being importable. This works because both are in the same package.
- **Dead directories**: `factors/`, `backtest/`, and `market_data/` are empty. Actual code: factor computation in `features/factors.py`, backtest in `models/backtest.py`, market data fetcher in `data/market_data.py`.
- **`.env` is gitignored** — must contain `DATABASE_URL` (postgresql://alpha:alpha123@localhost:5432/sentiment_alpha) and optionally `TGB_COOKIE`.
- **FinBERT model** (`yiyanghkust/finbert-tone-chinese`) downloads ~410MB on first NLP run. Ensure enough disk space.
- **NLP device**: defaults to `cpu`. Supports `"cpu"`, `"cuda"`, and `"mps"` — falls back to CPU if the requested device is unavailable. LSTM auto-detects MPS/CUDA/CPU correctly. The `run_pipeline.py` always runs NLP on CPU — to use GPU/MPS, call `nlp.sentiment` directly with `device="cuda"` or `device="mps"`.
- **baostock rate limiting**: `data/market_data.py` uses 0.2s intervals. Server IPs may still be throttled.
- **Crawler rate limits**: eastmoney has 5s between stocks, taoguba has 1.5–3s between requests. Don't reduce these — IP blocks are real.
- **Posts dedup**: `ON CONFLICT (url) DO NOTHING` in `posts` table. Re-running crawlers is safe — no duplicates.
- **eastmoney detail fetch is broken**: `fetch_details=True` produces empty content due to JS-rendered pages and encoding issues. Always use default `listing-only` mode — titles alone are sufficient for sentiment scoring.

## Database

- PostgreSQL 16 via Docker. Credentials: `alpha` / `alpha123` / `sentiment_alpha`.
- Schema: `sql/schema.sql` (auto-applied on first `docker compose up`).
- Tables: `posts`, `daily_factors`, `market_daily`, `index_daily`.
- Multiple modules create their own engine (some use `crawlers.config.engine`, others call `create_engine` directly from env).
