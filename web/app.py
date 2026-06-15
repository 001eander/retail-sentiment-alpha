"""Dashboard web app — FastAPI + Jinja2 + Chart.js."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy import text

from crawlers.config import engine

app = FastAPI(title="零售情绪 Alpha Dashboard")
_TPL = Path(__file__).parent / "templates" / "dashboard.html"


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(_TPL)


@app.get("/api/dashboard")
async def api_dashboard():
    """Return all dashboard data as JSON (shape matches frontend expectations)."""
    with engine.connect() as conn:
        # ── summary stats ──
        total = conn.execute(text("SELECT count(*) FROM posts")).scalar() or 0
        scored = conn.execute(
            text("SELECT count(*) FROM posts WHERE sentiment IS NOT NULL")
        ).scalar() or 0
        today = conn.execute(
            text("SELECT count(*) FROM posts WHERE fetched_at::date = CURRENT_DATE")
        ).scalar() or 0
        sent_avg = conn.execute(
            text("SELECT avg(sentiment) FROM posts WHERE sentiment IS NOT NULL")
        ).scalar() or 0

        # ── platform breakdown ──
        plat_rows = conn.execute(
            text("SELECT platform, count(*) FROM posts GROUP BY platform")
        ).fetchall()
        plat_map = {r[0]: r[1] for r in plat_rows}
        platforms = {
            "eastmoney": plat_map.get("eastmoney", 0),
            "taoguba": plat_map.get("taoguba", 0),
            "others": sum(v for k, v in plat_map.items() if k not in ("eastmoney", "taoguba")),
        }

        # ── recent posts ──
        recent_rows = conn.execute(
            text(
                "SELECT stock_code, stock_name, title, sentiment, post_time, platform "
                "FROM posts ORDER BY post_time DESC LIMIT 20"
            )
        ).fetchall()
        recent_posts = [
            {
                "stock_code": r[0],
                "stock_name": r[1],
                "title": r[2],
                "sentiment": float(r[3]) if r[3] is not None else None,
                "post_time": r[4].isoformat() if r[4] else None,
                "platform": r[5],
            }
            for r in recent_rows
        ]

        # ── sentiment distribution ──
        dist_rows = conn.execute(
            text(
                "SELECT "
                "  CASE WHEN sentiment > 0.2 THEN 'positive' "
                "       WHEN sentiment < -0.2 THEN 'negative' "
                "       ELSE 'neutral' END as label, "
                "  count(*) as cnt "
                "FROM posts WHERE sentiment IS NOT NULL "
                "GROUP BY label"
            )
        ).fetchall()
        dist_map = {r[0]: r[1] for r in dist_rows}
        sentiment_dist = {
            "positive": dist_map.get("positive", 0),
            "neutral": dist_map.get("neutral", 0),
            "negative": dist_map.get("negative", 0),
        }

        # ── top active stocks (with avg sentiment) ──
        top_rows = conn.execute(
            text(
                "SELECT stock_code, stock_name, count(*) as cnt, "
                "  COALESCE(avg(sentiment), 0) as avg_s "
                "FROM posts GROUP BY stock_code, stock_name "
                "ORDER BY cnt DESC LIMIT 10"
            )
        ).fetchall()
        top_stocks = [
            {
                "code": r[0],
                "name": r[1],
                "count": r[2],
                "avg_sentiment": round(float(r[3]), 2),
            }
            for r in top_rows
        ]

    return {
        "stats": {
            "total": total,
            "today": today,
            "scored": scored,
            "sentiment_avg": round(float(sent_avg), 2),
        },
        "platforms": platforms,
        "recent_posts": recent_posts,
        "sentiment_dist": sentiment_dist,
        "top_stocks": top_stocks,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
