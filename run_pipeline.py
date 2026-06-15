"""Master pipeline: NLP → factors → ablation → LSTM → backtest."""

from __future__ import annotations

import logging
import time
from typing import Any

from sqlalchemy.exc import OperationalError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_full_pipeline(
    nlp_limit: int | None = None,
    skip_nlp: bool = False,
    skip_lstm: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline: NLP → factors → ablation → LSTM → backtest."""
    summary: dict[str, Any] = {}
    abort = False

    # ── Step 1 — NLP ─────────────────────────────────────────────────────
    if not skip_nlp:
        logger.info("─" * 48)
        logger.info("STEP 1/5  NLP Sentiment Scoring")
        t0 = time.time()
        try:
            from nlp.sentiment import run_sentiment_pipeline
            n = run_sentiment_pipeline(limit=nlp_limit)
            summary["nlp"] = {"scored": n, "elapsed_s": round(time.time() - t0, 1)}
            logger.info("NLP: %d posts scored (%.1fs)", n, summary["nlp"]["elapsed_s"])
        except OperationalError:
            logger.exception("DB connection error — aborting")
            summary["nlp"] = {"error": "DB error"}
            abort = True
        except Exception as exc:
            logger.exception("NLP failed: %s", exc)
            summary["nlp"] = {"error": str(exc)}
    else:
        summary["nlp"] = {"skipped": True}

    # ── Step 2 — Factors ──────────────────────────────────────────────────
    if not abort:
        logger.info("─" * 48)
        logger.info("STEP 2/5  Factor Computation")
        t0 = time.time()
        try:
            from features.factors import run_pipeline as run_factor_pipeline
            fs = run_factor_pipeline()
            summary["factors"] = {**fs, "elapsed_s": round(time.time() - t0, 1)}
            logger.info("Factors: %d rows, %.1fs", fs.get("factors_written", 0),
                        summary["factors"]["elapsed_s"])
        except OperationalError:
            logger.exception("DB connection error — aborting")
            summary["factors"] = {"error": "DB error"}
            abort = True
        except Exception as exc:
            logger.exception("Factors failed: %s", exc)
            summary["factors"] = {"error": str(exc)}

    # ── Step 3 — Ablation Experiments ─────────────────────────────────────
    ablation_results: dict[str, Any] | None = None
    if not abort:
        logger.info("─" * 48)
        logger.info("STEP 3/5  Ablation Experiments (3 × 3 models)")
        t0 = time.time()
        try:
            from models.regression import run_ablation_experiments
            ablation_results = run_ablation_experiments()
            cmp = ablation_results.get("comparison_table")
            if cmp is not None and not cmp.empty:
                best = cmp.iloc[0]
                summary["ablation"] = {
                    "configs": len(ablation_results) - 1,  # minus comparison_table
                    "best_combo": f"{best.name[0]} + {best.name[1]}",
                    "best_r2": round(float(best["R²"]), 6),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                logger.info("Ablation: best = %s (R²=%.4f), %.1fs",
                            summary["ablation"]["best_combo"],
                            summary["ablation"]["best_r2"],
                            summary["ablation"]["elapsed_s"])
        except OperationalError:
            logger.exception("DB error")
            summary["ablation"] = {"error": "DB error"}
        except Exception as exc:
            logger.exception("Ablation failed: %s", exc)
            summary["ablation"] = {"error": str(exc)}

    # ── Step 4 — LSTM ─────────────────────────────────────────────────────
    lstm_results: dict[str, Any] | None = None
    if not abort and not skip_lstm:
        logger.info("─" * 48)
        logger.info("STEP 4/5  LSTM Deep Learning")
        t0 = time.time()
        try:
            from models.lstm import run_lstm_experiment
            lstm_results = run_lstm_experiment()
            if lstm_results:
                m = lstm_results.get("test_metrics", {})
                summary["lstm"] = {
                    "r2": round(float(m.get("R²", float("nan"))), 6),
                    "mse": round(float(m.get("MSE", float("nan"))), 8),
                    "params": lstm_results.get("n_params", 0),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                logger.info("LSTM: R²=%.4f, %d params, %.1fs",
                            summary["lstm"]["r2"],
                            summary["lstm"]["n_params"],
                            summary["lstm"]["elapsed_s"])
            else:
                summary["lstm"] = {"skipped": "no data"}
        except ImportError as exc:
            logger.warning("LSTM skipped: %s", exc)
            summary["lstm"] = {"error": str(exc)}
        except Exception as exc:
            logger.exception("LSTM failed: %s", exc)
            summary["lstm"] = {"error": str(exc)}
    else:
        summary["lstm"] = {"skipped": skip_lstm}

    # ── Step 5 — Backtest (best model from ablation) ──────────────────────
    if not abort and ablation_results is not None:
        logger.info("─" * 48)
        logger.info("STEP 5/5  Backtest (best ablation model)")
        t0 = time.time()
        try:
            from features.factors import build_model_dataset
            from models.backtest import run_backtest
            from models.regression import train_test_split_by_time
            from models.regression import train_ridge, train_lasso, train_lightgbm

            # Use combined (量价+另类) config for backtest
            ds = build_model_dataset(feature_groups=["sentiment", "traditional"])
            if ds.empty:
                raise RuntimeError("Empty dataset")
            sc, dt = ds["stock_code"], ds["trade_date"]
            y = ds["fwd_ret_1d"]
            X = ds.drop(columns=["stock_code", "trade_date", "fwd_ret_1d"])
            nm = X.isna().any(axis=1) | y.isna()
            if nm.any():
                X, y = X.loc[~nm], y.loc[~nm]
                sc, dt = sc.loc[~nm], dt.loc[~nm]

            Xtr, Xte, ytr, yte, tr_i, te_i, _ = train_test_split_by_time(X, y, dt)

            models_for_bt = {
                "ridge": train_ridge(Xtr, ytr),
                "lasso": train_lasso(Xtr, ytr),
                "lightgbm": train_lightgbm(Xtr, ytr, Xte, yte),
            }

            bt = run_backtest(
                models_dict=models_for_bt,
                X_test=Xte,
                y_test=yte,
                stock_codes=sc.iloc[te_i].reset_index(drop=True),
                test_dates=dt.iloc[te_i].reset_index(drop=True),
            )
            summary["backtest"] = {
                "models": list(bt.keys()),
                "elapsed_s": round(time.time() - t0, 1),
            }
            for mn, r in bt.items():
                ls = r["metrics"].get("long_short", {})
                summary["backtest"].setdefault("long_short", {})[mn] = {
                    "sharpe": ls.get("sharpe_ratio"),
                    "ann_return": ls.get("annualized_return"),
                }
            logger.info("Backtest: %.1fs — %d model(s)",
                        summary["backtest"]["elapsed_s"], len(bt))
        except Exception as exc:
            logger.warning("Backtest skipped: %s", exc)
            summary["backtest"] = {"error": str(exc)}

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FULL PIPELINE SUMMARY")
    print("=" * 60)
    for step, data in summary.items():
        print(f"  {step}:")
        if isinstance(data, dict):
            for k, v in data.items():
                print(f"    {k}: {v}")
    print("=" * 60)
    return summary


if __name__ == "__main__":
    run_full_pipeline()
