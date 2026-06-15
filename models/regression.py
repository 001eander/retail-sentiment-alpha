"""
Regression models for retail-sentiment-alpha factor backtesting.

Trains Ridge, LASSO, and LightGBM on factor-derived features to predict
forward returns (``fwd_ret_1d``).  The module provides a full pipeline:

1. Load data from :func:`features.factors.build_model_dataset`
2. Chronological train / test split with standardisation
3. Train each model with cross-validated hyper-parameter selection
4. Evaluate and compare results
5. Generate diagnostic plots

Usage
-----
::

    uv run python -m models.regression
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------


def load_data(
    start_date: str | None = None,
    end_date: str | None = None,
    feature_groups: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load the factor dataset and separate features / target / dates.

    Calls ``features.factors.build_model_dataset()``, splits the result
    into a feature matrix *X*, a target vector *y* (``fwd_ret_1d``), and
    the ``trade_date`` column needed for chronological splitting.

    Parameters
    ----------
    start_date : str or None
        Passed through to ``build_model_dataset``.  ``None`` uses the
        default in that function.
    end_date : str or None
        Passed through to ``build_model_dataset``.
    feature_groups : list[str] or None
        Passed through to ``build_model_dataset``.  Controls which factor
        groups are included in the feature set for ablation experiments.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (factor + control columns).  ``stock_code`` and
        ``trade_date`` are dropped from the feature set.
    y : pd.Series
        Target — next-day forward return.
    dates : pd.Series
        Trade date of each row (for time-based splitting).

    Raises
    ------
    ValueError
        If the dataset is empty — instructs the user to run the factor
        pipeline first.
    """
    # Lazy import to avoid circular dependencies at package level
    from features.factors import build_model_dataset

    kwargs: dict[str, str | list[str]] = {}
    if start_date is not None:
        kwargs["start_date"] = start_date
    if end_date is not None:
        kwargs["end_date"] = end_date
    if feature_groups is not None:
        kwargs["feature_groups"] = feature_groups

    dataset = build_model_dataset(**kwargs)

    if dataset.empty:
        raise ValueError(
            "Empty dataset returned by build_model_dataset().\n\n"
            "You must run the factor pipeline first to populate the\n"
            "daily_factors and market_daily tables:\n\n"
            "    uv run python -m features.factors\n\n"
            "This will compute daily sentiment factors from social-media\n"
            "posts and build the modeling dataset."
        )

    # Keep a copy of the trade_date column before we drop it from X
    dates: pd.Series = dataset["trade_date"].copy()

    # Separate features and target
    y: pd.Series = dataset["fwd_ret_1d"].copy()
    X: pd.DataFrame = dataset.drop(columns=["stock_code", "trade_date", "fwd_ret_1d"])

    # Drop rows with NaN in either features or target
    nan_mask = X.isna().any(axis=1) | y.isna()
    n_nan = nan_mask.sum()
    if n_nan:
        logger.warning("Dropping %d row(s) with NaN values.", n_nan)
        X = X.loc[~nan_mask]
        y = y.loc[~nan_mask]
        dates = dates.loc[~nan_mask]

    # Warn if the dataset is very small
    if len(X) < 50:
        logger.warning(
            "Very small dataset: %d rows (fewer than 50).  "
            "Model performance may be unreliable.",
            len(X),
        )

    logger.info(
        "Loaded dataset: %d rows, %d feature(s), date range %s → %s.",
        len(X),
        X.shape[1],
        dates.min().date() if hasattr(dates.min(), "date") else dates.min(),
        dates.max().date() if hasattr(dates.max(), "date") else dates.max(),
    )
    logger.info("Feature columns: %s", list(X.columns))
    return X, y, dates


# ---------------------------------------------------------------------------
# 2. Chronological train / test split
# ---------------------------------------------------------------------------


def train_test_split_by_time(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    test_ratio: float = 0.2,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
    np.ndarray, np.ndarray, StandardScaler,
]:
    """Split data chronologically and standardise features.

    Sorts by unique ``dates``, uses the *last* ``test_ratio`` fraction of
    dates for the test set, and the remainder for training.  A
    :class:`~sklearn.preprocessing.StandardScaler` is fitted on the
    training set and applied to both partitions.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    dates : pd.Series
        Trade date for each row (aligned with *X* and *y*).
    test_ratio : float
        Fraction of *unique* dates to hold out for testing
        (default 0.2).

    Returns
    -------
    X_train : pd.DataFrame
        Standardised training features.
    X_test : pd.DataFrame
        Standardised test features.
    y_train : pd.Series
        Training target.
    y_test : pd.Series
        Test target.
    train_idx : np.ndarray
        Integer indices of the training rows in the original arrays.
    test_idx : np.ndarray
        Integer indices of the test rows.
    scaler : StandardScaler
        Fitted scaler (for later use on new data).
    """
    # Sort unique dates chronologically
    unique_dates = sorted(dates.unique())
    n_test = max(1, int(len(unique_dates) * test_ratio))
    test_date_set = set(unique_dates[-n_test:])

    test_mask = dates.isin(test_date_set).to_numpy(dtype=bool)
    train_mask = ~test_mask

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    X_train_raw = X.iloc[train_idx]
    X_test_raw = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Standardise
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=X_train_raw.columns,
        index=X_train_raw.index,
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=X_test_raw.columns,
        index=X_test_raw.index,
    )

    logger.info(
        "Chronological split: train %d rows (%s → %s), "
        "test %d rows (%s → %s).",
        len(X_train),
        unique_dates[0],
        unique_dates[-n_test - 1] if n_test < len(unique_dates) else unique_dates[0],
        len(X_test),
        unique_dates[-n_test],
        unique_dates[-1],
    )
    return X_train, X_test, y_train, y_test, train_idx, test_idx, scaler


# ---------------------------------------------------------------------------
# 3. Ridge regression (cross-validated alpha)
# ---------------------------------------------------------------------------


def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alphas: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train a Ridge regression model with cross-validated alpha selection.

    Uses :class:`~sklearn.model_selection.GridSearchCV` with
    :class:`~sklearn.linear_model.Ridge` and 5-fold CV to choose the
    best regularisation strength.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (already standardised).
    y_train : pd.Series
        Training target.
    alphas : np.ndarray or None
        Candidate alpha values.  Defaults to ``logspace(-3, 3, 20)``.

    Returns
    -------
    dict
        ``model`` : fitted :class:`~sklearn.linear_model.Ridge`
            Estimator with the best alpha.
        ``best_alpha`` : float
            Selected regularisation strength.
        ``cv_scores`` : np.ndarray
            Mean negative MSE per alpha from cross-validation.
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 20)

    param_grid = {"alpha": alphas}
    gscv = GridSearchCV(
        Ridge(),
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
    )
    gscv.fit(X_train, y_train)

    best_alpha = gscv.best_params_["alpha"]
    # mean_test_score is neg_mean_squared_error (higher = better);
    # negate to get positive MSE for interpretability.
    cv_scores = -gscv.cv_results_["mean_test_score"]

    logger.info(
        "Ridge — best alpha: %.6f, CV MSE: %.6f",
        best_alpha,
        cv_scores[gscv.best_index_],
    )

    return {
        "model": gscv.best_estimator_,
        "best_alpha": best_alpha,
        "cv_scores": cv_scores,
        "alphas": alphas,
    }


# ---------------------------------------------------------------------------
# 4. LASSO regression (cross-validated alpha)
# ---------------------------------------------------------------------------


def train_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alphas: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train a LASSO regression model with cross-validated alpha selection.

    Uses :class:`~sklearn.linear_model.LassoCV` with 5-fold CV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (already standardised).
    y_train : pd.Series
        Training target.
    alphas : np.ndarray or None
        Candidate alpha values.  Defaults to ``logspace(-4, 1, 20)``.

    Returns
    -------
    dict
        ``model`` : fitted :class:`~sklearn.linear_model.LassoCV`
            The entire CV object (so that ``model.coef_`` is available).
        ``best_alpha`` : float
            Selected regularisation strength.
        ``cv_scores`` : np.ndarray
            Mean MSE per alpha across CV folds.
    """
    if alphas is None:
        alphas = np.logspace(-4, 1, 20)

    lasso_cv = LassoCV(
        alphas=alphas,
        cv=5,
        max_iter=5000,
        random_state=42,
    )
    with warnings.catch_warnings():
        # LassoCV may emit convergence warnings on some alpha values;
        # the best alpha will still be valid.
        warnings.simplefilter("ignore", category=UserWarning)
        lasso_cv.fit(X_train, y_train)

    best_alpha = lasso_cv.alpha_
    # mse_path_ shape: (n_alphas, n_folds) → mean over folds
    cv_scores = lasso_cv.mse_path_.mean(axis=1)

    n_selected = int(np.sum(lasso_cv.coef_ != 0))
    logger.info(
        "LASSO — best alpha: %.6f, CV MSE: %.6f, non-zero coef: %d / %d",
        best_alpha,
        cv_scores[list(lasso_cv.alphas_).index(best_alpha)]
        if best_alpha in lasso_cv.alphas_
        else float("nan"),
        n_selected,
        len(lasso_cv.coef_),
    )

    if n_selected == 0:
        logger.warning(
            "LASSO selected NO features — all coefficients are zero.  "
            "Consider lowering the alpha range."
        )

    return {
        "model": lasso_cv,
        "best_alpha": best_alpha,
        "cv_scores": cv_scores,
        "alphas": lasso_cv.alphas_,
    }


# ---------------------------------------------------------------------------
# 5. LightGBM
# ---------------------------------------------------------------------------


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Train a LightGBM regressor with early stopping.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_test : pd.DataFrame
        Validation features (used for early stopping).
    y_test : pd.Series
        Validation target.

    Returns
    -------
    dict
        ``model`` : fitted :class:`lightgbm.LGBMRegressor`
        ``feature_importance`` : pd.DataFrame
            Columns ``feature`` and ``importance``, sorted descending.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required for tree-based models.\n"
            "Install with:  pip install lightgbm   (or: uv add lightgbm)"
        ) from exc

    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,  # suppress internal chatter; callback handles logging
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
    )

    best_iter = lgb_model.best_iteration_
    logger.info(
        "LightGBM — best iteration: %d, validation %s: %.6f",
        best_iter,
        "l2",
        lgb_model.best_score_["valid_0"]["l2"],
    )

    # Feature importance
    importance_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": lgb_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return {
        "model": lgb_model,
        "feature_importance": importance_df,
    }


# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict[str, float]:
    """Compute standard regression metrics on a held-out test set.

    Parameters
    ----------
    model : sklearn or lightgbm estimator
        Fitted model with a ``predict`` method.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True target values.
    model_name : str
        Human-readable name (used in log output).

    Returns
    -------
    dict
        ``r2``, ``mse``, ``rmse``, ``mae``.
    """
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, y_pred)

    logger.info(
        "%s — R²: %.4f, MSE: %.6f, RMSE: %.6f, MAE: %.6f",
        model_name,
        r2,
        mse,
        rmse,
        mae,
    )
    return {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae}


# ---------------------------------------------------------------------------
# 7. Model comparison
# ---------------------------------------------------------------------------


def compare_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a comparison table from evaluation results.

    Parameters
    ----------
    results : dict
        ``{model_name: metrics_dict}`` where *metrics_dict* comes from
        :func:`evaluate_model`.

    Returns
    -------
    pd.DataFrame
        Models sorted by R² (best first).
    """
    records = []
    for name, metrics in results.items():
        records.append(
            {
                "Model": name,
                "R²": metrics["r2"],
                "MSE": metrics["mse"],
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
            }
        )
    comparison = pd.DataFrame(records).sort_values("R²", ascending=False)
    comparison.reset_index(drop=True, inplace=True)

    # Print a formatted comparison table
    logger.info("=" * 80)
    logger.info("MODEL COMPARISON (sorted by R²)")
    logger.info("=" * 80)
    for _, row in comparison.iterrows():
        logger.info(
            "  %-12s  R²: %7.4f  MSE: %9.6f  RMSE: %9.6f  MAE: %9.6f",
            row["Model"],
            row["R²"],
            row["MSE"],
            row["RMSE"],
            row["MAE"],
        )
    logger.info("-" * 80)

    print("\n" + comparison.to_string(float_format=lambda v: f"{v:.6f}"))
    return comparison


# ---------------------------------------------------------------------------
# 8. Plotting
# ---------------------------------------------------------------------------


def plot_results(
    results: dict[str, Any],
    output_dir: str = "outputs",
) -> None:
    """Generate and save diagnostic plots to *output_dir*.

    Plots
    -----
    * ``predicted_vs_actual.png`` — scatter of predicted vs true returns
      for each model (one subplot per model).
    * ``feature_importance.png`` — bar chart of coefficients (Ridge /
      LASSO) or gain importance (LightGBM).  Skipped when there is only
      **1** feature.
    * ``alpha_selection.png`` — CV MSE vs alpha for Ridge and LASSO
      (semilog-x).

    Parameters
    ----------
    results : dict
        Rich results dictionary from :func:`run_modeling_pipeline`.
        Expected keys: ``ridge``, ``lasso``, ``lightgbm``,
        ``feature_names``, ``X_test``, ``y_test``.
    output_dir : str
        Directory to save images (created if it does not exist).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_test: pd.DataFrame = results["X_test"]
    y_test: pd.Series = results["y_test"]
    feature_names: list[str] = results["feature_names"]

    # ── 1. Predicted vs actual ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    models_for_plot = [
        ("Ridge", results["ridge"]["model"]),
        ("LASSO", results["lasso"]["model"]),
        ("LightGBM", results["lightgbm"]["model"]),
    ]

    for ax, (name, model) in zip(axes, models_for_plot):  # noqa: B905  # pyright: ignore
        y_pred = model.predict(X_test)
        ax.scatter(y_test, y_pred, alpha=0.5, s=15, edgecolors="none")
        # Diagonal line
        lims = [
            min(y_test.min(), y_pred.min()),
            max(y_test.max(), y_pred.max()),
        ]
        ax.plot(lims, lims, "r--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(name)
        ax.axis("square")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    fig.suptitle("Predicted vs Actual Returns", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)
    logger.info("Saved predicted_vs_actual.png")

    # ── 2. Feature importance ──────────────────────────────────────────
    # Skip when only 1 feature — the bar chart is trivially uninformative
    n_features = len(feature_names)
    if n_features > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, name in zip(axes, ["Ridge", "LASSO", "LightGBM"]):  # noqa: B905  # pyright: ignore
            model_entry = results[name.lower()]  # type: ignore[misc]
            model = model_entry["model"]

            if name == "LightGBM":
                imp = model.feature_importances_
                title_suffix = "(gain)"
            else:
                imp = model.coef_
                title_suffix = "(coefficient)"

            # Sort by absolute value
            order = np.argsort(np.abs(imp))[::-1]
            sorted_features = [feature_names[i] for i in order]
            sorted_imp = imp[order]

            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sorted_imp]
            ax.barh(range(len(sorted_imp)), sorted_imp, color=colors, height=0.7)
            ax.set_yticks(range(len(sorted_imp)))
            ax.set_yticklabels(sorted_features, fontsize=8)
            ax.axvline(0, color="gray", linewidth=0.5)
            ax.set_title(f"{name} {title_suffix}")
            ax.invert_yaxis()

        fig.suptitle("Feature Importance / Coefficients", fontsize=13)
        fig.tight_layout()
        fig.savefig(out / "feature_importance.png", dpi=150)
        plt.close(fig)
        logger.info("Saved feature_importance.png")
    else:
        logger.info(
            "Skipping feature_importance plot — only %d feature(s).", n_features
        )

    # ── 3. Alpha selection (CV score vs alpha) ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, key, title in zip(
        axes, ["ridge", "lasso"], ["Ridge CV (neg MSE)", "LASSO CV (MSE)"]
    ):
        entry = results[key]
        alphas = entry["alphas"]
        cv_scores = entry["cv_scores"]
        best_alpha = entry["best_alpha"]

        ax.semilogx(alphas, cv_scores, marker=".", linestyle="-", linewidth=1.5)
        ax.axvline(best_alpha, color="red", linestyle="--", alpha=0.6,
                   label=f"best α = {best_alpha:.5f}")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("CV MSE")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Alpha Selection — Cross-Validation MSE", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "alpha_selection.png", dpi=150)
    plt.close(fig)
    logger.info("Saved alpha_selection.png")


# ---------------------------------------------------------------------------
# 9. Full pipeline
# ---------------------------------------------------------------------------


def run_modeling_pipeline(
    test_ratio: float = 0.2,
    ridge_alphas: np.ndarray | None = None,
    lasso_alphas: np.ndarray | None = None,
    output_dir: str = "outputs",
) -> dict[str, Any]:
    """Orchestrate the full modeling workflow.

    Steps
    -----
    1. Load data (:func:`load_data`)
    2. Chronological split (:func:`train_test_split_by_time`)
    3. Train Ridge (:func:`train_ridge`)
    4. Train LASSO (:func:`train_lasso`)
    5. Train LightGBM (:func:`train_lightgbm`)
    6. Evaluate all models (:func:`evaluate_model`)
    7. Compare (:func:`compare_models`)
    8. Plot (:func:`plot_results`)

    Parameters
    ----------
    test_ratio : float
        Fraction of dates for the test set (default 0.2).
    ridge_alphas : np.ndarray or None
        Alpha candidates for Ridge.
    lasso_alphas : np.ndarray or None
        Alpha candidates for LASSO.
    output_dir : str
        Directory for diagnostic plots.

    Returns
    -------
    dict
        Comprehensive results dict containing:
        ``ridge``, ``lasso``, ``lightgbm`` (model results),
        ``metrics`` (evaluation metrics per model),
        ``comparison`` (comparison DataFrame),
        ``X_test``, ``y_test``, ``feature_names``.
    """
    logger.info("=" * 60)
    logger.info("MODELING PIPELINE STARTED")
    logger.info("=" * 60)

    # Step 1: Load
    X, y, dates = load_data()
    # Step 2: Split
    X_train, X_test, y_train, y_test, train_idx, test_idx, scaler = (
        train_test_split_by_time(X, y, dates, test_ratio=test_ratio)
    )

    feature_names = list(X.columns)

    # Step 3: Ridge
    logger.info("-" * 40)
    logger.info("Training Ridge …")
    ridge_result = train_ridge(X_train, y_train, alphas=ridge_alphas)

    # Step 4: LASSO
    logger.info("-" * 40)
    logger.info("Training LASSO …")
    lasso_result = train_lasso(X_train, y_train, alphas=lasso_alphas)

    # Step 5: LightGBM
    logger.info("-" * 40)
    logger.info("Training LightGBM …")
    lgbm_result = train_lightgbm(X_train, y_train, X_test, y_test)

    # Step 6: Evaluate
    logger.info("-" * 40)
    logger.info("Evaluating models …")
    ridge_metrics = evaluate_model(ridge_result["model"], X_test, y_test, "Ridge")
    lasso_metrics = evaluate_model(lasso_result["model"], X_test, y_test, "LASSO")
    lgbm_metrics = evaluate_model(lgbm_result["model"], X_test, y_test, "LightGBM")

    metrics = {
        "Ridge": ridge_metrics,
        "LASSO": lasso_metrics,
        "LightGBM": lgbm_metrics,
    }

    # Step 7: Compare
    logger.info("-" * 40)
    comparison = compare_models(metrics)

    # Step 8: Plot
    plot_results(
        {
            "ridge": ridge_result,
            "lasso": lasso_result,
            "lightgbm": lgbm_result,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": feature_names,
        },
        output_dir=output_dir,
    )

    logger.info("=" * 60)
    logger.info("MODELING PIPELINE COMPLETE")
    logger.info("=" * 60)

    return {
        "ridge": ridge_result,
        "lasso": lasso_result,
        "lightgbm": lgbm_result,
        "metrics": metrics,
        "comparison": comparison,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ---------------------------------------------------------------------------
# 10. Ablation experiments
# ---------------------------------------------------------------------------


def run_ablation_experiments() -> dict[str, Any]:
    """Run factor-group ablation experiments comparing model performance.

    Defines three feature-group configurations:

    * ``"纯量价"`` — traditional price-volume factors only
    * ``"纯另类"`` — sentiment / alternative factors only
    * ``"量价+另类"`` — both sentiment and traditional factors

    For each configuration a full modeling pipeline is run (Ridge, LASSO,
    LightGBM).  A comparison table is printed and a comprehensive results
    dictionary is returned.

    Returns
    -------
    dict
        Nested results::

            {
                "纯量价": {"Ridge": metrics, "LASSO": metrics, "LightGBM": metrics},
                "纯另类": {"Ridge": metrics, "LASSO": metrics, "LightGBM": metrics},
                "量价+另类": {"Ridge": metrics, "LASSO": metrics, "LightGBM": metrics},
                "comparison_table": pd.DataFrame,
            }

        Each *metrics* dict contains ``r2``, ``mse``, ``rmse``, ``mae``.
    """
    logger.info("=" * 70)
    logger.info("FACTOR-GROUP ABLATION EXPERIMENTS")
    logger.info("=" * 70)

    # ── 1. Define configurations ─────────────────────────────────────────
    configs: dict[str, list[str]] = {
        "纯量价": ["traditional"],
        "纯另类": ["sentiment"],
        "量价+另类": ["sentiment", "traditional"],
    }

    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    comparison_records: list[dict[str, Any]] = []

    for label, groups in configs.items():
        logger.info("")
        logger.info("─" * 50)
        logger.info("Config: %s  (groups: %s)", label, groups)
        logger.info("─" * 50)

        # Step 1: Load data with the current feature-group filter
        try:
            X, y, dates = load_data(feature_groups=groups)
        except ValueError as exc:
            logger.warning(
                "Skipping config '%s' — %s", label, exc,
            )
            continue

        # Step 2: Chronological split
        X_train, X_test, y_train, y_test, train_idx, test_idx, scaler = (
            train_test_split_by_time(X, y, dates)
        )

        # Step 3: Train models
        logger.info("  Training Ridge …")
        ridge_result = train_ridge(X_train, y_train)

        logger.info("  Training LASSO …")
        lasso_result = train_lasso(X_train, y_train)

        logger.info("  Training LightGBM …")
        lgbm_result = train_lightgbm(X_train, y_train, X_test, y_test)

        # Step 4: Evaluate
        ridge_metrics = evaluate_model(
            ridge_result["model"], X_test, y_test, f"{label} / Ridge",
        )
        lasso_metrics = evaluate_model(
            lasso_result["model"], X_test, y_test, f"{label} / LASSO",
        )
        lgbm_metrics = evaluate_model(
            lgbm_result["model"], X_test, y_test, f"{label} / LightGBM",
        )

        config_metrics = {
            "Ridge": ridge_metrics,
            "LASSO": lasso_metrics,
            "LightGBM": lgbm_metrics,
        }
        all_metrics[label] = config_metrics

        for model_name, metrics in config_metrics.items():
            comparison_records.append({
                "Feature Group": label,
                "Model": model_name,
                "R²": metrics["r2"],
                "MSE": metrics["mse"],
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
            })

    # ── 5. Build and print comparison table ──────────────────────────────
    comparison_table = pd.DataFrame(comparison_records)
    if not comparison_table.empty:
        comparison_table.sort_values("R²", ascending=False, inplace=True)
        comparison_table.reset_index(drop=True, inplace=True)

        print("\n" + "=" * 90)
        print("  ABLATION EXPERIMENT COMPARISON  (sorted by R²)")
        print("=" * 90)
        print(
            comparison_table.to_string(
                float_format=lambda v: f"{v:.6f}",
                index=False,
            )
        )
        print("-" * 90)
        print()
    else:
        logger.warning("No ablation results to compare — all configs were skipped.")

    logger.info("=" * 70)
    logger.info("ABLATION EXPERIMENTS COMPLETE")
    logger.info("=" * 70)

    return {
        **all_metrics,
        "comparison_table": comparison_table,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    results = run_modeling_pipeline()
    print("\nPipeline finished.")
    print(f"  Training rows:    {results['n_train']}")
    print(f"  Test rows:        {results['n_test']}")
    print(f"  Best model (R²):  {results['comparison'].iloc[0]['Model']} "
          f"({results['comparison'].iloc[0]['R²']:.4f})")
