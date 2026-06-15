"""
LSTM deep learning model for stock return prediction.

Converts cross-sectional factor data into per-stock sequences and trains
an LSTM regressor to predict next-day forward returns (``fwd_ret_1d``).

Usage
-----
::

    uv run python -m models.lstm
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger.info("LSTM using device: %s", device)


# ---------------------------------------------------------------------------
# 1. Data → sequences
# ---------------------------------------------------------------------------


def build_sequences(
    df: pd.DataFrame,
    seq_len: int = 20,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a cross-sectional DataFrame into LSTM-ready sequences.

    For each stock (grouped by ``stock_code``):
    - Sort by ``trade_date``
    - Create sliding windows of length *seq_len*
    - Each window produces one sample

    Parameters
    ----------
    df : pd.DataFrame
        Dataset from ``build_model_dataset()`` (must contain ``stock_code``,
        ``trade_date``, ``fwd_ret_1d``, and feature columns).
    seq_len : int
        Number of days per input sequence (default 20).
    feature_cols : list[str] or None
        Which columns to use as features.  If ``None``, all numeric columns
        except ``stock_code``, ``trade_date``, ``fwd_ret_1d`` are used.

    Returns
    -------
    X : np.ndarray
        Shape ``(n_samples, seq_len, n_features)``.
    y : np.ndarray
        Shape ``(n_samples,)`` — ``fwd_ret_1d`` of the last day in each window.
    dates : np.ndarray
        Shape ``(n_samples,)`` — ``trade_date`` of the last day in each window.
    stock_codes : np.ndarray
        Shape ``(n_samples,)`` — stock code for each sample.

    Notes
    -----
    Stocks with fewer than *seq_len* rows after filtering are silently
    skipped (a debug log is emitted).
    Windows containing any NaN in features or target are dropped.
    """
    if df.empty:
        logger.warning("build_sequences received empty DataFrame.")
        return (
            np.empty((0, seq_len, 0)),
            np.empty((0,)),
            np.empty((0,)),
            np.empty((0,)),
        )

    # Infer feature columns if not provided
    if feature_cols is None:
        feature_cols = [
            c
            for c in df.columns
            if c not in ("stock_code", "trade_date", "fwd_ret_1d")
            and np.issubdtype(df[c].dtype, np.number)
        ]

    if not feature_cols:
        logger.error("No numeric feature columns found.")
        return (
            np.empty((0, seq_len, 0)),
            np.empty((0,)),
            np.empty((0,)),
            np.empty((0,)),
        )

    n_features = len(feature_cols)
    X_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    dates_blocks: list[np.ndarray] = []
    codes_blocks: list[np.ndarray] = []

    for stock_code, group in df.groupby("stock_code"):
        group = group.sort_values("trade_date").reset_index(drop=True)
        n = len(group)
        if n < seq_len:
            logger.debug(
                "Stock %s: only %d rows, need %d — skipping.",
                stock_code,
                n,
                seq_len,
            )
            continue

        # Extract feature values and target once for this stock
        feat_vals = group[feature_cols].values.astype(np.float64)
        target_vals = group["fwd_ret_1d"].values.astype(np.float64)
        date_vals = group["trade_date"].values
        code_vals = np.full(n, stock_code)

        # Slide window
        for i in range(n - seq_len + 1):
            window_feat = feat_vals[i : i + seq_len]
            if np.any(np.isnan(window_feat)):
                continue
            last_target = target_vals[i + seq_len - 1]
            if np.isnan(last_target):
                continue

            X_blocks.append(window_feat)
            y_blocks.append(np.array([last_target]))
            dates_blocks.append(np.array([date_vals[i + seq_len - 1]]))
            codes_blocks.append(np.array([stock_code]))

    if not X_blocks:
        logger.warning(
            "No valid sequences could be built — check data availability."
        )
        return (
            np.empty((0, seq_len, n_features)),
            np.empty((0,)),
            np.empty((0,)),
            np.empty((0,)),
        )

    X = np.concatenate(X_blocks, axis=0).reshape(-1, seq_len, n_features)
    y = np.concatenate(y_blocks, axis=0)
    dates = np.concatenate(dates_blocks, axis=0)
    stock_codes = np.concatenate(codes_blocks, axis=0)

    logger.info(
        "Built %d sequences (seq_len=%d, %d features) from %d stock(s).",
        len(X),
        seq_len,
        n_features,
        df["stock_code"].nunique(),
    )
    return X, y, dates, stock_codes


# ---------------------------------------------------------------------------
# 2. LSTM model definition
# ---------------------------------------------------------------------------


class LSTMPredictor(nn.Module):
    """LSTM regressor for next-day stock return prediction.

    Parameters
    ----------
    input_dim : int
        Number of features at each time step.
    hidden_dim : int
        Hidden state size of the LSTM (default 64).
    num_layers : int
        Number of stacked LSTM layers (default 2).
    dropout : float
        Dropout probability between LSTM layers (default 0.3).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, seq_len, input_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, 1)`` — predicted return.
        """
        # lstm_out: (batch_size, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        return self.fc(last_out)  # (batch_size, 1)


# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------


def train_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.001,
    patience: int = 15,
) -> dict[str, Any]:
    """Train an LSTM model with early stopping.

    Parameters
    ----------
    model : nn.Module
        Instance of :class:`LSTMPredictor` (or compatible).
    train_loader : DataLoader
        Batched training data (``(X, y)``).
    val_loader : DataLoader
        Batched validation data (``(X, y)``).
    epochs : int
        Maximum number of training epochs (default 100).
    lr : float
        Adam learning rate (default 0.001).
    patience : int
        Early-stopping patience on validation loss (default 15).

    Returns
    -------
    dict
        ``model`` : best model state restored.
        ``train_losses`` : list of epoch-level training MSE.
        ``val_losses`` : list of epoch-level validation MSE.
        ``best_epoch`` : epoch with lowest validation loss (1-indexed).
    """
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] = {}
    epochs_without_improvement = 0

    logger.info(
        "Training LSTM — %d parameters, %d epochs max, lr=%.4f, patience=%d.",
        sum(p.numel() for p in model.parameters()),
        epochs,
        lr,
        patience,
    )

    for epoch in range(1, epochs + 1):
        # ── Training ────────────────────────────────────────────────────
        model.train()
        epoch_train_loss = 0.0
        n_train_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)  # (batch, 1)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = epoch_train_loss / max(n_train_batches, 1)
        train_losses.append(avg_train_loss)

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        epoch_val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)

                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                epoch_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = epoch_val_loss / max(n_val_batches, 1)
        val_losses.append(avg_val_loss)

        # ── Log ─────────────────────────────────────────────────────────
        if epoch == 1 or epoch % 10 == 0:
            logger.info(
                "Epoch %3d/%d — train_loss: %.6f, val_loss: %.6f",
                epoch,
                epochs,
                avg_train_loss,
                avg_val_loss,
            )

        # ── Early stopping ──────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state = {
                k: v.clone().cpu() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping triggered at epoch %d — "
                    "best epoch %d (val_loss=%.6f).",
                    epoch,
                    best_epoch,
                    best_val_loss,
                )
                break

    # Restore best model
    model.load_state_dict(best_state)
    logger.info(
        "Training complete — best epoch %d, best val_loss %.6f.",
        best_epoch,
        best_val_loss,
    )

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
    }


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------


def evaluate_lstm(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: StandardScaler | None = None,
) -> dict[str, float]:
    """Evaluate an LSTM model on a held-out test set.

    Parameters
    ----------
    model : nn.Module
        Trained LSTM model.
    X_test : np.ndarray
        Test sequences, shape ``(n_samples, seq_len, n_features)``.
    y_test : np.ndarray
        True target values, shape ``(n_samples,)``.
    scaler_y : StandardScaler or None
        If provided, predictions and *y_test* are inverse-transformed back to
        original scale before computing metrics.

    Returns
    -------
    dict
        ``R²`` : Coefficient of determination.
        ``MSE`` : Mean squared error.
        ``RMSE`` : Root mean squared error.
        ``MAE`` : Mean absolute error.
    """
    model.eval()
    model.to(device)

    # Batch predictions through the model to handle large test sets
    dataset = TensorDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(y_test)
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_preds: list[np.ndarray] = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds.append(preds.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0).ravel()
    y_true = y_test.copy()

    # Inverse transform if scaler provided
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_true = scaler_y.inverse_transform(y_true.reshape(-1, 1)).ravel()

    # Compute metrics
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)

    logger.info(
        "LSTM — R²: %.4f, MSE: %.6f, RMSE: %.6f, MAE: %.6f",
        r2,
        mse,
        rmse,
        mae,
    )

    return {"R²": r2, "MSE": mse, "RMSE": rmse, "MAE": mae}


# ---------------------------------------------------------------------------
# 5. Full experiment orchestration
# ---------------------------------------------------------------------------


def run_lstm_experiment(df: pd.DataFrame | None = None) -> dict[str, Any] | None:
    """Orchestrate a complete LSTM experiment.

    Steps
    -----
    1. Load data (if *df* is ``None``, call ``build_model_dataset()``).
    2. Chronological split: train 60 %, val 20 %, test 20 % by date.
    3. Build sequences for each split.
    4. Scale features with :class:`~sklearn.preprocessing.StandardScaler`
       (fit on train only).
    5. Scale target with :class:`~sklearn.preprocessing.StandardScaler`
       (fit on train; inverse-transform for evaluation).
    6. Create :class:`~torch.utils.data.DataLoader` s.
    7. Train :class:`LSTMPredictor`.
    8. Evaluate on test set.
    9. Return results.

    Parameters
    ----------
    df : pd.DataFrame or None
        Pre-built dataset (optional).  When ``None``, the dataset is loaded
        via ``features.factors.build_model_dataset()``.

    Returns
    -------
    dict or None
        ``model`` : trained :class:`LSTMPredictor`.
        ``train_losses``, ``val_losses`` : loss histories.
        ``best_epoch`` : epoch with lowest validation loss.
        ``test_metrics`` : evaluation metrics dict.
        ``n_train``, ``n_val``, ``n_test`` : sample counts.
        Returns ``None`` when the dataset is empty.
    """
    # ── 1. Load data ────────────────────────────────────────────────────
    if df is None:
        try:
            from features.factors import build_model_dataset
        except ImportError as exc:
            logger.error(
                "Cannot import build_model_dataset — %s", exc
            )
            return None

        logger.info("Loading model dataset …")
        df = build_model_dataset()
    else:
        df = df.copy()

    if df.empty:
        logger.error(
            "Empty dataset — cannot run LSTM experiment.\n\n"
            "Run the factor pipeline first:\n\n"
            "    uv run python -m features.factors"
        )
        return None

    # Ensure sorted
    df = df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)

    # Infer feature columns
    feature_cols = [
        c
        for c in df.columns
        if c not in ("stock_code", "trade_date", "fwd_ret_1d")
        and np.issubdtype(df[c].dtype, np.number)
    ]

    if not feature_cols:
        logger.error("No numeric feature columns found in dataset.")
        return None

    logger.info(
        "Dataset: %d rows, %d feature(s), %d stock(s).",
        len(df),
        len(feature_cols),
        df["stock_code"].nunique(),
    )

    # ── 2. Chronological split ──────────────────────────────────────────
    unique_dates = sorted(df["trade_date"].unique())
    n_dates = len(unique_dates)
    train_cut = int(n_dates * 0.6)
    val_cut = int(n_dates * 0.8)

    train_dates = set(unique_dates[:train_cut])
    val_dates = set(unique_dates[train_cut:val_cut])
    test_dates = set(unique_dates[val_cut:])

    train_df = df[df["trade_date"].isin(train_dates)].copy()
    val_df = df[df["trade_date"].isin(val_dates)].copy()
    test_df = df[df["trade_date"].isin(test_dates)].copy()

    logger.info(
        "Chronological split: train %d dates (%s → %s), "
        "val %d dates, test %d dates.",
        len(train_dates),
        unique_dates[0],
        unique_dates[train_cut - 1] if train_cut > 0 else unique_dates[0],
        len(val_dates),
        len(test_dates),
    )

    # ── 3. Scale features ───────────────────────────────────────────────
    scaler_X = StandardScaler()
    train_df[feature_cols] = scaler_X.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler_X.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler_X.transform(test_df[feature_cols])

    # ── 4. Scale target ─────────────────────────────────────────────────
    scaler_y = StandardScaler()
    train_target = scaler_y.fit_transform(
        train_df[["fwd_ret_1d"]]
    ).ravel()
    train_df["fwd_ret_1d"] = train_target

    val_df["fwd_ret_1d"] = scaler_y.transform(
        val_df[["fwd_ret_1d"]]
    ).ravel()
    test_df["fwd_ret_1d"] = scaler_y.transform(
        test_df[["fwd_ret_1d"]]
    ).ravel()

    # ── 5. Build sequences ──────────────────────────────────────────────
    seq_len = 20
    X_train, y_train, _, _ = build_sequences(
        train_df, seq_len=seq_len, feature_cols=feature_cols
    )
    X_val, y_val, _, _ = build_sequences(
        val_df, seq_len=seq_len, feature_cols=feature_cols
    )
    X_test, y_test, _, _ = build_sequences(
        test_df, seq_len=seq_len, feature_cols=feature_cols
    )

    if len(X_train) == 0:
        logger.error(
            "No training sequences could be built — "
            "check that stocks have at least %d consecutive days.",
            seq_len,
        )
        return None

    logger.info(
        "Sequences: train %d, val %d, test %d.",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # ── 6. Create DataLoaders ───────────────────────────────────────────
    batch_size = 64
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # ── 7. Train LSTM ────────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Training LSTM …")
    logger.info("-" * 50)

    model = LSTMPredictor(input_dim=len(feature_cols))
    train_result = train_lstm(
        model,
        train_loader,
        val_loader,
        epochs=100,
        lr=0.001,
        patience=15,
    )

    # ── 8. Evaluate on test set ──────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Evaluating on test set …")
    logger.info("-" * 50)

    test_metrics = evaluate_lstm(
        train_result["model"], X_test, y_test, scaler_y=scaler_y
    )

    # ── 9. Return results ────────────────────────────────────────────────
    results = {
        "model": train_result["model"],
        "train_losses": train_result["train_losses"],
        "val_losses": train_result["val_losses"],
        "best_epoch": train_result["best_epoch"],
        "test_metrics": test_metrics,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "seq_len": seq_len,
        "n_features": len(feature_cols),
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
    }

    logger.info("=" * 50)
    logger.info("LSTM EXPERIMENT COMPLETE")
    logger.info("=" * 50)
    logger.info(
        "Test — R²: %.4f, MSE: %.6f",
        test_metrics["R²"],
        test_metrics["MSE"],
    )

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    results = run_lstm_experiment()
    if results:
        print(f"LSTM Test R²: {results['test_metrics']['R²']:.4f}")
    else:
        print("LSTM experiment failed — see logs above.")
