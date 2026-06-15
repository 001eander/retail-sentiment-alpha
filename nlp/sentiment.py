"""Batch sentiment scoring module using FinBERT-Chinese.

Maps Chinese financial text to a real-valued sentiment score in [-1, +1]
using ``yiyanghkust/finbert-tone-chinese`` from HuggingFace.

Score formula:  ``score = prob_positive - prob_negative``
    +1 strongly positive, 0 neutral, -1 strongly negative.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text
from transformers import pipeline

from crawlers.config import engine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "yiyanghkust/finbert-tone-chinese"
"""HuggingFace model identifier."""

PROGRESS_INTERVAL = 10
"""Log a progress line every N posts."""

MAX_INPUT_CHARS = 512
"""Truncate combined title+content to this many characters."""

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(device: str = "cpu") -> pipeline:
    """Load the FinBERT-Chinese sentiment-analysis pipeline.

    Parameters
    ----------
    device:
        Target device string (``"cpu"`` or ``"mps"``).  Falls back to CPU when
        MPS is requested but not available.

    Returns
    -------
    ``transformers.pipeline``
        A sentiment-analysis pipeline ready for inference.
    """
    resolved = _resolve_device(device)
    logger.info("Loading model %s on device=%s ...", MODEL_NAME, resolved)
    clf = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=resolved,
    )
    logger.info("Model loaded successfully (device=%s)", resolved)
    return clf


def _resolve_device(device: str) -> int | str:
    """Resolve a device string to a value accepted by ``pipeline(device=...)``.

    * ``"cpu"`` → ``-1``
    * ``"mps"`` → ``-1`` if MPS is unavailable, else ``"mps"``.

    Returns
    -------
    ``-1`` (CPU) or ``"mps"``.
    """
    if device == "cpu":
        return -1
    if device == "mps":
        try:
            import torch  # noqa: F811

            if torch.backends.mps.is_available():
                logger.info("MPS device is available, using MPS.")
                return "mps"
            logger.warning("MPS requested but not available; falling back to CPU.")
        except ImportError:
            logger.warning("torch not available; falling back to CPU.")
        return -1
    logger.warning("Unknown device %r; falling back to CPU.", device)
    return -1


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_texts(
    texts: list[str],
    classifier: pipeline,
    batch_size: int = 1,
) -> list[float]:
    """Run the classifier on a list of texts and return sentiment scores.

    Parameters
    ----------
    texts:
        Input strings to classify.
    classifier:
        A HuggingFace sentiment-analysis pipeline (from :func:`load_model`).
    batch_size:
        Number of texts to process per batch (default ``1``).

    Returns
    -------
    list[float]
        Sentiment scores in ``[-1, 1]``, one per input text.  A score of
        ``0.0`` is returned for any text that failed during inference.
    """
    if not texts:
        return []

    scores: list[float] = []
    # Process in user-defined batches
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            results = classifier(batch)
        except Exception:
            logger.exception("Batch inference failed for texts %d–%d", start, start + len(batch) - 1)
            scores.extend(0.0 for _ in batch)
            continue

        # classifier returns a single dict when batch_size=1 pipeline-side,
        # but we always feed a list so it returns a list.
        if not isinstance(results, list):
            results = [results]

        for item in results:
            try:
                label = item["label"].lower()
                prob = item["score"]
                if label in ("positive", "label_1"):
                    scores.append(prob)
                elif label in ("negative", "label_2"):
                    scores.append(-prob)
                else:  # neutral / label_0
                    scores.append(0.0)
            except (KeyError, TypeError) as exc:
                logger.warning("Malformed classifier result %r: %s", item, exc)
                scores.append(0.0)

    return scores


def _prepare_text(post: dict[str, Any]) -> str | None:
    """Combine *title* and *content* from a post into a single input string.

    Returns ``None`` when no usable text is available (post is empty after
    stripping).
    """
    title = (post.get("title") or "").strip()
    content = (post.get("content") or "").strip()
    combined = (title + "  " + content).strip()
    if not combined:
        return None
    # Truncate to avoid exceeding model's max input length
    return combined[:MAX_INPUT_CHARS]


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def get_unscored_posts(
    engine: Any,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch posts that have not yet been scored.

    Selects rows where ``sentiment IS NULL`` and a non-empty ``content``
    column exists.

    Parameters
    ----------
    engine:
        SQLAlchemy engine instance.
    limit:
        Maximum number of posts to return (``None`` = no limit).

    Returns
    -------
    list[dict]
        Each dict has keys ``id``, ``title``, ``content``.
    """
    sql = text(
        "SELECT id, title, content FROM posts "
        "WHERE sentiment IS NULL AND content IS NOT NULL AND content != '' "
        "ORDER BY id"
    )
    if limit is not None:
        sql = text(
            "SELECT id, title, content FROM posts "
            "WHERE sentiment IS NULL AND content IS NOT NULL AND content != '' "
            "ORDER BY id LIMIT :limit"
        )

    with engine.connect() as conn:
        rows = conn.execute(sql, {"limit": limit} if limit is not None else {})
        posts = [{"id": row.id, "title": row.title, "content": row.content} for row in rows]

    logger.info("Found %d unscored posts%s", len(posts), f" (limit={limit})" if limit else "")
    return posts


def update_sentiments(
    engine: Any,
    scores: list[tuple[int, float]],
) -> int:
    """Batch-update the ``sentiment`` column for a list of post IDs.

    Parameters
    ----------
    engine:
        SQLAlchemy engine instance.
    scores:
        Sequence of ``(post_id, sentiment_score)`` pairs.

    Returns
    -------
    int
        Number of rows updated.
    """
    if not scores:
        return 0

    stmt = text("UPDATE posts SET sentiment = :score WHERE id = :id")
    params = [{"id": post_id, "score": score} for post_id, score in scores]

    with engine.begin() as conn:
        result = conn.execute(stmt, params)

    logger.info("Updated %d / %d post sentiment scores", result.rowcount, len(scores))
    return result.rowcount


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_sentiment_pipeline(
    limit: int | None = None,
    device: str = "cpu",
    batch_size: int = 1,
) -> int:
    """Run the full sentiment-scoring pipeline.

    Steps
    -----
    1. Load the FinBERT model.
    2. Fetch unscored posts from the database.
    3. Prepare text (title + content).
    4. Score in batches.
    5. Persist results back to the database.

    Parameters
    ----------
    limit:
        Maximum number of posts to score (``None`` = all unscored).
    device:
        Device string passed to :func:`load_model`.
    batch_size:
        Batch size for inference.

    Returns
    -------
    int
        Number of posts successfully scored and written to the database.
    """
    # 1. Load model
    classifier = load_model(device=device)

    # 2. Fetch unscored posts
    posts = get_unscored_posts(engine, limit=limit)
    if not posts:
        logger.info("No unscored posts to process.")
        return 0

    # 3. Prepare texts
    prepared: list[tuple[int, str]] = []
    skipped = 0
    for post in posts:
        text_input = _prepare_text(post)
        if text_input is None:
            skipped += 1
            logger.debug("Skipping post %d — no usable text content", post["id"])
            continue
        prepared.append((post["id"], text_input))

    if skipped:
        logger.info("Skipped %d post(s) with empty content", skipped)

    if not prepared:
        logger.info("No posts with usable text after content preparation.")
        return 0

    texts = [t for _, t in prepared]

    # 4. Score
    logger.info("Scoring %d posts (batch_size=%d) ...", len(texts), batch_size)
    scores = score_texts(texts, classifier, batch_size=batch_size)

    # 5. Pair IDs with scores
    id_score_pairs = list(zip([pid for pid, _ in prepared], scores))

    # Log progress periodically
    for i in range(0, len(id_score_pairs), PROGRESS_INTERVAL):
        batch_slice = id_score_pairs[i : i + PROGRESS_INTERVAL]
        vals = [s for _, s in batch_slice]
        avg = sum(vals) / len(vals) if vals else 0.0
        logger.info(
            "Progress: %d / %d scored (current batch avg=%.4f)",
            min(i + PROGRESS_INTERVAL, len(id_score_pairs)),
            len(id_score_pairs),
            avg,
        )

    # 6. Write to database
    updated = update_sentiments(engine, id_score_pairs)
    return updated


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.info("Starting sentiment scoring ...")
    n = run_sentiment_pipeline(limit=None)
    logger.info("Scored %d posts", n)
