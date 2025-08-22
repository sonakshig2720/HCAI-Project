# project2/views.py

import os
import logging
import gzip
import pandas as pd
import numpy as np

# Use a non‑GUI backend for matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from django.conf import settings
from django.shortcuts import render
from sklearn.model_selection import train_test_split

from .representation import build_vectorizer
from .classifier import TextClassifier
from .active_learning import (
    random_sampling,
    uncertainty_sampling,
    margin_sampling,
    entropy_sampling,
    run_pool_al,
    run_batch_pool_al,
)

logger = logging.getLogger(__name__)

# Paths
DATA_DIR   = os.path.join(settings.BASE_DIR, "project2", "data")
CSV_GZ     = os.path.join(DATA_DIR, "imdb.csv.gz")
VEC_PATH   = os.path.join(settings.BASE_DIR, "project2", "models", "vec.pkl")
CLF_PATH   = os.path.join(settings.BASE_DIR, "project2", "models", "clf.pkl")
PLOT_DIR   = os.path.join(settings.BASE_DIR, "static", "plots")
PLOT_PATH  = os.path.join(PLOT_DIR, "al_curve.png")


# ---------------------------
# Helpers
# ---------------------------
def _load_imdb_gz() -> pd.DataFrame:
    """
    Load the IMDB dataset from the committed gzip file only.
    Expected columns: 'review', 'sentiment'.
    """
    if not os.path.exists(CSV_GZ):
        raise RuntimeError(
            "Dataset not found. Expected file at project2/data/imdb.csv.gz.\n"
            "Please add the gzipped CSV to your repo (no Git LFS)."
        )
    try:
        df = pd.read_csv(CSV_GZ, compression="gzip", encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Could not read gzip dataset: {e}")

    # Normalize & sanity‑check
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"review", "sentiment"}.issubset(df.columns):
        raise RuntimeError(
            f"Unexpected columns: {list(df.columns)}. "
            "Expected 'review' and 'sentiment'."
        )
    return df


def _extract_texts_labels(df: pd.DataFrame):
    df.columns = [c.strip().lower() for c in df.columns]
    texts = df["review"].astype(str).fillna("").tolist()

    y = (
        df["sentiment"].astype(str)
        .str.strip()
        .str.lower()
        .map({"positive": 1, "negative": 0})
    )
    if y.isnull().any():
        raise ValueError("Sentiment must contain only 'positive' or 'negative'.")
    labels = y.astype(int).tolist()
    return texts, labels


def _safe_split(texts, labels, test_size=0.3, random_state=42):
    y = np.array(labels)
    if len(set(y)) > 1:
        return train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
    # fallback without stratify if only one class somehow
    return train_test_split(texts, labels, test_size=test_size, random_state=random_state)


def _cap_active(n_pool: int, init_size: int, budget: int, batch_size: int):
    """
    Clamp AL parameters so we never request more samples than available.
    """
    init_size  = max(1, min(init_size, n_pool))
    budget     = max(1, min(budget, max(1, n_pool - init_size)))
    batch_size = max(1, min(batch_size, budget))
    return init_size, budget, batch_size


# ---------------------------
# Main view
# ---------------------------
def index(request):
    """
    - Passive training (70/30 split)
    - Pool-based active learning (single or batch)
    """
    train_accuracy = None
    active_curves  = None
    active_form    = {}
    message        = None
    error          = None

    # 1) Load dataset from gz (only)
    try:
        df = _load_imdb_gz()
        texts, labels = _extract_texts_labels(df)
    except Exception as e:
        return render(request, "project2/index.html", {
            "error": str(e),
            "train_accuracy": None,
            "active_curves": None,
            "active_form": {},
            "message": None,
        })

    # 2) Branches
    if request.method == "POST" and "active_submit" in request.POST:
        # ----- Active Learning -----
        try:
            init_size  = int(request.POST.get("init_size", 10))
            budget     = int(request.POST.get("budget", 50))
            batch_size = int(request.POST.get("batch_size", 1))
            strategy   = request.POST.get("strategy", "random").strip().lower()

            X_pool, X_test, y_pool, y_test = _safe_split(texts, labels, test_size=0.3, random_state=42)
            n_pool = len(X_pool)
            init_size, budget, batch_size = _cap_active(n_pool, init_size, budget, batch_size)

            active_form = {
                "init_size": init_size,
                "budget": budget,
                "batch_size": batch_size,
                "strategy": strategy,
            }

            strat_map = {
                "random":      random_sampling,
                "uncertainty": uncertainty_sampling,
                "margin":      margin_sampling,
                "entropy":     entropy_sampling,
            }
            strat_fn = strat_map.get(strategy, random_sampling)

            if batch_size > 1:
                xs, ys = run_batch_pool_al(
                    pool_texts=X_pool, pool_labels=y_pool,
                    X_test=X_test, y_test=y_test,
                    init_size=init_size, budget=budget,
                    strategy_fn=strat_fn, batch_size=batch_size
                )
            else:
                ys = run_pool_al(
                    pool_texts=X_pool, pool_labels=y_pool,
                    X_test=X_test, y_test=y_test,
                    init_size=init_size, budget=budget,
                    strategy_fn=strat_fn
                )
                xs = list(range(1, len(ys) + 1))

            active_curves = ys

            if xs and active_curves:
                os.makedirs(PLOT_DIR, exist_ok=True)
                plt.figure(figsize=(6.2, 4.3))
                plt.plot(xs, active_curves, marker="o")
                plt.xlabel("Number of labeled examples")
                plt.ylabel("Test accuracy")
                plt.title(f"AL: {strategy.capitalize()} (init={init_size}, batch={batch_size})")
                plt.tight_layout()
                plt.savefig(PLOT_PATH)
                plt.close()

            message = "Active learning run completed."
        except Exception as e:
            logger.exception("Active learning failed")
            error = f"Active learning error: {e}"

    elif request.method == "POST":
        # ----- Passive Training -----
        try:
            X_train, X_test, y_train, y_test = _safe_split(texts, labels, test_size=0.3, random_state=42)
            vec = build_vectorizer()
            clf = TextClassifier(vec)
            clf.fit(X_train, y_train)
            train_accuracy = clf.score(X_test, y_test)

            os.makedirs(os.path.dirname(VEC_PATH), exist_ok=True)
            clf.save(VEC_PATH, CLF_PATH)

            message = (
                f"Training complete. Accuracy: {train_accuracy:.4f}. "
                f"Saved models to: {VEC_PATH} and {CLF_PATH}"
            )
        except Exception as e:
            logger.exception("Training failed")
            error = f"Training error: {e}"

    # 3) Render
    return render(request, "project2/index.html", {
        "train_accuracy": train_accuracy,
        "active_curves": active_curves,
        "active_form": active_form,
        "message": message,
        "error": error,
    })
