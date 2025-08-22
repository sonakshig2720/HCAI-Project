# project2/utils_data.py
"""
Utilities to load the IMDB dataset from a gzipped CSV committed in the repo.

Expected path: project2/data/imdb.csv.gz
Columns required: 'review', 'sentiment' where sentiment ∈ {'positive','negative'}.

No network, no extra packages required beyond pandas.
"""

from __future__ import annotations
import os
import gzip
import pandas as pd
from django.conf import settings


DATA_DIR = os.path.join(settings.BASE_DIR, "project2", "data")
CSV_GZ   = os.path.join(DATA_DIR, "imdb.csv.gz")


def load_imdb_dataset() -> pd.DataFrame:
    """
    Load the IMDB dataset from the committed gzip. Raises a clear error
    if missing or malformed.
    """
    if not os.path.exists(CSV_GZ):
        raise RuntimeError(
            "Dataset not found. Expected 'project2/data/imdb.csv.gz' to exist.\n"
            "Please add the gzipped CSV to your repo (no LFS)."
        )

    try:
        df = pd.read_csv(CSV_GZ, compression="gzip", encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Could not read gzip dataset: {e}")

    # normalize & sanity check
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"review", "sentiment"}.issubset(df.columns):
        raise RuntimeError(
            f"Unexpected columns: {list(df.columns)}. "
            "Expected 'review' and 'sentiment'."
        )
    return df
