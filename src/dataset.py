"""Preparação de dados para treino: split temporal + balanceamento."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.config import settings

TARGET = "is_fraud"
DROP_COLS = ["transaction_id", "timestamp", "ip_address"]
CAT_COLS = ["merchant_category", "merchant_country", "customer_country", "device_id"]


def load_features(path: Path | None = None) -> pl.DataFrame:
    path = path or Path(settings.data_processed_dir) / "features.parquet"
    if not path.exists():
        raise FileNotFoundError("Rode `python -m src.features` antes.")
    return pl.read_parquet(path)


def temporal_split(df: pl.DataFrame, test_size: float = 0.2):
    """Faz o split temporal (treina no passado, testa no futuro)."""
    df = df.sort("timestamp")
    cut = int(len(df) * (1 - test_size))
    return df.head(cut), df.tail(len(df) - cut)


def _encode(df: pl.DataFrame) -> pl.DataFrame:
    """Transforma colunas categóricas em one-hot encoding."""
    encodable = [c for c in CAT_COLS if c in df.columns]
    return df.to_dummies(columns=encodable, drop_first=False)


def to_xy(df: pl.DataFrame):
    """Separa as features (X) do target (y) e converte para numpy."""
    df = _encode(df).drop([c for c in DROP_COLS if c in df.columns])
    y = df[TARGET].to_numpy().astype(np.int8)
    X = df.drop(TARGET).to_pandas().select_dtypes(include=[np.number]).fillna(0.0)
    cols = list(X.columns)
    return X.to_numpy(dtype=np.float32), y, cols


def balance(X: np.ndarray, y: np.ndarray, strategy: str = "smote", seed: int = 42):
    """Aplica SMOTE ou Under-sampling apenas no conjunto de treino."""
    if strategy == "smote":
        sampler = SMOTE(random_state=seed, k_neighbors=5)
    elif strategy == "under":
        sampler = RandomUnderSampler(random_state=seed)
    else:
        return X, y
    return sampler.fit_resample(X, y)
