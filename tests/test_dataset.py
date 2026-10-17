import numpy as np
import polars as pl

from src.dataset import balance, temporal_split, to_xy
from src.generate_dataset import generate_dataset


def _toy():
    """Gera um dataset pequeno para testes rápidos."""
    pdf = generate_dataset(n_transactions=1_000, n_customers=100, fraud_ratio=0.05, seed=2)
    return pl.from_pandas(pdf)


def test_temporal_split_keeps_order():
    df = _toy().sort("timestamp")
    train, test = temporal_split(df)
    # Garante que a data máxima do treino seja menor que a mínima do teste
    assert train["timestamp"].max() <= test["timestamp"].min()


def test_to_xy_shapes():
    df = _toy()
    X, y, cols = to_xy(df)
    assert X.shape[0] == len(y) == df.height
    assert X.shape[1] == len(cols)


def test_balance_increases_minority():
    X = np.random.rand(500, 5).astype("float32")
    y = np.concatenate([np.zeros(490, dtype="int8"), np.ones(10, dtype="int8")])
    Xb, yb = balance(X, y, strategy="smote")
    # O SMOTE deve criar mais exemplos da classe minoritária
    assert yb.sum() > y.sum()
