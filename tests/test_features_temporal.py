import polars as pl
import pytest

from src.generate_dataset import generate_dataset
from src.features import add_temporal_windows


@pytest.fixture
def small_df() -> pl.DataFrame:
    pdf = generate_dataset(n_transactions=800, n_customers=80, fraud_ratio=0.02, seed=11)
    return pl.from_pandas(pdf)


def test_temporal_window_columns_exist(small_df):
    df = add_temporal_windows(small_df)
    for label in ("15m", "1h", "24h"):
        assert f"tx_count_{label}" in df.columns
        assert f"amt_sum_{label}" in df.columns
        assert f"amt_avg_{label}" in df.columns


def test_temporal_count_is_at_least_one(small_df):
    df = add_temporal_windows(small_df)
    assert (df["tx_count_15m"] >= 1).all()
