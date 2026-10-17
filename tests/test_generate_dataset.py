import polars as pl
import pytest
from src.generate_dataset import generate_dataset

@pytest.fixture
def raw_df() -> pl.DataFrame:
    # Gera um dataset pequeno para testar rápido
    pdf = generate_dataset(n_transactions=2_000, n_customers=200, fraud_ratio=0.01, seed=7)
    return pl.from_pandas(pdf)

def test_generate_dataset_schema(raw_df):
    expected = {"transaction_id", "customer_id", "timestamp", "amount", "merchant_category", "merchant_country", "customer_country", "device_id", "ip_address", "is_fraud"}
    assert expected.issubset(set(raw_df.columns))

def test_generate_dataset_fraud_ratio(raw_df):
    ratio = raw_df["is_fraud"].sum() / raw_df.height
    assert 0 < ratio < 0.05
