import polars as pl
import pytest
from src.data_pipeline import clean
from src.generate_dataset import generate_dataset

@pytest.fixture
def raw_df() -> pl.DataFrame:
    pdf = generate_dataset(n_transactions=2_000, n_customers=200, fraud_ratio=0.01, seed=7)
    return pl.from_pandas(pdf)

def test_clean_removes_duplicates(raw_df):
    # Duplica as primeiras 50 linhas
    df = pl.concat([raw_df, raw_df.head(50)])
    cleaned = clean(df)
    assert cleaned["transaction_id"].n_unique() == cleaned.height

def test_clean_removes_non_positive_amounts(raw_df):
    # Força 5 valores negativos
    df = raw_df.with_columns(
        pl.when(pl.int_range(pl.len()) < 5)
        .then(pl.lit(-10.0))
        .otherwise(pl.col("amount"))
        .alias("amount")
    )
    cleaned = clean(df)
    assert (cleaned["amount"] > 0).all()

def test_clean_raises_on_missing_columns():
    bad = pl.DataFrame({"transaction_id": ["a"]})
    with pytest.raises(ValueError):
        clean(bad)
