import polars as pl
from src.generate_dataset import generate_dataset
from src.features import add_mcc_and_customer_stats

def test_mcc_stats_created():
    pdf = generate_dataset(n_transactions=500, n_customers=50, fraud_ratio=0.02, seed=5)
    df = add_mcc_and_customer_stats(pl.from_pandas(pdf))
    expected = {"customer_mcc_avg", "customer_mcc_std", "customer_avg_amount", "customer_std_amount", "amount_zscore", "amount_vs_mcc_avg"}
    assert expected.issubset(set(df.columns))

def test_zscore_no_nulls():
    pdf = generate_dataset(n_transactions=500, n_customers=50, fraud_ratio=0.02, seed=5)
    df = add_mcc_and_customer_stats(pl.from_pandas(pdf))
    assert df["amount_zscore"].null_count() == 0
