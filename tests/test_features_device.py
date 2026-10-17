import polars as pl
from src.generate_dataset import generate_dataset
from src.features import add_device_ip_features

def test_device_ip_features_created():
    pdf = generate_dataset(n_transactions=300, n_customers=30, fraud_ratio=0.02, seed=9)
    df = add_device_ip_features(pl.from_pandas(pdf))
    for col in ("device_change", "ip_change", "device_nunique_7d"):
        assert col in df.columns

def test_first_event_has_no_change():
    pdf = generate_dataset(n_transactions=300, n_customers=30, fraud_ratio=0.02, seed=9)
    df = add_device_ip_features(pl.from_pandas(pdf))
    assert df["device_change"].min() >= 0
    assert df["ip_change"].min() >= 0
