import polars as pl

from src.generate_dataset import generate_dataset
from src.features import add_geographic_features


def test_geo_features_created():
    pdf = generate_dataset(n_transactions=400, n_customers=40, fraud_ratio=0.02, seed=3)
    df = add_geographic_features(pl.from_pandas(pdf))
    for col in ("distance_km", "is_cross_border", "time_since_last_s", "velocity_kmh"):
        assert col in df.columns


def test_cross_border_matches_countries():
    pdf = generate_dataset(n_transactions=400, n_customers=40, fraud_ratio=0.02, seed=3)
    df = add_geographic_features(pl.from_pandas(pdf))
    same = df.filter(pl.col("customer_country") == pl.col("merchant_country"))
    assert (same["is_cross_border"] == 0).all()
