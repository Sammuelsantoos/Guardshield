from __future__ import annotations
import argparse
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import settings

MCC_CODES = ["5411", "5812", "5912", "5541", "5999", "6011", "4814", "5732", "7995", "5967"]
COUNTRIES_LEGIT = ["BR", "US", "AR", "MX", "PT"]
COUNTRIES_FRAUD = ["RU", "NG", "CN", "KP"]

def _random_timestamps(n: int, start: str, end: str, rng: np.random.Generator) -> np.ndarray:
    start_ts = np.datetime64(start)
    end_ts = np.datetime64(end)
    delta = (end_ts - start_ts).astype("timedelta64[s]").astype(np.int64)
    seconds = rng.integers(0, delta, size=n)
    return (start_ts + seconds.astype("timedelta64[s]")).astype("datetime64[s]")

def generate_dataset(n_transactions: int = 200_000, n_customers: int = 5_000, fraud_ratio: float = 0.008, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customer_ids = [f"CUST-{i:06d}" for i in range(n_customers)]
    customer_pool = rng.choice(customer_ids, size=n_transactions, replace=True)
    device_pool = {c: f"DEV-{uuid.uuid4().hex[:10]}" for c in customer_ids}
    timestamps = _random_timestamps(n_transactions, "2024-01-01", "2024-06-30", rng)
    
    # Valores base (distribuição log-normal)
    amounts = np.round(rng.lognormal(mean=3.2, sigma=1.1, size=n_transactions), 2)
    amounts = np.clip(amounts, 1.0, 20_000.0)
    
    merchant_categories = rng.choice(MCC_CODES, size=n_transactions)
    merchant_countries = rng.choice(COUNTRIES_LEGIT + COUNTRIES_FRAUD, size=n_transactions, p=[0.70, 0.05, 0.04, 0.03, 0.03, 0.05, 0.04, 0.03, 0.03])
    customer_countries = np.array(["BR"] * n_transactions)
    
    device_ids = np.array([device_pool[c] for c in customer_pool])
    ip_addresses = [f"{rng.integers(1, 255)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}" for _ in range(n_transactions)]
    
    # Variável alvo (is_fraud)
    is_fraud = (rng.random(n_transactions) < fraud_ratio).astype(np.int8)

    # Injetando padrões característicos nas fraudes
    fraud_idx = np.where(is_fraud == 1)[0]
    if len(fraud_idx):
        amounts[fraud_idx] *= rng.uniform(2.0, 6.0, size=len(fraud_idx))
        merchant_countries[fraud_idx] = rng.choice(COUNTRIES_FRAUD, size=len(fraud_idx), p=[0.4, 0.3, 0.2, 0.1])
        merchant_categories[fraud_idx] = rng.choice(["7995", "5967", "6011"], size=len(fraud_idx))

    transaction_ids = [f"TX-{uuid.uuid4().hex[:16]}" for _ in range(n_transactions)]
    
    df = pd.DataFrame({
        "transaction_id": transaction_ids, 
        "customer_id": customer_pool, 
        "timestamp": timestamps,
        "amount": amounts, 
        "merchant_category": merchant_categories, 
        "merchant_country": merchant_countries,
        "customer_country": customer_countries, 
        "device_id": device_ids, 
        "ip_address": ip_addresses,
        "is_fraud": is_fraud,
    }).sort_values("timestamp").reset_index(drop=True)
    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Gera dataset sintético de fraudes.")
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--customers", type=int, default=5_000)
    parser.add_argument("--fraud-ratio", type=float, default=0.008)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(settings.data_raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transactions_raw.parquet"

    df = generate_dataset(n_transactions=args.n, n_customers=args.customers, fraud_ratio=args.fraud_ratio, seed=args.seed)
    df.to_parquet(out_path, index=False)
    
    frauds = int(df["is_fraud"].sum())
    print(f"[GuardShield] Dataset gerado: {len(df):,} transações | fraudes: {frauds:,} ({frauds / len(df):.3%})")
    print(f"[GuardShield] Arquivo: {out_path}")

if __name__ == "__main__":
    main()
