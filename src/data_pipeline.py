from __future__ import annotations
import argparse
from pathlib import Path
import duckdb
import polars as pl
from sqlalchemy import text
from src.config import settings
from src.database import engine
from src.models import Transaction  # noqa: F401 — garante o registro no metadata

REQUIRED_COLUMNS = [
    "transaction_id", "customer_id", "timestamp", "amount", 
    "merchant_category", "merchant_country", "customer_country", 
    "device_id", "ip_address", "is_fraud"
]

def load_raw(path: Path | None = None) -> pl.DataFrame:
    path = path or Path(settings.data_raw_dir) / "transactions_raw.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset bruto não encontrado em {path}. Rode `python -m src.generate_dataset`."
        )
    return pl.read_parquet(path)

def clean(df: pl.DataFrame) -> pl.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")
        
    df = df.with_columns(
        pl.col("timestamp").cast(pl.Datetime("us")), 
        pl.col("amount").cast(pl.Float64),
        pl.col("is_fraud").cast(pl.Int8), 
        pl.col("merchant_category").cast(pl.Utf8).str.strip_chars(),
        pl.col("merchant_country").cast(pl.Utf8).str.to_uppercase(), 
        pl.col("customer_country").cast(pl.Utf8).str.to_uppercase(),
    )
    
    df = df.filter(
        pl.col("transaction_id").is_not_null() & 
        pl.col("customer_id").is_not_null() & 
        pl.col("timestamp").is_not_null() & 
        (pl.col("amount") > 0)
    )
    
    # Remove duplicatas mantendo a primeira ocorrência
    df = df.unique(subset=["transaction_id"], keep="first")
    
    return df.sort("timestamp")

def save_processed(df: pl.DataFrame) -> Path:
    out_dir = Path(settings.data_processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transactions.parquet"
    df.write_parquet(out_path)
    return out_path

def analytics_summary(path: Path | None = None) -> dict:
    path = path or Path(settings.data_processed_dir) / "transactions.parquet"
    con = duckdb.connect()
    try:
        # Usa o DuckDB para fazer agregações super rápidas no arquivo Parquet
        row = con.execute(f"""
            SELECT 
                COUNT(*) AS total, 
                SUM(is_fraud) AS frauds, 
                AVG(amount) AS avg_amount, 
                SUM(CASE WHEN is_fraud = 1 THEN amount END) AS fraud_amount, 
                COUNT(DISTINCT customer_id) AS customers 
            FROM read_parquet('{path.as_posix()}')
        """).fetchone()
    finally:
        con.close()
    return dict(zip(["total", "frauds", "avg_amount", "fraud_amount", "customers"], row))

def ensure_schema() -> None:
    from src.database import Base
    Base.metadata.create_all(bind=engine)

def load_to_postgres(df: pl.DataFrame) -> int:
    ensure_schema()
    pdf = df.to_pandas()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE transactions RESTART IDENTITY"))
    pdf.to_sql("transactions", engine, if_exists="append", index=False, chunksize=2000)
    return len(pdf)

def run(load_db: bool = False) -> None:
    raw = load_raw()
    cleaned = clean(raw)
    out = save_processed(cleaned)
    print(f"[GuardShield] Processado: {out}")
    print(f"[GuardShield] Resumo: {analytics_summary(out)}")
    if load_db:
        print(f"[GuardShield] Carregado no PostgreSQL: {load_to_postgres(cleaned):,} linhas")

def main() -> None:
    parser = argparse.ArgumentParser(description="ETL do GuardShield.")
    parser.add_argument("--load-db", action="store_true", help="Carrega no PostgreSQL ao final.")
    args = parser.parse_args()
    run(load_db=args.load_db)

if __name__ == "__main__":
    main()
