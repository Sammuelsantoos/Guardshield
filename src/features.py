"""Feature store do GuardShield — janelas temporais deslizantes via DuckDB."""
from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from src.config import settings

WINDOWS = {"15m": 900, "1h": 3600, "24h": 86400}


def add_temporal_windows(df: pl.DataFrame) -> pl.DataFrame:
    con = duckdb.connect()
    con.register("tx", df.to_arrow())

    selects = []
    for label, seconds in WINDOWS.items():
        selects.append(f"""
            COUNT(*) OVER (
                PARTITION BY customer_id
                ORDER BY timestamp
                RANGE BETWEEN INTERVAL '{seconds} seconds' PRECEDING AND CURRENT ROW
            ) AS tx_count_{label}
        """)
        selects.append(f"""
            COALESCE(SUM(amount) OVER (
                PARTITION BY customer_id
                ORDER BY timestamp
                RANGE BETWEEN INTERVAL '{seconds} seconds' PRECEDING AND CURRENT ROW
            ), 0) AS amt_sum_{label}
        """)
        selects.append(f"""
            COALESCE(AVG(amount) OVER (
                PARTITION BY customer_id
                ORDER BY timestamp
                RANGE BETWEEN INTERVAL '{seconds} seconds' PRECEDING AND CURRENT ROW
            ), 0) AS amt_avg_{label}
        """)

    sql = f"SELECT *, {', '.join(selects)} FROM tx ORDER BY customer_id, timestamp"
    try:
        return con.execute(sql).pl()
    finally:
        con.close()


def build_features(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pl.DataFrame:
    input_path = input_path or Path(settings.data_processed_dir) / "transactions.parquet"
    output_path = output_path or Path(settings.data_processed_dir) / "features.parquet"

    df = pl.read_parquet(input_path)
    df = add_temporal_windows(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    return df


if __name__ == "__main__":
    out = build_features()
    print(f"[GuardShield] Features temporais: {out.shape}")
