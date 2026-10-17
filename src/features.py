"""Feature store do GuardShield — janelas temporais deslizantes via DuckDB."""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from src.config import settings

WINDOWS = {"15m": 900, "1h": 3600, "24h": 86400}

COUNTRY_COORDS: dict[str, tuple[float, float]] = {
    "BR": (-14.2350, -51.9253), "US": (37.0902, -95.7129),
    "AR": (-38.4161, -63.6167), "MX": (23.6345, -102.5528),
    "PT": (39.3999, -8.2245), "RU": (61.5240, 105.3188),
    "NG": (9.0820, 8.6753), "CN": (35.8617, 104.1954),
    "KP": (40.3399, 127.5101),
}


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


def _haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def add_geographic_features(df: pl.DataFrame) -> pl.DataFrame:
    pdf = df.to_pandas()

    def coord(country: str, idx: int):
        c = COUNTRY_COORDS.get(country, COUNTRY_COORDS["BR"])
        return c[idx]

    cust_lat = pdf["customer_country"].map(lambda c: coord(c, 0)).to_numpy()
    cust_lon = pdf["customer_country"].map(lambda c: coord(c, 1)).to_numpy()
    mer_lat = pdf["merchant_country"].map(lambda c: coord(c, 0)).to_numpy()
    mer_lon = pdf["merchant_country"].map(lambda c: coord(c, 1)).to_numpy()

    pdf["distance_km"] = _haversine(cust_lat, cust_lon, mer_lat, mer_lon)
    pdf["is_cross_border"] = (pdf["customer_country"] != pdf["merchant_country"]).astype("int8")

    pdf = pdf.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    pdf["time_since_last_s"] = (
        pdf.groupby("customer_id")["timestamp"].diff().dt.total_seconds().fillna(86_400)
    )
    pdf["velocity_kmh"] = np.where(
        pdf["time_since_last_s"] > 0,
        pdf["distance_km"] / (pdf["time_since_last_s"] / 3600.0),
        0.0,
    )
    pdf["velocity_kmh"] = pdf["velocity_kmh"].clip(upper=20_000)

    return pl.from_pandas(pdf)


def build_features(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pl.DataFrame:
    input_path = input_path or Path(settings.data_processed_dir) / "transactions.parquet"
    output_path = output_path or Path(settings.data_processed_dir) / "features.parquet"

    df = pl.read_parquet(input_path)
    df = add_temporal_windows(df)
    df = add_geographic_features(df) # <-- Nova função adicionada aqui

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    return df


if __name__ == "__main__":
    out = build_features()
    print(f"[GuardShield] Features temporais e geográficas: {out.shape}")
