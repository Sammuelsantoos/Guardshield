from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import polars as pl
from src.config import settings

REQUIRED = {
    "transaction_id": pl.Utf8, "customer_id": pl.Utf8, "timestamp": pl.Datetime, 
    "amount": pl.Float64, "merchant_category": pl.Utf8, "merchant_country": pl.Utf8, 
    "customer_country": pl.Utf8, "device_id": pl.Utf8, "ip_address": pl.Utf8, 
    "is_fraud": pl.Int8
}

@dataclass
class ValidationReport:
    path: str
    rows: int = 0
    schema_errors: list[str] = field(default_factory=list)
    null_counts: dict[str, int] = field(default_factory=dict)
    duplicate_ids: int = 0
    non_positive_amounts: int = 0
    fraud_count: int = 0
    fraud_ratio: float = 0.0

    @property
    def is_ok(self) -> bool:
        return (
            not self.schema_errors 
            and self.duplicate_ids == 0 
            and self.non_positive_amounts == 0 
            and all(v == 0 for v in self.null_counts.values())
        )

def validate(path: Path | str | None = None) -> ValidationReport:
    path = Path(path or (Path(settings.data_raw_dir) / "transactions_raw.parquet"))
    df = pl.read_parquet(path)
    report = ValidationReport(path=str(path), rows=df.height)

    for col, dtype in REQUIRED.items():
        if col not in df.columns:
            report.schema_errors.append(f"coluna ausente: {col}")
            continue
        actual = df.schema[col]
        if dtype == pl.Datetime:
            if not str(actual).startswith("Datetime"):
                report.schema_errors.append(f"{col}: esperado Datetime, obtido {actual}")
        elif actual != dtype:
            report.schema_errors.append(f"{col}: esperado {dtype}, obtido {actual}")

    report.null_counts = {c: int(df[c].null_count()) for c in df.columns}
    report.duplicate_ids = int(df.height - df["transaction_id"].n_unique())
    report.non_positive_amounts = int((df["amount"] <= 0).sum())
    report.fraud_count = int(df["is_fraud"].sum())
    report.fraud_ratio = report.fraud_count / df.height if df.height else 0.0
    return report

def print_report(report: ValidationReport) -> None:
    print("=" * 64)
    print(f"Relatório de Validação — {report.path}")
    print("=" * 64)
    print(f"Linhas totais        : {report.rows:,}")
    print(f"Fraudes              : {report.fraud_count:,} ({report.fraud_ratio:.4%})")
    print(f"IDs duplicados       : {report.duplicate_ids}")
    print(f"Amounts <= 0         : {report.non_positive_amounts}")
    print(f"Erros de schema      : {report.schema_errors or 'nenhum'}")
    print(f"Nulos por coluna     : {'(sem nulos)' if all(v == 0 for v in report.null_counts.values()) else ''}")
    
    if any(report.null_counts.values()):
        for col, n in report.null_counts.items():
            if n: print(f"  - {col}: {n}")
            
    print("=" * 64)
    print("STATUS:", "OK ✅" if report.is_ok else "ATENÇÃO ⚠️")

if __name__ == "__main__":
    print_report(validate())
