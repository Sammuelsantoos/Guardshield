from pathlib import Path
import polars as pl
from src.generate_dataset import generate_dataset
from src.validate import validate

def _write_sample(tmp_path: Path) -> Path:
    pdf = generate_dataset(n_transactions=500, n_customers=50, fraud_ratio=0.02, seed=1)
    path = tmp_path / "tx.parquet"
    pl.from_pandas(pdf).write_parquet(path)
    return path

def test_validate_ok(tmp_path):
    report = validate(_write_sample(tmp_path))
    assert report.rows == 500 
    assert report.duplicate_ids == 0 
    assert report.non_positive_amounts == 0 
    assert report.schema_errors == []
    assert report.is_ok

def test_validate_detects_duplicates(tmp_path):
    path = _write_sample(tmp_path)
    df = pl.read_parquet(path)
    # Adiciona 5 linhas duplicadas
    pl.concat([df, df.head(5)]).write_parquet(path)
    report = validate(path)
    assert report.duplicate_ids == 5 
    assert not report.is_ok
