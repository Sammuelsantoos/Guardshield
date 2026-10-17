"""Treino supervisionado — XGBoost e LightGBM com split temporal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from xgboost import XGBClassifier

from src.config import settings
from src.dataset import balance, load_features, temporal_split, to_xy

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def _metrics(y_true, y_proba, threshold=0.5) -> dict:
    """Calcula as principais métricas de classificação binária."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _build_model(name: str, pos_weight: float):
    """Instancia o modelo com hiperparâmetros padrão."""
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            scale_pos_weight=pos_weight, eval_metric="aucpr",
            tree_method="hist", n_jobs=-1, random_state=42,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
    raise ValueError(f"Modelo desconhecido: {name}")


def train(name: str = "xgboost", strategy: str = "smote") -> dict:
    """Pipeline completo de treinamento e avaliação."""
    df = load_features()
    
    # 1. Split Temporal (Treina no passado, testa no futuro)
    train_df, test_df = temporal_split(df, test_size=0.2)

    # 2. Transformação para Numpy
    X_tr, y_tr, cols = to_xy(train_df)
    X_te, y_te, _ = to_xy(test_df)
    
    # Garantir que as colunas do teste sejam as mesmas do treino
    assert list(cols) == list(to_xy(test_df)[2])

    # 3. Balanceamento (SMOTE) APENAS no treino
    X_tr_b, y_tr_b = balance(X_tr, y_tr, strategy=strategy)

    # 4. Treinamento
    pos_weight = max(1.0, (y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    model = _build_model(name, pos_weight)
    
    print(f"[GuardShield] Treinando {name} com {len(X_tr_b):,} amostras...")
    model.fit(X_tr_b, y_tr_b)

    # 5. Avaliação
    proba = model.predict_proba(X_te)[:, 1]
    metrics = _metrics(y_te, proba)

    # 6. Salvando modelo e relatório
    out = MODEL_DIR / f"{name}.pkl"
    joblib.dump({"model": model, "columns": cols, "name": name}, out)

    report = {"model": name, "strategy": strategy, "metrics": metrics, "path": str(out)}
    Path(MODEL_DIR / f"{name}_report.json").write_text(json.dumps(report, indent=2))
    
    print(f"[GuardShield] {name} -> {metrics}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "lightgbm"])
    parser.add_argument("--strategy", default="smote", choices=["smote", "under", "none"])
    args = parser.parse_args()
    train(args.model, args.strategy)
