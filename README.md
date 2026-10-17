# GuardShield Fraud Engine & Analytics

Engine de detecção de fraudes financeiras em tempo real e em lote, combinando regras de negócio, Machine Learning supervisionado, Isolation Forest para anomalias e explicabilidade com SHAP.

![CI](https://github.com/Sammuelsantoos/Guardshield/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Status do desenvolvimento
- [x] **Dia 1** — Setup, Docker, PostgreSQL, dataset sintético, ETL e validação
- [ ] Dia 2 — Feature engineering, ML supervisionado, Isolation Forest, SHAP
- [ ] Dia 3 — API FastAPI, Dashboard Streamlit, Docker Compose final

## Stack Tecnológica
- **Linguagem & Execução:** Python 3.12+, Docker, Docker Compose
- **Ingestão & ETL:** Pandas, Polars, DuckDB, SQLAlchemy
- **Banco de Dados:** PostgreSQL
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM, Imbalanced-learn
- **Explicabilidade (XAI):** SHAP
- **API & Servidor:** FastAPI, Pydantic, Uvicorn
- **Visualização & Dashboard:** Streamlit
- **DevOps & QA:** Pytest, GitHub Actions, Git

## Como rodar o projeto (Dia 1)
```bash
# 1. Suba o banco de dados
cp .env.example .env
make up

# 2. Gere o dataset sintético
make generate

# 3. Rode o ETL e a validação
make etl
make validate

# 4. Rode os testes
make test
