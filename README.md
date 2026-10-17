# GuardShield Fraud Engine & Analytics

Engine de detecção de fraudes financeiras em tempo real e em lote, combinando regras de negócio, Machine Learning supervisionado, Isolation Forest para anomalias e explicabilidade com SHAP.

## Status do desenvolvimento
- [ ] **Dia 1** — Setup, Docker, PostgreSQL, dataset sintético, ETL e validação
- [ ] Dia 2 — Feature engineering, ML supervisionado, Isolation Forest, SHAP
- [ ] Dia 3 — API FastAPI, Dashboard Streamlit, Docker Compose final

## Como rodar
```bash
cp .env.example .env
make up
make generate
make etl
make validate
