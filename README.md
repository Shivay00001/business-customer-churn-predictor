# Business Customer Churn Predictor

FastAPI microservice that scores customer churn risk with a RandomForest model.
Built for VisionQuantech client engagements: point it at a client's subscriber
data, retrain, and it serves `P(churn)` with risk levels over a keyed API.

## Endpoints

| Method | Path      | Auth      | Purpose                          |
|--------|-----------|-----------|----------------------------------|
| GET    | `/health` | none      | Liveness + model status → 200    |
| POST   | `/predict`| X-API-Key | Churn probability for one customer |

`POST /predict` body:

```json
{
  "tenure": 2,
  "monthly_charges": 85.0,
  "total_charges": 170.0,
  "support_calls": 5,
  "contract_type": 0
}
```

(`contract_type`: 0 = month-to-month, 1 = one year, 2 = two year)

Response:

```json
{
  "churn_probability": 0.99,
  "risk_level": "High",
  "recommendation": "Proactive outreach"
}
```

All errors are structured JSON (`{"error": "...", "message": "..."}`) — stack
traces are never sent to clients.

## Safety rule (important)

**There is no mock fallback.** If the model artifact is missing or corrupt,
the server refuses to boot (`ModelNotTrainedError` in the logs) and
`/predict` returns HTTP 503. A broken deployment is never mistaken for a
working one. The 8-row CSV in `data/` is a **demo dataset** so fresh clones
boot — always retrain on real client data before paid use (see below).

## Quick start (local)

```bash
cp .env.example .env
# generate a key and paste it into .env as API_KEY:
python -c "import secrets; print(secrets.token_urlsafe(32))"

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train_model.py data/customer_data.csv   # trains -> models/churn_v1.pkl
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Then:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"tenure":2,"monthly_charges":85.0,"total_charges":170.0,"support_calls":5,"contract_type":0}'
```

## Retraining on client data

```bash
python train_model.py /path/to/client_data.csv
```

The CSV must contain the feature columns `tenure, monthly_charges,
total_charges, support_calls, contract_type` and a `churn` label column
(0/1). A `customer_id` column is ignored if present. Set `MODEL_PATH` if the
artifact should live elsewhere.

## Environment variables

| Var              | Required | Default              | Purpose                                            |
|------------------|----------|----------------------|----------------------------------------------------|
| `API_KEY`        | yes      | —                    | Shared secret for `X-API-Key` (min 16 chars); server refuses to boot without it |
| `MODEL_PATH`     | no       | `models/churn_v1.pkl`| Where the trained `.pkl` artifact lives            |
| `ALLOWED_ORIGINS`| no       | (empty)              | Comma-separated CORS origins, e.g. `https://app.example.com`. Empty = same-origin only |
| `PORT`           | no       | `8000`               | Listen port                                        |
| `DEBUG`          | no       | `false`              | `true` enables `/docs` and `/redoc` (dev only)     |

## Docker

```bash
docker build -t churn-predictor .
docker run -p 8000:8000 -e API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" churn-predictor
```

The image bakes in a model trained on the demo dataset; pass a volume with
your own `churn_v1.pkl` and `MODEL_PATH` for client deployments. `API_KEY`
must be supplied at runtime — the container exits otherwise.

## Deploy on Render

1. Push this repo to GitHub, then **New → Web Service** → connect the repo.
2. Runtime: Docker. Render auto-detects the `Dockerfile`.
3. Add environment variables: `API_KEY` (generate a strong one), optionally
   `MODEL_PATH`, `ALLOWED_ORIGINS`, `DEBUG=false`.
4. Deploy. Health check path: `/health`.

## Tests

```bash
# trains the demo model, boots the server, asserts 200/401/422 + loud model-missing boot failure
API_KEY=<your key> ./tests/smoke_test.sh
```

## Project layout

```
src/main.py        FastAPI app (auth, validation, JSON errors, CORS)
src/core/model.py  ChurnModel: train / load / predict — no mock fallback
train_model.py     CLI: train from CSV -> models/churn_v1.pkl
tests/smoke_test.sh Boot + endpoint smoke test (kept in repo)
data/              Demo training CSV (8 rows — retrain on client data)
Dockerfile         Builds with a baked-in demo model; needs API_KEY at runtime
```

## License

MIT — see `LICENSE`.
