#!/usr/bin/env bash
# Smoke test: boots the API, hits endpoints, asserts expected statuses.
# Run: API_KEY=<your key> ./tests/smoke_test.sh   (trains demo model first)
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-18001}"
export API_KEY="${API_KEY:?Set API_KEY before running (min 16 chars)}"
export MODEL_PATH="${MODEL_PATH:-$REPO/models/churn_v1.pkl}"
BASE="http://127.0.0.1:$PORT"

pass=0; fail=0
check() { # name expected actual
  if [ "$2" = "$3" ]; then echo "PASS: $1 (got $3)"; pass=$((pass+1));
  else echo "FAIL: $1 (expected $2, got $3)"; fail=$((fail+1)); fi
}

echo "== training demo model =="
python3 "$REPO/train_model.py" "$REPO/data/customer_data.csv" >/dev/null
[ -f "$MODEL_PATH" ] || { echo "FAIL: model artifact not created"; exit 1; }

echo "== booting server =="
cd "$REPO"
uvicorn src.main:app --host 127.0.0.1 --port "$PORT" >/tmp/churn-smoke.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 30); do curl -sf "$BASE/health" >/dev/null && break; sleep 0.5; done

echo "== hitting endpoints =="
check "GET /health -> 200" 200 "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")"

# predict without key -> 401
check "POST /predict (no key) -> 401" 401 \
  "$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/predict" \
     -H 'Content-Type: application/json' \
     -d '{"tenure":2,"monthly_charges":85.0,"total_charges":170.0,"support_calls":5,"contract_type":0}')"

# predict with bad key -> 401
check "POST /predict (bad key) -> 401" 401 \
  "$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/predict" \
     -H 'Content-Type: application/json' -H 'X-API-Key: wrong-key' \
     -d '{"tenure":2,"monthly_charges":85.0,"total_charges":170.0,"support_calls":5,"contract_type":0}')"

# predict with valid key, high-risk customer -> 200 + real probability
resp="$(curl -s -w '\n%{http_code}' -X POST "$BASE/predict" \
     -H 'Content-Type: application/json' -H "X-API-Key: $API_KEY" \
     -d '{"tenure":2,"monthly_charges":85.0,"total_charges":170.0,"support_calls":5,"contract_type":0}')"
code="$(echo "$resp" | tail -1)"; body="$(echo "$resp" | head -n -1)"
check "POST /predict (valid) -> 200" 200 "$code"
prob="$(echo "$body" | python3 -c 'import sys,json; print(json.load(sys.stdin)["churn_probability"])')"
echo "  -> churn_probability=$prob"

# invalid input -> 422, JSON error, no stack trace
code="$(curl -s -o /tmp/churn-bad.json -w '%{http_code}' -X POST "$BASE/predict" \
     -H 'Content-Type: application/json' -H "X-API-Key: $API_KEY" \
     -d '{"tenure":-5,"monthly_charges":85.0,"total_charges":170.0,"support_calls":5,"contract_type":9}')"
check "POST /predict (bad input) -> 422" 422 "$code"
grep -q '"error"' /tmp/churn-bad.json && ! grep -qi 'Traceback' /tmp/churn-bad.json \
  && { echo "PASS: structured JSON error, no stack trace"; pass=$((pass+1)); } \
  || { echo "FAIL: error body leaks or is not JSON"; fail=$((fail+1)); }

# model missing at boot -> server refuses to start (loud failure, no silent fallback)
MISSING_PATH="/tmp/churn-nope-$RANDOM/churn_v1.pkl"
MODEL_PATH="$MISSING_PATH" uvicorn src.main:app --host 127.0.0.1 --port "$((PORT+1))" >/tmp/churn-badboot.log 2>&1 &
BADSRV=$!
sleep 10
if kill -0 $BADSRV 2>/dev/null; then
  echo "FAIL: server booted despite missing model"; kill $BADSRV 2>/dev/null; fail=$((fail+1))
else
  grep -qi 'model artifact not found\|ModelNotTrainedError' /tmp/churn-badboot.log \
    && { echo "PASS: missing model -> boot refused loudly (ModelNotTrainedError in logs)"; pass=$((pass+1)); } \
    || { echo "FAIL: bad-boot log lacks ModelNotTrainedError"; fail=$((fail+1)); }
fi

echo "== $pass passed, $fail failed =="
exit $((fail > 0))
