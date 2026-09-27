"""Customer Churn Predictor API.

Auth: every non-public endpoint requires the X-API-Key header matching the
API_KEY env var. /health is public. /docs is disabled outside debug mode.
"""

import os
import secrets
from functools import lru_cache

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.model import ChurnModel, ModelNotTrainedError


# ---------------------------------------------------------------- config
def _require_api_key() -> str:
    key = os.environ.get("API_KEY", "")
    if len(key) < 16:
        raise RuntimeError(
            "API_KEY env var is missing or too short (<16 chars). "
            "Generate one (e.g. `python -c 'import secrets; print(secrets.token_urlsafe(32))'`) "
            "and set it. The server refuses to start without a key."
        )
    return key


DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

app = FastAPI(
    title="Customer Churn Predictor API",
    description="RandomForest churn-risk scoring. Auth via X-API-Key header.",
    version="2.0.0",
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None,
    openapi_url="/openapi.json" if DEBUG else None,
)

# CORS: configurable, never wide-open-by-default. Comma-separated origins.
_allowed = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
    allow_credentials=False,
)


# ------------------------------------------------------- structured errors
@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Request body failed validation.",
            "details": exc.errors(),
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "message": exc.detail},
    )


@app.exception_handler(ModelNotTrainedError)
async def model_missing_handler(_: Request, exc: ModelNotTrainedError):
    # 503: the service is up, but it cannot serve predictions right now.
    return JSONResponse(
        status_code=503,
        content={"error": "model_not_trained", "message": str(exc)},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception):
    # Never leak stack traces to clients.
    app.logger.error("Unhandled error: %r", exc) if hasattr(app, "logger") else None
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "An unexpected error occurred."},
    )


# ----------------------------------------------------------------- models
class CustomerData(BaseModel):
    tenure: int = Field(..., ge=0, le=1200, description="Months as customer")
    monthly_charges: float = Field(..., ge=0, le=1_000_000)
    total_charges: float = Field(..., ge=0, le=100_000_000)
    support_calls: int = Field(..., ge=0, le=10_000)
    contract_type: int = Field(..., ge=0, le=2,
                              description="0=month-to-month, 1=one year, 2=two year")


def _check_api_key(x_api_key: str | None) -> None:
    expected = _require_api_key()
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise StarletteHTTPException(status_code=401, detail="Invalid or missing API key.")


@lru_cache
def _predictor() -> ChurnModel:
    return ChurnModel()


@app.on_event("startup")
async def _startup_load_model():
    # Refuse to boot without a valid API key...
    _require_api_key()
    # ...and fail the boot LOUDLY if the model artifact is missing/corrupt,
    # so a broken deploy is never mistaken for a working one.
    _predictor().ensure_loaded()


# --------------------------------------------------------------- endpoints
@app.get("/health")
async def health():
    model = _predictor()
    trained = model.model_path.exists()
    return {"status": "ok", "model_loaded": trained, "model_path": str(model.model_path)}


@app.post("/predict")
async def predict_churn(data: CustomerData, x_api_key: str | None = Header(default=None)):
    _check_api_key(x_api_key)
    features = data.model_dump()
    probability = _predictor().predict_proba(features)
    return {
        "churn_probability": round(probability, 4),
        "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low",
        "recommendation": "Proactive outreach" if probability > 0.5 else "Standard monitoring",
    }
