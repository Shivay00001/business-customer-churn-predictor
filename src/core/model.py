"""Churn prediction model wrapper.

Hard rule: there is NO mock/silent fallback. If the model artifact is missing
or unreadable, every prediction fails loudly via ModelNotTrainedError so a
broken deployment is never mistaken for a working one.
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class ModelNotTrainedError(RuntimeError):
    """Raised when the model artifact is missing or cannot be loaded."""


MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/churn_v1.pkl"))


class ChurnModel:
    EXPECTED_FEATURES = [
        "tenure",
        "monthly_charges",
        "total_charges",
        "support_calls",
        "contract_type",
    ]

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None

    # -- training ---------------------------------------------------------
    def train(self, data_path: str, n_estimators: int = 100) -> None:
        df = pd.read_csv(data_path)
        missing = {"customer_id", "churn"} - set(df.columns)
        if missing:
            raise ValueError(f"Training CSV missing columns: {sorted(missing)}")
        missing_feats = set(self.EXPECTED_FEATURES) - set(df.columns)
        if missing_feats:
            raise ValueError(f"Training CSV missing feature columns: {sorted(missing_feats)}")

        X = df[self.EXPECTED_FEATURES]
        y = df["churn"]

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X, y)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        self.model = model

    # -- serving ----------------------------------------------------------
    def _load(self):
        if not self.model_path.exists():
            raise ModelNotTrainedError(
                f"Model artifact not found at '{self.model_path}'. "
                "Train the model first (python train_model.py) or check MODEL_PATH. "
                "Refusing to return a placeholder prediction."
            )
        try:
            return joblib.load(self.model_path)
        except Exception as exc:  # corrupted artifact is as bad as missing
            raise ModelNotTrainedError(
                f"Could not load model artifact '{self.model_path}': {exc}. "
                "Retrain the model (python train_model.py)."
            ) from exc

    def ensure_loaded(self):
        """Load the model now. Raises ModelNotTrainedError if unavailable."""
        if self.model is None:
            self.model = self._load()
        return self.model

    def predict_proba(self, features: dict) -> float:
        """features: dict with exactly EXPECTED_FEATURES keys. Returns P(churn)."""
        self.ensure_loaded()

        expected_n = getattr(self.model, "n_features_in_", len(self.EXPECTED_FEATURES))
        if len(features) != expected_n:
            raise ValueError(
                f"Expected {expected_n} features, got {len(features)}."
            )

        X = np.array([features[k] for k in self.EXPECTED_FEATURES],
                     dtype=float).reshape(1, -1)
        proba = self.model.predict_proba(X)[0]
        return float(proba[1])
