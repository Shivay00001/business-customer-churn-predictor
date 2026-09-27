#!/usr/bin/env python3
"""Train the churn model from a CSV and save the artifact.

Usage:
    python train_model.py [path/to/data.csv]

Env:
    MODEL_PATH  where to write the .pkl artifact (default: models/churn_v1.pkl)

The 8-row CSV in data/ is a DEMO dataset so fresh clones boot. For any paid
client work, retrain on the client's real customer data:
    python train_model.py client_data.csv
Client CSVs must have columns: tenure, monthly_charges, total_charges,
support_calls, contract_type, churn (+ any customer_id column is ignored).
"""

import os
import sys
from pathlib import Path

from src.core.model import ChurnModel

DEFAULT_CSV = "data/customer_data.csv"


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_CSV)
    if not csv_path.exists():
        print(f"ERROR: training CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    model = ChurnModel()
    model.train(str(csv_path))
    print(f"OK: model trained on {csv_path} -> {model.model_path}")


if __name__ == "__main__":
    main()
