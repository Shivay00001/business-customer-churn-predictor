FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake a trained model into the image so a fresh container serves
# predictions immediately (trained on the 8-row demo dataset;
# retrain on client data for paid use). The API refuses to start
# without API_KEY set (see .env.example / src/main.py).
RUN python train_model.py data/customer_data.csv

EXPOSE 8000

# API_KEY must be provided at runtime (Render: env vars, or docker -e).
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
