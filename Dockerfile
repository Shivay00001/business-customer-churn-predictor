FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Training typically happens outside or as a separate step
# RUN python -c "from src.core.model import ChurnModel; ChurnModel().train('data/customer_data.csv')"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]