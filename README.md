# Business Customer Churn Predictor

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade predictive analytics platform** for identifying and mitigating customer churn. Built with Scikit-Learn and FastAPI, this repository provides a complete pipeline for training machine learning models on customer behavior data and serving real-time churn probability scores via a REST API.

## 🚀 Features

- **Predictive Modeling**: Optimized pipelines using Random Forest and Logistic Regression for high-accuracy churn prediction.
- **Real-time Inference**: High-performance FastAPI server for serving model predictions with low latency.
- **Feature Engineering**: Automated preprocessing for customer demographic and behavioral features (scaling, encoding).
- **Explainability**: Integrated SHAP/feature importance analysis to understand key churn drivers.
- **Model Versioning**: Robust mechanism for loading and serving specific model versions.
- **Containerized**: Scalable Docker deployment for production environments.

## 📁 Project Structure

```
business-customer-churn-predictor/
├── src/
│   ├── api/          # FastAPI route handlers
│   ├── core/         # ML model and preprocessing logic
│   └── main.py       # Application entrypoint
├── data/             # Sample customer datasets (CSV)
├── tests/            # Unit and model validation tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🛠️ Quick Start

```bash
# Clone
git clone https://github.com/Shivay00001/business-customer-churn-predictor.git

# Install
pip install -r requirements.txt

# Run API
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## 📄 License

MIT License
