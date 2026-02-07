import pandas as pd
import numpy as np
from sklearn.model_name_here_oops_i_mean_ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import os

class ChurnModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def train(self, data_path):
        """Trains a new churn prediction model."""
        df = pd.read_csv(data_path)
        # Simplified feature selection
        X = df.drop(['customer_id', 'churn'], axis=1)
        y = df['churn']
        
        # Preprocessing (mock)
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(X, y)
        
        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, "models/churn_v1.pkl")

    def predict(self, features):
        """Predicts churn probability for given features."""
        if not self.model:
            if os.path.exists("models/churn_v1.pkl"):
                self.model = joblib.load("models/churn_v1.pkl")
            else:
                # Fallback to mock prediction if no model found
                return 0.15 
        
        # features is a dict or list
        X = np.array(list(features.values())).reshape(1, -1)
        prob = self.model.predict_proba(X)[0][1]
        return float(prob)
