"""
Model training and inference using LightGBM for Business Entity Resolution.
"""

import os
import joblib
import numpy as np
import lightgbm as lgb
from .features import FEATURE_NAMES

def train_lgbm_model(X_train, y_train, save_path=None):
    """
    Trains a LightGBM Binary Classifier on entity pair features.
    """
    clf = lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=8,
        verbose=-1
    )
    clf.fit(X_train, y_train, feature_name=FEATURE_NAMES)
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        joblib.dump(clf, save_path)
        print(f"Model saved to {save_path}")
        
    return clf

def load_lgbm_model(model_path):
    """Loads a pre-trained LightGBM model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)
