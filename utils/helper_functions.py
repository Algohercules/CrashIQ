"""
CrashIQ Helper Functions
Utility functions for dataset loading, preprocessing, model training, evaluation, and inference.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Severity label mappings
SEVERITY_MAPPING = {
    'Fatal crash': 0,
    'Major injury': 1,
    'Minor injury': 2
}

INV_SEVERITY_MAPPING = {v: k for k, v in SEVERITY_MAPPING.items()}

# Top 5 predictive features identified via Random Forest feature importance
DEFAULT_FEATURE_COLS = [
    'Vehicle_Speed',
    'Crash_Time',
    'Age',
    'Over_Speeding',
    'Lane_Width'
]

# Best hyperparameters tuned via Optuna Bayesian Optimization
BEST_PARAMS = {
    'Random Forest': {
        'n_estimators': 193,
        'max_depth': 7,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'bootstrap': False,
        'random_state': 42
    },
    'XGBoost': {
        'n_estimators': 102,
        'learning_rate': 0.03318,
        'max_depth': 5,
        'min_child_weight': 7.37,
        'subsample': 0.651,
        'colsample_bytree': 0.805,
        'reg_lambda': 0.423,
        'reg_alpha': 0.284,
        'random_state': 42,
        'eval_metric': 'mlogloss'
    },
    'LightGBM': {
        'n_estimators': 60,
        'learning_rate': 0.0426,
        'num_leaves': 18,
        'min_child_samples': 16,
        'reg_alpha': 0.807,
        'reg_lambda': 0.682,
        'colsample_bytree': 0.920,
        'random_state': 42,
        'verbose': -1
    },
    'CatBoost': {
        'iterations': 100,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 6.0,
        'verbose': 0,
        'random_state': 42
    }
}


def find_dataset_path():
    """Locate the dataset file in possible standard locations."""
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', 'data', 'Data_Sheet.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'Data Sheet.csv'),
        'data/Data_Sheet.csv',
        'Data Sheet.csv'
    ]
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    raise FileNotFoundError("Could not find Data Sheet.csv in standard locations.")


def load_and_preprocess_data(filepath=None, include_eda_bins=True):
    """Load dataset, add engineered features, and encode target."""
    if filepath is None:
        filepath = find_dataset_path()
    
    df = pd.read_csv(filepath)
    
    # Feature Engineering
    df['Over_Speeding'] = df['Vehicle_Speed'] - df['Speed_Limit']
    df['Over_Speeding_binary'] = np.where(df['Over_Speeding'] > 0, 1, 0)
    
    if include_eda_bins:
        df['Vehicle_Speed_Range'] = pd.cut(
            df['Vehicle_Speed'],
            bins=[0, 30, 60, 90, 120],
            labels=['0-30', '30-60', '60-90', '90-120'],
            include_lowest=True
        )
        df['Age_Range'] = pd.cut(
            df['Age'],
            bins=[18, 30, 40, 50, 60, 70, 80],
            labels=['18-30', '30-40', '40-50', '50-60', '60-70', '70-80'],
            include_lowest=True
        )
        df['Lane_Width_Range'] = pd.cut(
            df['Lane_Width'],
            bins=[3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
            labels=['3.0-3.1', '3.1-3.2', '3.2-3.3', '3.3-3.4', '3.4-3.5'],
            include_lowest=True
        )
    
    # Encode target severity
    if 'Crash_Severity' in df.columns:
        df['Severity_Encoded'] = df['Crash_Severity'].map(SEVERITY_MAPPING)
        
    return df


def get_train_test_split(df, feature_cols=None, test_size=0.2, random_state=42):
    """Split data into stratified train and test sets."""
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
        
    X = df[feature_cols]
    y = df['Severity_Encoded']
    
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def initialize_model(model_name='Random Forest'):
    """Instantiate a model with its best hyperparameter configuration."""
    params = BEST_PARAMS.get(model_name, {})
    if model_name == 'Random Forest':
        return RandomForestClassifier(**params)
    elif model_name == 'XGBoost':
        return XGBClassifier(**params)
    elif model_name == 'LightGBM':
        return LGBMClassifier(**params)
    elif model_name == 'CatBoost':
        return CatBoostClassifier(**params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def evaluate_model(model, X_test, y_test):
    """Compute accuracy, classification report, and confusion matrix."""
    y_pred = model.predict(X_test)
    if hasattr(y_pred, 'flatten'):
        y_pred = y_pred.flatten()
        
    accuracy = accuracy_score(y_test, y_pred)
    target_names = [INV_SEVERITY_MAPPING[i] for i in sorted(INV_SEVERITY_MAPPING.keys())]
    
    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'report_dict': report_dict,
        'report_str': report_str,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }


def save_model(model, feature_cols=DEFAULT_FEATURE_COLS, filepath='models/best_model.pkl', metadata=None):
    """Serialize the trained model and associated metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    bundle = {
        'model': model,
        'feature_cols': feature_cols,
        'severity_mapping': SEVERITY_MAPPING,
        'inv_severity_mapping': INV_SEVERITY_MAPPING,
        'metadata': metadata or {}
    }
    with open(filepath, 'wb') as f:
        pickle.dump(bundle, f)
    return filepath


def load_model(filepath='models/best_model.pkl'):
    """Load serialized model bundle."""
    if not os.path.exists(filepath):
        candidates = [
            os.path.join(os.path.dirname(__file__), '..', filepath),
            filepath
        ]
        for c in candidates:
            if os.path.exists(c):
                filepath = c
                break
    with open(filepath, 'rb') as f:
        bundle = pickle.load(f)
    return bundle


def predict_severity(model_bundle, vehicle_speed, speed_limit, crash_time, age, lane_width):
    """
    Run prediction on given road and driver parameters.
    
    Returns:
        dict containing:
            - predicted_class: string ('Fatal crash', 'Major injury', 'Minor injury')
            - predicted_class_idx: int
            - probabilities: dict of {class_name: prob}
            - risk_level: 'High' | 'Medium' | 'Low'
            - safety_recommendations: list of actionable policy/safety tips
    """
    model = model_bundle['model']
    feature_cols = model_bundle.get('feature_cols', DEFAULT_FEATURE_COLS)
    inv_mapping = model_bundle.get('inv_severity_mapping', INV_SEVERITY_MAPPING)
    
    over_speeding = vehicle_speed - speed_limit
    
    input_dict = {
        'Vehicle_Speed': [float(vehicle_speed)],
        'Crash_Time': [float(crash_time)],
        'Age': [float(age)],
        'Over_Speeding': [float(over_speeding)],
        'Lane_Width': [float(lane_width)]
    }
    input_df = pd.DataFrame(input_dict)[feature_cols]
    
    pred_idx = int(model.predict(input_df)[0])
    pred_label = inv_mapping.get(pred_idx, 'Unknown')
    
    # Calculate probabilities
    probs = {}
    if hasattr(model, 'predict_proba'):
        prob_values = model.predict_proba(input_df)[0]
        for idx, prob in enumerate(prob_values):
            label = inv_mapping.get(idx, f"Class {idx}")
            probs[label] = float(prob)
    else:
        for idx in inv_mapping:
            label = inv_mapping[idx]
            probs[label] = 1.0 if idx == pred_idx else 0.0

    # Risk level & recommendations
    recommendations = []
    if over_speeding > 20:
        recommendations.append("🚨 Extreme Overspeeding: Deploy automated speed enforcement cameras and rumble strips.")
    elif over_speeding > 0:
        recommendations.append("⚠️ Speeding Alert: Implement dynamic variable speed limit signs (VSL).")
        
    if crash_time in [3, 4, 15, 16]:
        recommendations.append("🕒 Peak Crash Window: Increase high-visibility police patrolling during 3-4 AM & 3-4 PM peak risk hours.")
        
    if lane_width < 3.2:
        recommendations.append("🛣️ Narrow Lane Hazard: Consider lane-widening to 3.5m standard and upgrading side crash barriers.")
        
    if age < 25 or age > 65:
        recommendations.append("👤 Vulnerable Driver Demographic: Enhance high-contrast reflective lane markings and advanced warning signage.")

    fatal_prob = probs.get('Fatal crash', 0.0)
    major_prob = probs.get('Major injury', 0.0)
    
    if pred_label == 'Fatal crash' or fatal_prob > 0.4:
        risk_level = 'Critical'
    elif pred_label == 'Major injury' or (fatal_prob + major_prob) > 0.6:
        risk_level = 'High'
    else:
        risk_level = 'Moderate'
        
    return {
        'predicted_class': pred_label,
        'predicted_class_idx': pred_idx,
        'probabilities': probs,
        'risk_level': risk_level,
        'over_speeding': over_speeding,
        'safety_recommendations': recommendations
    }
