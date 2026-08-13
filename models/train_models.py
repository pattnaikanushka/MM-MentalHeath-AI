import os
import sys
import json
import numpy as np
import pickle
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

def log(msg):
    print(msg)
    sys.stdout.flush()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

log("--- 1. Training Calibrated Multimodal Models with StandardScaler ---")

NUMERIC_NAMES = [
    'Age', 'Sleep_Duration', 'Physical_Activity', 'Work_Stress', 'Social_Support',
    'Screen_Time', 'Caffeine_Intake', 'Alcohol_Units', 'Financial_Stress', 'Depression_History',
    'Anxiety_History', 'Daily_Steps', 'Heart_Rate_Var', 'Cortisol_Index', 'Cognitive_Fatigue',
    'Emotional_Reg', 'Life_Satisfaction', 'Mood_Swings'
]
ALL_FEATURE_NAMES = NUMERIC_NAMES + [
    'Facial_Tension_Idx', 'Facial_Eye_Aspect_Ratio', 'Facial_Brow_Furrow', 'Facial_Mouth_Curvature',
    'Speech_Pitch_Variance', 'Speech_Energy_RMS', 'Speech_Zero_Crossing_Rate'
]

N_SAMPLES = 4000
X_all = np.zeros((N_SAMPLES, 25))

# Feature distributions matching realistic bounds
X_all[:, 0] = np.random.uniform(18, 65, N_SAMPLES)
X_all[:, 1] = np.random.uniform(4, 10, N_SAMPLES)
X_all[:, 2] = np.random.uniform(0, 120, N_SAMPLES)
X_all[:, 3] = np.random.uniform(1, 10, N_SAMPLES)
X_all[:, 4] = np.random.uniform(1, 10, N_SAMPLES)
X_all[:, 5] = np.random.uniform(1, 12, N_SAMPLES)
X_all[:, 6] = np.random.uniform(0, 500, N_SAMPLES)
X_all[:, 7] = np.random.uniform(0, 20, N_SAMPLES)
X_all[:, 8] = np.random.uniform(1, 10, N_SAMPLES)
X_all[:, 9] = np.random.randint(0, 6, N_SAMPLES)
X_all[:, 10] = np.random.randint(0, 6, N_SAMPLES)
X_all[:, 11] = np.random.uniform(2000, 15000, N_SAMPLES)
X_all[:, 12] = np.random.uniform(25, 90, N_SAMPLES)
X_all[:, 13] = np.random.uniform(5, 25, N_SAMPLES)
X_all[:, 14] = np.random.uniform(1, 10, N_SAMPLES)
X_all[:, 15] = np.random.uniform(1, 10, N_SAMPLES)
X_all[:, 16] = np.random.uniform(1, 10, N_SAMPLES)
X_all[:, 17] = np.random.randint(0, 10, N_SAMPLES)
X_all[:, 18] = np.random.uniform(10, 90, N_SAMPLES)
X_all[:, 19] = np.random.uniform(0.15, 0.35, N_SAMPLES)
X_all[:, 20] = np.random.uniform(0.0, 1.0, N_SAMPLES)
X_all[:, 21] = np.random.uniform(-0.5, 0.5, N_SAMPLES)
X_all[:, 22] = np.random.uniform(20, 150, N_SAMPLES)
X_all[:, 23] = np.random.uniform(0.01, 0.5, N_SAMPLES)
X_all[:, 24] = np.random.uniform(0.02, 0.3, N_SAMPLES)

# Standardize Features via StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# Composite stress index normalized between -2 and +2
composite = (
    0.25 * X_scaled[:, 3] + 0.20 * X_scaled[:, 8] + 0.20 * X_scaled[:, 18] + 
    0.15 * X_scaled[:, 22] + 0.15 * X_scaled[:, 13] - 0.20 * X_scaled[:, 4] + np.random.normal(0, 0.5, N_SAMPLES)
)

# Imbalanced label mapping (Severe_Stress as minority class)
y_cls = np.digitize(composite, bins=[-0.6, 0.3, 1.1])
y_dep = np.clip(12 * composite + 42 + np.random.normal(0, 2.5, N_SAMPLES), 15, 90)
y_anx = np.clip(14 * composite + 55 + np.random.normal(0, 2.5, N_SAMPLES), 15, 92)
y_str = np.clip(15 * composite + 65 + np.random.normal(0, 2.5, N_SAMPLES), 15, 95)
y_reg = np.column_stack([y_dep, y_anx, y_str])

# Stratified Train/Val/Test Split (80/10/10)
X_tr, X_temp, y_tr_c, y_temp_c, y_tr_r, y_temp_r = train_test_split(
    X_scaled, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
)
X_val, X_te, y_val_c, y_te_c, y_val_r, y_te_r = train_test_split(
    X_temp, y_temp_c, y_temp_r, test_size=0.5, random_state=42, stratify=y_temp_c
)

# Compute Sample Weights for Class Imbalance
cls_counts = np.bincount(y_tr_c)
cls_weights = {i: float(len(y_tr_c) / (len(cls_counts) * cls_counts[i])) for i in range(len(cls_counts))}
sample_weights_tr = np.array([cls_weights[label] for label in y_tr_c])

# Train Base XGBoost Classifier with Sample Weighting
base_cls = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, eval_metric='mlogloss', random_state=42)
base_cls.fit(X_tr, y_tr_c, sample_weight=sample_weights_tr)

# Model Calibration using CalibratedClassifierCV
calibrated_cls = CalibratedClassifierCV(estimator=base_cls, method='sigmoid', cv=3)
calibrated_cls.fit(X_tr, y_tr_c)

# Train MultiOutput XGBRegressor
jitter_X_tr = X_tr + np.random.normal(0, 0.05, X_tr.shape)
fusion_reg = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42))
fusion_reg.fit(jitter_X_tr, y_tr_r)

# Evaluate Validation RMSE Error Bands for Severity Scores
val_reg_preds = fusion_reg.predict(X_val)
rmse_dep = float(np.sqrt(mean_squared_error(y_val_r[:, 0], val_reg_preds[:, 0])))
rmse_anx = float(np.sqrt(mean_squared_error(y_val_r[:, 1], val_reg_preds[:, 1])))
rmse_str = float(np.sqrt(mean_squared_error(y_val_r[:, 2], val_reg_preds[:, 2])))

regression_rmse = {
    "rmse_depression": round(rmse_dep, 1),
    "rmse_anxiety": round(rmse_anx, 1),
    "rmse_stress": round(rmse_str, 1)
}

# Save Scaler and Models
with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODELS_DIR, 'multimodal_fusion_classifier.pkl'), 'wb') as f:
    pickle.dump(calibrated_cls, f)

with open(os.path.join(MODELS_DIR, 'multimodal_fusion_regressor.pkl'), 'wb') as f:
    pickle.dump(fusion_reg, f)

with open(os.path.join(MODELS_DIR, 'regression_rmse.json'), 'w') as f:
    json.dump(regression_rmse, f, indent=2)

# Save Background Training Matrix
bg_samples = X_tr[np.random.choice(X_tr.shape[0], 150, replace=False)]
np.save(os.path.join(MODELS_DIR, 'bg_train_samples.npy'), bg_samples)

# Save Test Evaluation Holdout Split Data
np.savez(
    os.path.join(MODELS_DIR, 'test_eval_data.npz'),
    X_test=X_te, y_test_cls=y_te_c, y_test_reg=y_te_r
)

log("Model retraining with StandardScaler complete. All model artifacts saved successfully.")
