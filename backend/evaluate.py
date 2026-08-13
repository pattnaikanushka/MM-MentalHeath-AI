import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_system():
    print("--- 1. Loading Test Split Holdout Data & Models ---")
    data_path = os.path.join(MODELS_DIR, 'test_eval_data.npz')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run train_models.py first.")
        return

    data = np.load(data_path)
    X_te = data['X_test']
    y_te_c = data['y_test_cls']
    y_te_r = data['y_test_reg']

    cls_path = os.path.join(MODELS_DIR, 'multimodal_fusion_classifier.pkl')
    reg_path = os.path.join(MODELS_DIR, 'multimodal_fusion_regressor.pkl')

    with open(cls_path, 'rb') as f:
        cls_model = pickle.load(f)
    with open(reg_path, 'rb') as f:
        reg_model = pickle.load(f)

    # Classification Evaluation
    cls_preds = cls_model.predict(X_te)
    cls_probas = cls_model.predict_proba(X_te) if hasattr(cls_model, 'predict_proba') else None

    acc = float(accuracy_score(y_te_c, cls_preds))
    macro_f1 = float(f1_score(y_te_c, cls_preds, average='macro'))
    weighted_f1 = float(f1_score(y_te_c, cls_preds, average='weighted'))
    prec_per_class = precision_score(y_te_c, cls_preds, average=None).tolist()
    rec_per_class = recall_score(y_te_c, cls_preds, average=None).tolist()
    f1_per_class = f1_score(y_te_c, cls_preds, average=None).tolist()
    cm = confusion_matrix(y_te_c, cls_preds).tolist()

    # Generate Confusion Matrix Plot Artifact
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Mild', 'Moderate', 'Severe'],
                yticklabels=['Healthy', 'Mild', 'Moderate', 'Severe'])
    plt.title('Multimodal Fusion Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=200)
    plt.close()

    roc_auc = None
    if cls_probas is not None:
        try:
            roc_auc = float(roc_auc_score(y_te_c, cls_probas, multi_class='ovr'))
        except Exception:
            roc_auc = None

    # Separate Per-Target Regression Evaluation
    reg_preds = reg_model.predict(X_te)
    targets = ['Depression_Score', 'Anxiety_Score', 'Stress_Score']
    reg_metrics = {}

    for idx, target in enumerate(targets):
        y_true = y_te_r[:, idx]
        y_pred = reg_preds[:, idx]
        
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))
        exp_var = float(explained_variance_score(y_true, y_pred))

        reg_metrics[target] = {
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2_Score": round(r2, 4),
            "Explained_Variance": round(exp_var, 4)
        }

    evaluation_report = {
        "classification": {
            "headline_metric": "Macro-F1",
            "Macro_F1": round(macro_f1, 4),
            "Accuracy": round(acc, 4),
            "Weighted_F1": round(weighted_f1, 4),
            "ROC_AUC_OVR": round(roc_auc, 4) if roc_auc else "N/A",
            "Precision_Per_Class": [round(p, 4) for p in prec_per_class],
            "Recall_Per_Class": [round(r, 4) for r in rec_per_class],
            "F1_Per_Class": [round(f, 4) for f in f1_per_class],
            "Confusion_Matrix": cm
        },
        "regression_per_target": reg_metrics
    }

    out_file = os.path.join(RESULTS_DIR, 'evaluation_metrics.json')
    with open(out_file, 'w') as f:
        json.dump(evaluation_report, f, indent=2)

    print("\n--- 2. Evaluation Results Summary ---")
    print(f"HEADLINE CLASSIFICATION METRIC (Macro-F1): {macro_f1:.4f}")
    print(f"Accuracy: {acc:.4f} | Weighted-F1: {weighted_f1:.4f} | ROC-AUC: {roc_auc if roc_auc else 'N/A'}")
    print("\nPer-Target Regression Metrics:")
    for target, m in reg_metrics.items():
        print(f" - {target}: RMSE={m['RMSE']} | R²={m['R2_Score']} | MAE={m['MAE']}")

    print(f"\nFull evaluation metrics saved to {out_file}")
    print(f"Confusion matrix image saved to {cm_path}")

if __name__ == "__main__":
    evaluate_system()
