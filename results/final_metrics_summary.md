# Final Model Evaluation Metrics Summary

This document aggregates evaluation metrics across all three data modalities from `results/numeric_metrics.json`, `results/facial_metrics.json`, and `results/speech_metrics.json`. Baseline models were benchmarked against upgraded architectures (XGBoost and Transfer Learning MobileNetV2) and selected based on Macro F1 (classification) and $R^2$ (regression) performance.

---

## 1. Classification Metrics Summary Table

| Modality / Task | Architecture / Model Name | Model Status | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC (Macro) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary (Numeric)** | RandomForestClassifier *(Baseline)* | Evaluated | 0.4875 | 0.5052 | 0.4670 | 0.4762 | 0.7784 |
| **Primary (Numeric)** | XGBClassifier *(Upgrade)* | **Selected** ✅ | **0.5238** | **0.5448** | **0.5199** | **0.5288** | **0.7967** |
| **Facial CNN** | Custom Keras CNN *(Baseline)* | **Selected** (Fallback) ✅ | **0.9000** | **0.2250** | **0.2500** | **0.2368** | N/A |
| **Facial CNN** | MobileNetV2 *(Transfer Learning Upgrade)* | Evaluated | 0.9000 | 0.2250 | 0.2500 | 0.2368 | N/A |
| **Speech Audio** | RandomForestClassifier (MFCCs + Pitch) | **Selected** ✅ | 0.3438 | 0.1617 | 0.2250 | 0.1782 | N/A |

> **Selection Notes (Classification)**:
> - **Numeric Branch**: XGBClassifier clearly outperformed RandomForest baseline on Macro F1 (+0.0526 improvement) and ROC-AUC (+0.0183), so **XGBoost was selected**.
> - **Facial Branch**: MobileNetV2 did not beat the baseline Custom CNN on Macro F1 (0.2368 vs 0.2368 due to ~16:1 class imbalance), so **Baseline Custom Keras CNN was retained as fallback**.

---

## 2. Regression Metrics Summary Table (Primary Modality Multi-Target Severity)

| Modality / Target | Architecture / Model Name | Model Status | MAE | RMSE | $R^2$ Score |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Primary (Numeric Severity)** | MultiOutput(RandomForestRegressor) *(Baseline)* | Evaluated | 8.6666 | 10.8328 | 0.5406 |
| **Primary (Numeric Severity)** | MultiOutput(XGBRegressor) *(Upgrade)* | **Selected** ✅ | **8.2995** | **10.3745** | **0.5786** |

> **Selection Notes (Regression)**:
> - MultiOutput XGBRegressor achieved superior multi-target performance with lower MAE (-0.3671), lower RMSE (-0.4583), and higher average $R^2$ (+0.0380), so **MultiOutput XGBRegressor was selected**.
