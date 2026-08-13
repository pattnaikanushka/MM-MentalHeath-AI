# 🧠 MM-MentalHealth AI
### Explainable Multimodal Psychiatric Assessment System

> **⚠️ Important Disclaimer:** This is a computational decision-support prototype — **not a clinical diagnostic tool**. All outputs are research estimates and must not replace professional psychiatric evaluation.

---

## 📌 Project Description

MM-MentalHealth AI is a real-time, explainable AI system that performs multimodal mental health assessment by fusing signals from:

- **Facial Analysis** — Emotion detection via a Keras CNN with Grad-CAM visualization
- **Speech Analysis** — Acoustic features (MFCCs, pitch, energy) via a trained Random Forest
- **Behavioral Signals** — Sleep quality, social engagement, app usage patterns, idle time
- **Physiological Signals** — Heart rate and HRV estimated via remote photoplethysmography (rPPG) from the webcam

The system classifies participants into one of four mental health status categories (Healthy / Mild Stress / Moderate Stress / Severe Stress) and predicts continuous severity scores for Depression, Anxiety, and Stress — all with SHAP-based feature importance and RMSE error bands.

---

## ✨ Features

- 🎯 **4-Class Mental Health Classification** — XGBoost-powered multimodal fusion classifier
- 📊 **3-Target Severity Regression** — Depression, Anxiety, and Stress scores with RMSE error bands
- 🔍 **SHAP Explainability** — Top-10 contributing features visualized as horizontal bar charts
- 📷 **Live rPPG Heart Rate Estimation** — Non-contact heart rate & HRV via webcam green-channel analysis
- 🧬 **Modality Contribution Donut Chart** — Facial (34%), Behavioral (28%), Speech (26%), Physiological (3.2%)
- 📋 **Clinical Report Export** — Downloadable Markdown clinical summary with full metrics
- ⏱️ **Live Dashboard** — Real-time animated indicators, sparklines, and trend charts
- 🌐 **Browser-based Webcam** — Optional hardware camera & microphone capture via HTML5 API

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / Dashboard** | Streamlit + HTML5 + CSS3 + Canvas JS |
| **ML Models** | XGBoost, Scikit-learn (RandomForest, StandardScaler) |
| **Deep Learning** | TensorFlow / Keras (Facial CNN, 1D rPPG signal processing) |
| **Explainability** | SHAP (feature importance, background matrix) |
| **Computer Vision** | OpenCV |
| **Audio Processing** | Librosa, Soundfile |
| **Visualization** | Matplotlib, Seaborn |
| **Testing** | Playwright (E2E browser testing) |

---

## 📁 Project Structure

```
mm_mentalhealth_ai/
│
├── app/
│   └── app.py                          # Main Streamlit dashboard
│
├── backend/
│   ├── __init__.py
│   ├── engine.py                       # MultimodalAnalyticsEngine + rPPG processor
│   └── evaluate.py                     # Model evaluation & confusion matrix generation
│
├── models/
│   ├── train_models.py                 # Full model training pipeline (re-run if needed)
│   ├── multimodal_fusion_classifier.pkl # Primary XGBoost classifier (25 features)
│   ├── multimodal_fusion_regressor.pkl  # XGBoost multi-target regressor
│   ├── scaler.pkl                      # StandardScaler fitted on training data
│   ├── bg_train_samples.npy            # 150-sample background matrix for SHAP
│   ├── facial_cnn.h5                   # Keras CNN for facial emotion classification
│   ├── speech_model.pkl                # Random Forest for speech emotion
│   ├── numeric_classifier.pkl          # Numeric-only baseline classifier
│   ├── numeric_regressor.pkl           # Numeric-only baseline regressor
│   ├── feature_names.json              # 18 numeric feature names
│   ├── all_feature_names.json          # 25 multimodal feature names
│   ├── regression_rmse.json            # Stored RMSE error bands
│   └── test_eval_data.npz              # Held-out test split for evaluation
│
├── results/
│   ├── evaluation_metrics.json         # Full classification + regression metrics
│   ├── final_metrics_summary.md        # Model benchmark comparison table
│   ├── confusion_matrix.png            # Confusion matrix heatmap
│   ├── calibrated_rppg_dashboard.png   # Dashboard screenshot (from E2E test)
│   └── clinical_report_P001.md         # Example exported clinical report
│
├── sample_inputs/
│   ├── sample_face.jpg                 # Test image for facial branch
│   └── sample_speech.wav              # Test audio for speech branch
│
├── test_app.py                         # Playwright E2E automated test
├── requirements.txt                    # All Python dependencies
├── .env.example                        # Environment variable template
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

Make sure you have the following installed on your system:

| Requirement | Version | Download |
|---|---|---|
| **Python** | 3.10 or 3.11 | https://python.org/downloads |
| **pip** | Latest | Included with Python |
| **Git** | Any | https://git-scm.com (optional) |

> **Note on Python version:** Python 3.11 is recommended. TensorFlow has limited support on Python 3.12+.

---

### Step 1 — Clone or Extract the Project

**Option A — If you received a ZIP file:**
```bash
unzip mm_mentalhealth_ai.zip
cd mm_mentalhealth_ai
```

**Option B — If you cloned from Git:**
```bash
git clone <your-repo-url>
cd mm_mentalhealth_ai
```

---

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Create the virtual environment
python -m venv venv

# Activate it:
# On macOS / Linux:
source venv/bin/activate

# On Windows (Command Prompt):
venv\Scripts\activate.bat

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `streamlit`, `scikit-learn`, `xgboost`, `tensorflow`, `numpy`, `pandas`, `opencv-python`, `librosa`, `soundfile`, `matplotlib`, `seaborn`, `shap`, and `Pillow`.

> **⏱️ Expected time:** 3–8 minutes depending on your internet speed (TensorFlow is ~500MB).

---

### Step 4 — Environment Variables (Optional)

This app runs **fully offline** with no API keys needed. All ML models are bundled in the `models/` directory.

If you want to customize server settings:
```bash
cp .env.example .env
# Edit .env with your preferred settings (optional)
```

---

## 🚀 Running the Application

```bash
streamlit run app/app.py
```

The app will automatically open in your browser at:
```
http://localhost:8501
```

> **If it doesn't open automatically**, manually visit `http://localhost:8501` in your browser.

---

## 📊 Running Model Evaluation

To regenerate evaluation metrics and the confusion matrix plot:

```bash
python backend/evaluate.py
```

Output files saved to `results/`:
- `evaluation_metrics.json` — Full classification + regression metrics
- `confusion_matrix.png` — Confusion matrix heatmap

---

## 🔁 Retraining Models (Advanced)

If you want to retrain all models from scratch (generates new `.pkl` and `.h5` files):

```bash
python models/train_models.py
```

> **⚠️ Warning:** Retraining regenerates all model files using synthetic data with random seeds. The retrained models may produce slightly different metrics than the pre-trained versions bundled in `models/`.

---

## 🧪 Running E2E Tests

The `test_app.py` uses Playwright to launch a headless browser, navigate to the running Streamlit app, click the Export Report button, and capture a screenshot.

**Step 1 — Install Playwright browsers (one-time setup):**
```bash
playwright install chromium
```

**Step 2 — Make sure the app is already running (in a separate terminal):**
```bash
streamlit run app/app.py
```

**Step 3 — Run the test:**
```bash
python test_app.py
```

Output screenshot saved to `results/calibrated_rppg_dashboard.png`.

---

## 🏗️ Build & Deploy

### Deploy to Streamlit Community Cloud (Free)

1. Push your project to a **public GitHub repository**.
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **"New app"** and select your repository.
4. Set **Main file path** to: `app/app.py`
5. Click **Deploy**.

> **Note:** The `models/` directory (~10 MB of `.pkl` files) must be included in the repository for the app to load correctly.

### Deploy Locally as a Persistent Server

```bash
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 🔬 Model Performance Summary

| Task | Model | Metric | Score |
|---|---|---|---|
| 4-Class Classification | XGBoost (Multimodal Fusion) | Macro F1 | 0.356 |
| 4-Class Classification | XGBoost (Multimodal Fusion) | ROC-AUC (OVR) | 0.751 |
| Depression Score | XGBoost Regressor | RMSE | 6.76 |
| Anxiety Score | XGBoost Regressor | RMSE | 7.68 |
| Stress Score | XGBoost Regressor | RMSE | 8.43 |

> See `results/final_metrics_summary.md` for the full benchmark comparison table and per-class breakdown.

---

## 🚨 Troubleshooting

### `ModuleNotFoundError: No module named 'backend'`
Run the app from the **project root directory** (the folder containing `app/` and `backend/`):
```bash
cd mm_mentalhealth_ai    # make sure you're in the root
streamlit run app/app.py
```

### `tensorflow` install fails or is very slow
TensorFlow is large (~500MB). Make sure you have a stable internet connection. On Apple Silicon (M1/M2), use:
```bash
pip install tensorflow-macos tensorflow-metal
```

### `streamlit: command not found`
Your virtual environment is not activated. Run:
```bash
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate.bat   # Windows
```
Then retry `streamlit run app/app.py`.

### `Error: Keras model file not found` or `pickle.load` error
Make sure all files in `models/` are present. If you extracted from a ZIP, check no files were skipped. The critical files are:
- `models/multimodal_fusion_classifier.pkl`
- `models/multimodal_fusion_regressor.pkl`
- `models/scaler.pkl`

### Port 8501 already in use
Either close the other Streamlit instance, or run on a different port:
```bash
streamlit run app/app.py --server.port 8502
```

### Webcam/microphone not working
Click the **"🔴 Enable Hardware Camera & Mic"** button inside the dashboard. Your browser will prompt for permission — click **Allow**. The app works without a camera (it falls back to a simulated rPPG signal).

### `playwright install` fails
Playwright browsers are only needed for the automated E2E test (`test_app.py`). The main app runs fine without them.

---

## ⚠️ Clinical & Ethical Limitations

1. **Not a clinical diagnostic tool.** All outputs are computational estimates for research/demonstration purposes only.
2. **Modalities are not participant-linked.** The facial, speech, and behavioral datasets were collected from separate cohorts — scores across modalities must not be averaged or merged.
3. **rPPG is non-clinical.** Heart Rate and HRV values are derived from webcam green-channel analysis and are approximations, not medical-grade readings.
4. **Skin Temperature and GSR are estimated.** These values are simulated from learned statistical distributions, not measured by actual sensors.
5. **Facial dataset class imbalance (~16:1).** Minority high-stress class sensitivity is constrained by severe imbalance in the training data.

---

## 📄 License

This project was created for hackathon/research purposes. See individual library licenses for third-party dependencies.

---

## 👥 Authors

Built with MM-MentalHealth AI multimodal psychiatric assessment framework.

---

*For questions about model architecture or feature engineering, see `results/final_metrics_summary.md` and `models/train_models.py`.*
