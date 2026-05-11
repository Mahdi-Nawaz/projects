# 🏥 Hospital Readmission Analysis & Prediction System

**B.Tech CSE (AI/ML) | GNA University, Phagwara | Mahdi Nawaz**

---

## 📌 Project Overview

A complete machine learning system that predicts whether a patient will be
readmitted to hospital within 30 days of discharge. It includes:

- Multi-table dataset generation / integration
- Full data preprocessing & EDA
- Three ML models: Logistic Regression, Random Forest, Gradient Boosting
- Streamlit web application for real-time predictions

---

## ⚙️ Setup Instructions

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the dataset

```bash
python data/generate_data.py
```

### 3. Run the full ML pipeline (training + plots)

```bash
python ml_pipeline.py
```

### 4. Launch the Streamlit web application

```bash
streamlit run app/streamlit_app.py
```

Then open your browser at **http://localhost:8501**

---

## 📂 Project Structure

```
hospital_readmission/
├── data/
│   ├── generate_data.py       ← synthetic dataset generator
│   ├── patients.csv
│   ├── admissions.csv
│   ├── hospitals.csv
│   ├── diagnoses.csv
│   └── billing.csv
├── models/
│   ├── best_model.pkl         ← trained model (Logistic Regression)
│   ├── scaler.pkl             ← StandardScaler
│   └── encoders.pkl           ← LabelEncoders for each categorical
├── plots/
│   ├── 01_eda_overview.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_model_evaluation.png
│   └── 04_feature_importance.png
├── app/
│   └── streamlit_app.py       ← Streamlit web app
├── ml_pipeline.py             ← complete ML pipeline
├── requirements.txt
└── README.md
```

---

## 🧠 Machine Learning Pipeline

| Step | Description |
|------|-------------|
| 1 | Load 5 CSV tables (patients, hospitals, admissions, diagnoses, billing) |
| 2 | Merge using patient_id, admission_id, hospital_id |
| 3 | Feature engineering: comorbidity_count, out_of_pocket |
| 4 | Handle missing values, encode categoricals, scale numerics |
| 5 | EDA – 6 insightful visualizations |
| 6 | Train 3 ML models with class imbalance handling |
| 7 | Evaluate: Accuracy, Precision, Recall, F1, ROC-AUC |
| 8 | Save best model and preprocessing artifacts |

---

## 📊 Model Results

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression ★ | ~0.61 | ~0.55 | ~0.65 |
| Random Forest | ~0.64 | ~0.31 | ~0.63 |
| Gradient Boosting | ~0.63 | ~0.37 | ~0.63 |

---

## 🌐 Streamlit App Features

- **Prediction Page**: Input patient details → get risk score + recommendations
- **Analytics Page**: View all EDA and model evaluation plots
- **About Page**: Project documentation

---

## 📚 References

- Kaggle India Hospital Readmission Dataset
- Scikit-learn, Pandas, Seaborn, Streamlit documentation
