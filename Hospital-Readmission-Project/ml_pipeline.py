"""
Hospital Readmission Analysis & Prediction System
=================================================
Full ML Pipeline: Data Integration → Preprocessing → EDA → Model → Evaluation
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             roc_curve, classification_report)
import joblib
import warnings
import os
warnings.filterwarnings("ignore")

# ── Color palette ──────────────────────────────────────────────────────────────
BLUE   = "#2C7BB6"
RED    = "#D7191C"
GREEN  = "#1A9641"
ORANGE = "#FDAE61"
PURPLE = "#7B2D8B"

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – DATA LOADING & INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1: DATA LOADING & INTEGRATION")
print("=" * 65)

patients_df   = pd.read_csv("data/patients.csv")
hospitals_df  = pd.read_csv("data/hospitals.csv")
admissions_df = pd.read_csv("data/admissions.csv")
diagnoses_df  = pd.read_csv("data/diagnoses.csv")
billing_df    = pd.read_csv("data/billing.csv")

print(f"  Patients   : {patients_df.shape}")
print(f"  Hospitals  : {hospitals_df.shape}")
print(f"  Admissions : {admissions_df.shape}")
print(f"  Diagnoses  : {diagnoses_df.shape}")
print(f"  Billing    : {billing_df.shape}")

# Comorbidity count per admission
comorbidity_count = (diagnoses_df.groupby("admission_id")["comorbidity"]
                     .count().reset_index()
                     .rename(columns={"comorbidity": "comorbidity_count"}))

# Merge all tables
df = (admissions_df
      .merge(patients_df,  on="patient_id",  how="left")
      .merge(hospitals_df, on="hospital_id", how="left")
      .merge(billing_df,   on="admission_id", how="left")
      .merge(comorbidity_count, on="admission_id", how="left"))

df["comorbidity_count"] = df["comorbidity_count"].fillna(0).astype(int)
df["out_of_pocket"] = (df["total_charges"] - df["insurance_covered"]).clip(lower=0)

print(f"\n  Merged dataset : {df.shape}")
print(f"  Readmission rate : {df['readmitted_30days'].mean():.1%}\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – DATA PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 2: DATA PREPROCESSING")
print("=" * 65)

# Drop ID columns & low-value columns
drop_cols = ["admission_id", "patient_id", "hospital_id", "state"]
df.drop(columns=drop_cols, inplace=True)

# Missing values
print(f"  Missing values before cleaning:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
df.fillna(df.median(numeric_only=True), inplace=True)
print("  → Filled numeric NaNs with median")

# Encode categoricals
cat_cols = ["gender", "blood_group", "primary_diagnosis", "admission_type",
            "discharge_disposition", "hospital_type", "hospital_size",
            "insurance_type"]
le = LabelEncoder()
encoders = {}
for col in cat_cols:
    df[col] = le.fit(df[col]).transform(df[col])
    encoders[col] = le

print(f"  Encoded {len(cat_cols)} categorical columns")
print(f"  Final dataset shape: {df.shape}\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 65)

# Re-load raw (un-encoded) for EDA plots
raw = (admissions_df if False else None)  # already dropped; load fresh
patients_raw   = pd.read_csv("data/patients.csv")
admissions_raw = pd.read_csv("data/admissions.csv")
hospitals_raw  = pd.read_csv("data/hospitals.csv")
billing_raw    = pd.read_csv("data/billing.csv")
diag_raw       = pd.read_csv("data/diagnoses.csv")

eda = (admissions_raw
       .merge(patients_raw, on="patient_id", how="left")
       .merge(hospitals_raw, on="hospital_id", how="left")
       .merge(billing_raw, on="admission_id", how="left")
       .merge(comorbidity_count, on="admission_id", how="left"))
eda["comorbidity_count"] = eda["comorbidity_count"].fillna(0).astype(int)

# ── Figure 1: Class Distribution ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Hospital Readmission – Exploratory Data Analysis", fontsize=16, fontweight="bold", y=1.01)

counts = eda["readmitted_30days"].value_counts()
bars = axes[0, 0].bar(["Not Readmitted", "Readmitted"], counts.values, color=[GREEN, RED], edgecolor="white", linewidth=1.5)
for bar, v in zip(bars, counts.values):
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30, str(v),
                    ha="center", fontweight="bold", fontsize=12)
axes[0, 0].set_title("Readmission Class Distribution", fontweight="bold")
axes[0, 0].set_ylabel("Count")
axes[0, 0].spines[["top", "right"]].set_visible(False)

# ── Figure 2: Age distribution by readmission ─────────────────────────────────
for label, color, name in zip([0, 1], [GREEN, RED], ["Not Readmitted", "Readmitted"]):
    subset = eda[eda["readmitted_30days"] == label]["age"]
    axes[0, 1].hist(subset, bins=25, alpha=0.6, color=color, label=name, edgecolor="white")
axes[0, 1].set_title("Age Distribution by Readmission", fontweight="bold")
axes[0, 1].set_xlabel("Age")
axes[0, 1].legend()
axes[0, 1].spines[["top", "right"]].set_visible(False)

# ── Figure 3: Readmission rate by diagnosis ───────────────────────────────────
diag_rate = (eda.groupby("primary_diagnosis")["readmitted_30days"]
             .mean().sort_values(ascending=False))
colors_bar = [RED if v > 0.2 else BLUE for v in diag_rate.values]
axes[0, 2].barh(diag_rate.index, diag_rate.values * 100, color=colors_bar)
axes[0, 2].set_title("Readmission Rate by Diagnosis (%)", fontweight="bold")
axes[0, 2].set_xlabel("Readmission Rate (%)")
axes[0, 2].axvline(20, color="gray", linestyle="--", linewidth=1, label="20% threshold")
axes[0, 2].legend()
axes[0, 2].spines[["top", "right"]].set_visible(False)

# ── Figure 4: Length of Stay vs Readmission ───────────────────────────────────
axes[1, 0].boxplot(
    [eda[eda["readmitted_30days"] == 0]["length_of_stay"],
     eda[eda["readmitted_30days"] == 1]["length_of_stay"]],
    labels=["Not Readmitted", "Readmitted"],
    patch_artist=True,
    boxprops=dict(facecolor=BLUE, alpha=0.6)
)
axes[1, 0].set_title("Length of Stay vs Readmission", fontweight="bold")
axes[1, 0].set_ylabel("Days")
axes[1, 0].spines[["top", "right"]].set_visible(False)

# ── Figure 5: Comorbidity count vs Readmission ────────────────────────────────
comorbid_rate = (eda.groupby("comorbidity_count")["readmitted_30days"]
                 .mean() * 100)
axes[1, 1].plot(comorbid_rate.index, comorbid_rate.values,
                marker="o", color=PURPLE, linewidth=2, markersize=7)
axes[1, 1].fill_between(comorbid_rate.index, comorbid_rate.values, alpha=0.15, color=PURPLE)
axes[1, 1].set_title("Readmission Rate by Comorbidity Count (%)", fontweight="bold")
axes[1, 1].set_xlabel("Number of Comorbidities")
axes[1, 1].set_ylabel("Readmission Rate (%)")
axes[1, 1].spines[["top", "right"]].set_visible(False)

# ── Figure 6: Readmission by Admission Type ───────────────────────────────────
adm_rate = (eda.groupby("admission_type")["readmitted_30days"]
            .mean().sort_values(ascending=False))
axes[1, 2].bar(adm_rate.index, adm_rate.values * 100,
               color=[RED, ORANGE, BLUE][:len(adm_rate)], edgecolor="white")
axes[1, 2].set_title("Readmission Rate by Admission Type (%)", fontweight="bold")
axes[1, 2].set_ylabel("Readmission Rate (%)")
axes[1, 2].spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("plots/01_eda_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → plots/01_eda_overview.png")

# ── Figure 7: Correlation Heatmap ─────────────────────────────────────────────
num_cols = ["age", "length_of_stay", "num_procedures", "num_medications",
            "num_previous_admissions", "comorbidity_count", "total_charges", "readmitted_30days"]
corr_data = df[num_cols].copy()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_data.corr(), dtype=bool))
sns.heatmap(corr_data.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, mask=mask, ax=ax, linewidths=0.5)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → plots/02_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – FEATURE ENGINEERING & MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4: MODEL TRAINING")
print("=" * 65)

X = df.drop(columns=["readmitted_30days"])
y = df["readmitted_30days"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42, subsample=0.8),
}

results = {}
for name, model in models.items():
    Xtr = X_train_sc if name == "Logistic Regression" else X_train
    Xte = X_test_sc  if name == "Logistic Regression" else X_test
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]
    results[name] = {
        "model":     model,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "y_pred":    y_pred,
        "y_prob":    y_prob,
    }
    print(f"\n  {name}")
    print(f"    Accuracy  : {results[name]['accuracy']:.4f}")
    print(f"    Precision : {results[name]['precision']:.4f}")
    print(f"    Recall    : {results[name]['recall']:.4f}")
    print(f"    F1-Score  : {results[name]['f1']:.4f}")
    print(f"    ROC-AUC   : {results[name]['roc_auc']:.4f}")

# Best model by ROC-AUC
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best      = results[best_name]
print(f"\n  ★ Best model: {best_name} (ROC-AUC = {best['roc_auc']:.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 – EVALUATION PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 5: EVALUATION & PLOTS")
print("=" * 65)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Evaluation", fontsize=15, fontweight="bold")

# ── (a) Model Comparison Bar Chart ───────────────────────────────────────────
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
x = np.arange(len(metrics))
width = 0.25
palette = [BLUE, GREEN, RED]
for i, (mname, mres) in enumerate(results.items()):
    vals = [mres[m] for m in metrics]
    bars = axes[0].bar(x + i * width, vals, width, label=mname, color=palette[i], alpha=0.85)
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], fontsize=9)
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Model Comparison", fontweight="bold")
axes[0].legend(fontsize=8)
axes[0].spines[["top", "right"]].set_visible(False)

# ── (b) Best model: Confusion Matrix ─────────────────────────────────────────
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["Not Readmitted", "Readmitted"],
            yticklabels=["Not Readmitted", "Readmitted"],
            linewidths=0.5)
axes[1].set_title(f"Confusion Matrix\n({best_name})", fontweight="bold")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

# ── (c) ROC Curves ────────────────────────────────────────────────────────────
for (mname, mres), color in zip(results.items(), palette):
    fpr, tpr, _ = roc_curve(y_test, mres["y_prob"])
    axes[2].plot(fpr, tpr, color=color, linewidth=2,
                 label=f"{mname} (AUC={mres['roc_auc']:.3f})")
axes[2].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].set_title("ROC Curves – All Models", fontweight="bold")
axes[2].legend(fontsize=8)
axes[2].spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("plots/03_model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → plots/03_model_evaluation.png")

# ── Feature Importance (Random Forest) ───────────────────────────────────────
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)
top15 = importances.tail(15)

fig, ax = plt.subplots(figsize=(10, 6))
colors_feat = [RED if v > importances.quantile(0.80) else BLUE for v in top15.values]
top15.plot(kind="barh", ax=ax, color=colors_feat, edgecolor="white")
ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("plots/04_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → plots/04_feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 – SAVE BEST MODEL & ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
best_model_obj = results[best_name]["model"]
joblib.dump(best_model_obj, "models/best_model.pkl")
joblib.dump(scaler,         "models/scaler.pkl")
joblib.dump(encoders,       "models/encoders.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")

print("\n  Saved → models/best_model.pkl")
print("  Saved → models/scaler.pkl")
print("  Saved → models/encoders.pkl")
print("  Saved → models/feature_names.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 – SUMMARY REPORT (text)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
for mname, mres in results.items():
    marker = "★" if mname == best_name else " "
    print(f"  {marker} {mname:<22} | Acc:{mres['accuracy']:.3f} | F1:{mres['f1']:.3f} | AUC:{mres['roc_auc']:.3f}")

print(f"\n  Best Model  : {best_name}")
print(f"  Features    : {X.shape[1]}")
print(f"  Train size  : {len(X_train)}")
print(f"  Test size   : {len(X_test)}")
print("\n  All plots saved in /plots/")
print("  All models saved in /models/")
print("\n✅ Pipeline complete!")
