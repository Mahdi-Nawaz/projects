"""
Synthetic Hospital Readmission Dataset Generator
Mimics the Kaggle India Hospital Readmission Dataset structure
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 5000  # number of records

# ─── 1. Patients Table ────────────────────────────────────────────────────────
patient_ids = [f"P{str(i).zfill(5)}" for i in range(1, N + 1)]
genders = np.random.choice(["Male", "Female"], N, p=[0.52, 0.48])
ages = np.random.randint(18, 90, N)
blood_groups = np.random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], N)
states = np.random.choice(
    ["Punjab", "Maharashtra", "Tamil Nadu", "Uttar Pradesh", "Karnataka",
     "Rajasthan", "Gujarat", "West Bengal"], N
)

patients_df = pd.DataFrame({
    "patient_id": patient_ids,
    "age": ages,
    "gender": genders,
    "blood_group": blood_groups,
    "state": states
})

# ─── 2. Hospitals Table ───────────────────────────────────────────────────────
hospital_ids = [f"H{str(i).zfill(3)}" for i in range(1, 51)]
hospital_types = np.random.choice(["Government", "Private", "Trust"], 50, p=[0.4, 0.45, 0.15])
hospital_sizes = np.random.choice(["Small", "Medium", "Large"], 50, p=[0.3, 0.4, 0.3])
beds = np.where(hospital_sizes == "Small", np.random.randint(20, 100, 50),
        np.where(hospital_sizes == "Medium", np.random.randint(100, 300, 50),
                 np.random.randint(300, 800, 50)))

hospitals_df = pd.DataFrame({
    "hospital_id": hospital_ids,
    "hospital_type": hospital_types,
    "hospital_size": hospital_sizes,
    "num_beds": beds
})

# ─── 3. Admissions Table ──────────────────────────────────────────────────────
admission_ids = [f"A{str(i).zfill(6)}" for i in range(1, N + 1)]
assigned_hospitals = np.random.choice(hospital_ids, N)
primary_diagnoses = np.random.choice(
    ["Diabetes", "Hypertension", "Heart Failure", "COPD",
     "Pneumonia", "Sepsis", "Renal Failure", "Stroke"], N,
    p=[0.18, 0.17, 0.15, 0.12, 0.13, 0.10, 0.08, 0.07]
)
length_of_stay = np.random.randint(1, 30, N)
admission_types = np.random.choice(["Emergency", "Elective", "Urgent"], N, p=[0.5, 0.3, 0.2])
discharge_dispositions = np.random.choice(
    ["Home", "Skilled Nursing", "Rehab", "Another Hospital", "Expired"],
    N, p=[0.60, 0.20, 0.12, 0.05, 0.03]
)
num_procedures = np.random.randint(0, 8, N)
num_medications = np.random.randint(1, 20, N)
num_previous_admissions = np.random.randint(0, 10, N)

# Build readmission label with realistic correlations
readmission_prob = (
    0.05
    + 0.006 * np.clip(ages - 40, 0, None)                     # older = much higher risk
    + 0.12 * (num_previous_admissions > 2).astype(int)         # strong repeat history
    + 0.05 * num_previous_admissions * 0.03                    # continuous effect
    + 0.12 * (primary_diagnoses == "Heart Failure").astype(int)
    + 0.10 * (primary_diagnoses == "COPD").astype(int)
    + 0.09 * (primary_diagnoses == "Renal Failure").astype(int)
    + 0.06 * (primary_diagnoses == "Sepsis").astype(int)
    + 0.07 * (admission_types == "Emergency").astype(int)
    - 0.05 * (discharge_dispositions == "Skilled Nursing").astype(int)
    + 0.003 * length_of_stay
    + 0.05 * (num_procedures > 4).astype(int)
)
readmission_prob = np.clip(readmission_prob, 0.02, 0.75)
readmitted = (np.random.random(N) < readmission_prob).astype(int)

admissions_df = pd.DataFrame({
    "admission_id": admission_ids,
    "patient_id": patient_ids,
    "hospital_id": assigned_hospitals,
    "primary_diagnosis": primary_diagnoses,
    "admission_type": admission_types,
    "length_of_stay": length_of_stay,
    "discharge_disposition": discharge_dispositions,
    "num_procedures": num_procedures,
    "num_medications": num_medications,
    "num_previous_admissions": num_previous_admissions,
    "readmitted_30days": readmitted
})

# ─── 4. Diagnoses Table ───────────────────────────────────────────────────────
comorbidities = ["Diabetes", "Hypertension", "Obesity", "Heart Disease",
                 "COPD", "Asthma", "Renal Disease", "Liver Disease"]
diag_rows = []
for adm_id in admission_ids:
    n_comorbid = np.random.randint(0, 4)
    selected = np.random.choice(comorbidities, n_comorbid, replace=False)
    for diag in selected:
        diag_rows.append({"admission_id": adm_id, "comorbidity": diag})

diagnoses_df = pd.DataFrame(diag_rows)

# ─── 5. Billing Table ─────────────────────────────────────────────────────────
base_cost = 5000 + 2000 * length_of_stay + 1500 * num_procedures
insurance_types = np.random.choice(["Ayushman Bharat", "Private Insurance", "Self Pay"], N, p=[0.45, 0.30, 0.25])
total_charges = base_cost * np.random.uniform(0.9, 1.3, N)
insurance_covered = np.where(
    insurance_types == "Self Pay", 0,
    np.where(insurance_types == "Ayushman Bharat", total_charges * np.random.uniform(0.6, 0.9, N),
             total_charges * np.random.uniform(0.4, 0.8, N))
)

billing_df = pd.DataFrame({
    "admission_id": admission_ids,
    "insurance_type": insurance_types,
    "total_charges": total_charges.round(2),
    "insurance_covered": insurance_covered.round(2)
})

# ─── Save to CSV ──────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
patients_df.to_csv("data/patients.csv", index=False)
hospitals_df.to_csv("data/hospitals.csv", index=False)
admissions_df.to_csv("data/admissions.csv", index=False)
diagnoses_df.to_csv("data/diagnoses.csv", index=False)
billing_df.to_csv("data/billing.csv", index=False)

print("✅ All datasets generated successfully!")
print(f"  patients.csv   : {len(patients_df)} rows")
print(f"  hospitals.csv  : {len(hospitals_df)} rows")
print(f"  admissions.csv : {len(admissions_df)} rows")
print(f"  diagnoses.csv  : {len(diagnoses_df)} rows")
print(f"  billing.csv    : {len(billing_df)} rows")
print(f"\n  Readmission rate: {readmitted.mean():.1%}")
