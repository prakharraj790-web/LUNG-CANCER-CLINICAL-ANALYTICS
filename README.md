# 🫁 Lung Cancer Clinical Analytics & Machine Learning — 4-Tier Analytics Ladder

An enterprise-grade, end-to-end Data Analytics and Machine Learning project executed on the multi-year, multi-regional **Lung Cancer Global Clinical & Risk Factor Dataset (2015–2026)**.

---

> [!IMPORTANT]
> ### ⚠️ ETHICAL & CLINICAL REGULATORY DISCLAIMER
> **This dataset contains 2,000 synthetic patient records created for educational, analytical training, and methodological research purposes inspired by WHO and GLOBOCAN 2022 oncology statistics.**
> **The predictive models, risk ranking mechanisms, and prescriptive frameworks produced in this repository are NOT clinically validated medical devices and MUST NOT be used for real-world medical diagnosis, clinical prognosis, triage, or therapeutic decision-making.**

---

## 1. Problem Statement

Lung cancer is the foremost cause of cancer-related mortality globally (~1.8 million deaths annually). In clinical healthcare administration and oncology informatics, organizations face two core challenges:
1. **Target Leakage in Predictive Pipelines:** Machine learning models developed on observational electronic health record (EHR) registries often inadvertently ingest post-diagnosis variables (e.g., post-diagnostic survival duration or treatment course progression), creating artificially inflated offline test metrics that catastrophically fail in real prospective clinical environments.
2. **The "Last Mile" Gap in Analytics:** Models typically terminate at binary classification without providing continuous risk calibration or addressing real-world operational bottlenecks, such as fixed clinical audit bandwidth.

**Analytical Objective:** Develop an end-to-end, leakage-free analytics pipeline structured across the **4-Tier Analytics Ladder** (Descriptive, Diagnostic, Predictive, Prescriptive) to estimate patient survival status (`Survived: Yes / No`) using strictly baseline presentation features and operationalize predictions through a resource-constrained risk prioritization framework.

---

## 2. Dataset Description & Provenance

* **Dataset Name:** Lung Cancer Global Clinical & Risk Factor Dataset (2015–2026)
* **Dataset Reference / Source:** [Khurram Shahzad on Kaggle](https://www.kaggle.com/datasets/zkskhurram/lung-cancer-dataset)
* **Clinical Domain Inspirations:** [WHO Lung Cancer Fact Sheet](https://www.who.int/news-room/fact-sheets/detail/lung-cancer), [GLOBOCAN 2022 - IARC Cancer Statistics](https://gco.iarc.who.int/), [National Cancer Institute (NCI)](https://www.cancer.gov/)
* **Cohort Size:** 2,000 unique patients (`Patient_ID: LC-0001` to `LC-2000`).
* **Geographic Span:** All 6 WHO Regions (Western Pacific, Europe, Americas, South-East Asia, Eastern Mediterranean, Africa) across 60 countries.
* **Clinical Domain:** Demographics, lifestyle/occupational carcinogens, genetic driver mutations, 10 presenting symptoms, histological classifications (NSCLC subtypes, SCLC), anatomical staging (Stages I–IV), tumor size (cm), distant metastasis, diagnostic modality, primary assigned treatment, and survival outcomes.
* **Target Variable:** `Survived` (Binary: `Yes` = 756 [37.8%], `No` = 1,244 [62.2%]).
* **Synthetic Cohort Limitation:** While inspired by global epidemiology, records are synthetic and intended strictly for educational and analytical research purposes.

---

## 3. Technologies Used

* **Core Programming:** Python 3.14+
* **Web Application Framework:** Streamlit (interactive multi-section clinical portal)
* **Machine Learning & Modeling:** Scikit-learn (Pipelines, ColumnTransformer, StandardScaler, OneHotEncoder, LogisticRegression, RandomForestClassifier, ROC-AUC, Confusion Matrix)
* **Data Processing & Analytics:** Pandas, NumPy
* **Scientific Visualization:** Matplotlib, Seaborn
* **Reporting Automation:** python-docx (formal Word .docx report generation)

---

## 4. Data Dictionary

| Variable | Type | Clinical Domain | Modeling Role / Status | Description |
| :--- | :--- | :--- | :--- | :--- |
| `Patient_ID` | String | Identifier | **EXCLUDED** | Unique patient alphanumeric code (`LC-XXXX`) |
| `Diagnosis_Date` | Date | Metadata | **EXCLUDED** | Calendar timestamp of diagnostic confirmation |
| `Diagnosis_Year` | Integer | Metadata | **EXCLUDED** | Year of diagnosis (2015–2026) |
| `Age` | Integer | Demographic | Predictor | Patient age in years (30–89) |
| `Gender` | Categorical | Demographic | Predictor | Patient gender (`Male`, `Female`) |
| `WHO_Region` | Categorical | Demographic | Predictor | 6 World Health Organization geographical regions |
| `Country` | Categorical | Demographic | Predictor | 60 global countries |
| `Smoking_Status` | Categorical | Lifestyle Risk | Predictor | `Never Smoked`, `Former Smoker`, `Current Smoker` |
| `Cigarettes_Per_Day` | Integer | Lifestyle Risk | Predictor | Average cigarettes consumed per day (0 for non-smokers) |
| `Years_Smoking` | Integer | Lifestyle Risk | Predictor | Cumulative duration of smoking history in years |
| `Secondhand_Smoke` | Binary | Environmental | Predictor | Passive smoke exposure (`Yes`, `No`) |
| `Family_History` | Binary | Genetic Risk | Predictor | First-degree family history of lung cancer (`Yes`, `No`) |
| `Occupational_Hazard` | Binary | Environmental | Predictor | Workplace carcinogen exposure (`Yes`, `No`) |
| `Air_Pollution_Exposure`| Categorical | Environmental | Predictor | Ambient air pollution exposure (`Low`, `Moderate`, `High`) |
| `Alcohol_Use` | Categorical | Lifestyle Risk | Predictor | Consumption intensity (`No Alcohol`, `Moderate`, `Heavy`) |
| `BMI` | Float | Clinical Vitals | Predictor | Body Mass Index (16.0–40.0) |
| `Exercise_Frequency` | Categorical | Lifestyle Risk | Predictor | Physical activity level (`Low`, `Moderate`, `High`) |
| `Chronic_Lung_Disease`| Binary | Comorbidity | Predictor | History of pre-existing COPD / chronic lung disease (`Yes`, `No`) |
| `Asbestos_Exposure` | Binary | Environmental | Predictor | Direct occupational/environmental asbestos contact (`Yes`, `No`) |
| `Radon_Exposure` | Binary | Environmental | Predictor | Residential/workplace radon exposure (`Yes`, `No`) |
| `Previous_Cancer_History`| Binary | Comorbidity | Predictor | History of any prior malignancy (`Yes`, `No`) |
| `Genetic_Mutation` | Categorical | Biomarker | Predictor | Oncogenic mutations (`EGFR`, `ALK`, `KRAS`, `TP53`, `ROS1`, `BRAF`, etc.) |
| `Coughing`, `Shortness_of_Breath`, `Chest_Pain`, `Coughing_Blood`, `Fatigue`, `Weight_Loss`, `Wheezing`, `Recurrent_Infections`, `Swallowing_Difficulty`, `Finger_Clubbing` | Binary (x10) | Presentation Symptoms | Predictors | 10 clinical signs and symptoms evaluated at intake (`Yes`, `No`) |
| `Cancer_Type` | Categorical | Histology | Predictor | `NSCLC` (Non-Small Cell Lung Cancer), `SCLC` (Small Cell) |
| `NSCLC_Subtype` | Categorical | Histology | Predictor | `Adenocarcinoma`, `Squamous Cell`, `Large Cell`, `Not Applicable` |
| `Cancer_Stage` | Categorical | Staging | Predictor | Baseline anatomical extent: `Stage I`, `Stage II`, `Stage III`, `Stage IV` |
| `Tumor_Size_cm` | Float | Staging | Predictor | Primary tumor diameter in cm at diagnosis (0.5–13.3 cm) |
| `Metastasis` | Binary | Staging | Predictor | Distant metastatic disease present (`Yes`, `No`) |
| `Diagnosis_Method` | Categorical | Clinical Tool | Predictor | Method of diagnosis (`CT Scan`, `PET Scan`, `Biopsy`, `Bronchoscopy`, `LDCT Screening`, etc.) |
| `Treatment` | Categorical | Management | Predictor | Baseline planned therapy (`Surgery`, `Chemotherapy`, `Immunotherapy`, `Targeted Therapy`, etc.) |
| `Survival_Months` | Integer | Outcome Metric | **STRICTLY EXCLUDED** | Duration survived post-diagnosis (1–110 mos) — **Incurable Target Leakage** |
| `Survived` | Binary | Primary Target | **TARGET VARIABLE** | Survival status (`Yes`, `No`) |

---

## 4. Data Cleaning & Hygiene Protocols

A formal data hygiene audit verified:
1. **Entity Architecture:** Verified 2,000 unique `Patient_ID` records. Exactly 1 row per patient; unnecessary aggregation was bypassed.
2. **Duplicate Ingestion Scan:** 0 duplicate rows across all 41 columns.
3. **Missing Value Audit:** 0 null/NaN values across all 82,000 data cells.
4. **Data Type Coercion:** `Diagnosis_Date` converted to standard ISO datetime.
5. **Boundary & Anomaly Validation:** Checked physiological plausibility boundaries:
   - `Age`: 30 to 89 years (no negative values, no pediatric records).
   - `BMI`: 16.0 to 40.0 (clinically plausible range).
   - `Tumor_Size_cm`: 0.5 to 13.3 cm (all positive).
   - `Survival_Months`: 1 to 110 months.
6. **Cross-Feature Logic Checks:** Verified that all `Never Smoked` patients strictly have 0 `Cigarettes_Per_Day` and 0 `Years_Smoking`.

---

## 5. Four-Tier Analytics Methodology

```mermaid
flowchart LR
    T1["Tier 1: Descriptive\n'What happened?'\nCohort Baseline & Hygiene"] --> T2["Tier 2: Diagnostic\n'Why did it happen?'\n5 Publication EDA Charts"]
    T2 --> T3["Tier 3: Predictive\n'What will happen?'\nLeakage-Free ML Models"]
    T3 --> T4["Tier 4: Prescriptive\n'What action to take?'\nRisk Ranking & 10% Rule"]
```

### Tier 1 — Descriptive Analytics: What Happened?
Establishes the clinical and demographic baseline of the 2,000-patient intake:
* **Cohort Volume:** 2,000 patients.
* **Observed Survival Rate:** 37.8% (756 survivors vs. 1,244 non-survivors).
* **Mean Patient Age:** 60.75 years (Median: 61 yrs).
* **Mean Primary Tumor Diameter:** 4.55 cm.
* **Mean Survival Duration:** 31.76 months.

### Tier 2 — Diagnostic Analytics: What Associations Are Visible?
Investigates associative structures across demographic, lifestyle, staging, and therapeutic dimensions using 5 publication-standard figures. **Strict epidemiological rule: correlation is clearly distinguished from causation.**

### Tier 3 — Predictive Analytics: What Will Happen?
Trains predictive binary classification architectures on 36 baseline pre-treatment predictors to forecast survival probability while strictly excluding target-leaking fields.

### Tier 4 — Prescriptive Analytics: How Can Limited Resources Be Prioritized?
Converts continuous calibrated non-survival probabilities into a capacity-constrained operational decision rule (e.g., top 10% clinical case audit).

---

## 6. Exploratory Data Analysis (EDA) Findings

### Chart 1: Survival Distribution & Class Balance
* **Empirical Observation:** The cohort comprises 1,244 non-surviving patients (62.20%) and 756 surviving patients (37.80%), demonstrating a non-survival-to-survival ratio of 1.65:1. Class weighting protocols were integrated into predictive models to address this imbalance.

### Chart 2: Age Demographic Distribution
* **Empirical Observation:** Non-surviving patients had an observed mean age of 61.1 years, compared to 60.2 years among survivors. Stratified across age brackets, survival was 41.2% in patients <50 years, 38.4% in ages 50–64, 36.5% in ages 65–74, and 34.1% in ages 75+. The data shows an inverse descriptive association between age and survival proportion without establishing age as an isolated causal driver.

### Chart 3: Smoking Risk Factor Analysis
* **Empirical Observation:** Never Smokers (N=1,311) exhibited a 40.2% survival proportion, Former Smokers (N=212) exhibited 35.8%, and Current Smokers (N=477) exhibited 32.1%. While active smoking associates descriptively with lower survival proportions, this represents an observational correlation.

### Chart 4: Clinical Cancer Stage vs. Survival
* **Empirical Observation:** Survival proportion exhibited strong stage-stratified divergence:
  - **Stage I:** 58.9% survival (340 / 577)
  - **Stage II:** 44.8% survival (218 / 487)
  - **Stage III:** 28.5% survival (131 / 460)
  - **Stage IV:** 14.1% survival (67 / 476)
  This inverse relationship highlights the clinical importance of early-stage detection.

### Chart 5: Tumor Size, Metastasis & Treatment Modalities
* **Empirical Observation:** Patients presenting with distant metastasis had a mean tumor diameter of 6.8 cm vs. 3.7 cm in non-metastatic patients. Across therapeutic modalities, surgical cohorts exhibited higher observed survival (56.1% for Surgery, 48.9% for Surgery + Chemotherapy) than systemic Chemotherapy (27.3%) or Palliative Care (14.2%). **Critical Clinical Note:** This variance reflects clinical indication bias (patients with localized early-stage disease are eligible for curative-intent surgery, whereas advanced metastatic patients receive palliative care), not drug efficacy superiority.

---

## 7. Machine Learning & Target Leakage Prevention

### Strict Leakage Protocol
* `Survival_Months` is **strictly excluded**. In clinical reality, time survived is an outcome of diagnosis and post-diagnosis care, completely unavailable at intake. Retaining it would artificially inflate metrics through retrospective leakage.
* `Patient_ID` is **excluded** as an arbitrary primary key.
* `Diagnosis_Date` and `Diagnosis_Year` are **excluded** to prevent chronological confounding.

### Preprocessing & Architecture
* **Split:** 80% Train (1,600 patients) / 20% Test (400 patients), stratified by target, `random_state=42`.
* **Preprocessing Pipeline:** `StandardScaler` for continuous features, `OneHotEncoder(handle_unknown='ignore')` for categorical features.
* **Models Evaluated:**
  1. **Logistic Regression:** Linear log-odds estimator with balanced class weighting.
  2. **Random Forest Classifier:** 200 trees, max depth 12, balanced subsample weighting.

---

## 8. Model Evaluation Results (Test Split: N=400)

| Metric | Logistic Regression | Random Forest Classifier |
| :--- | :---: | :---: |
| **Accuracy** | **0.7025** (70.25%) | **0.6950** (69.50%) |
| **Precision** | **0.5777** (57.77%) | **0.5707** (57.07%) |
| **Recall (Sensitivity)** | **0.7881** (78.81%) | **0.7748** (77.48%) |
| **F1-Score** | **0.6667** | **0.6573** |
| **ROC-AUC** | **0.7562** | **0.7693** |
| **True Positives (TP)** | 119 | 117 |
| **True Negatives (TN)** | 162 | 161 |
| **False Positives (FP)** | 87 | 88 |
| **False Negatives (FN)** | 32 | 34 |

*Note: Evaluated on test set ($N=400$ patients, 20% holdout). Evaluated using balanced class weighting to ensure high sensitivity/recall for survival identification.*

---

## 9. Feature Importance

Feature importance extracted from the Random Forest ensemble indicates that the most influential predictors at baseline presentation are:
1. `Cancer_Stage` (Stage I vs. Stage IV indicators)
2. `Tumor_Size_cm` (continuous tumor diameter)
3. `Metastasis` (presence of distant organ metastasis)
4. `Treatment` (planned surgical vs. palliative modality)
5. `Age` (continuous patient age)
6. `Smoking_Status` / `Cigarettes_Per_Day` / `Years_Smoking`

**Governance Clarification:** Feature importance reflects mathematical variance reduction in the decision trees and predictive association, not biological causality.

---

## 10. Prescriptive Analytics & Resource-Constrained Rule

### Risk Prioritization Framework
Patients are ranked according to their predicted probability of non-survival ($P(\text{unfavorable outcome}) = 1 - P(\text{Survived = Yes})$).
* **High Predicted Risk:** Upper 25th percentile ($P \ge 0.70$)
* **Medium Predicted Risk:** 25th to 75th percentile ($0.45 \le P < 0.70$)
* **Lower Predicted Risk:** Lower 25th percentile ($P < 0.45$)

### Resource-Constrained Operational Rule (Top 10% Queue)
In an operational setting where clinical audit staff can manually review only **10% of new patient intakes** (200 patients in a 2,000-patient cohort):
* The algorithm ranks the cohort and queues the **top 200 highest-risk individuals**.
* **Cohort Profile:** The flagged decile exhibits an average predicted unfavorable probability of **~85%**, a **~75% metastasis rate**, and a mean tumor diameter of **~7.2 cm**.
* **Operational Action:** Automatically routing these patients to expedited multidisciplinary tumor board review ensures specialized palliative, surgical, and genetic therapy assessments occur without delay.

---

## 11. Limitations

1. **Synthetic Data:** The dataset is synthetic, inspired by real-world oncology aggregates; it cannot reflect all biological nuances of real patient pathophysiology.
2. **Clinical Indication Confounding:** Treatment variables in observational data reflect the severity of the disease at presentation, precluding causal inference regarding treatment efficacy.
3. **Generalizability:** The model has not been validated on real clinical cohorts or distinct healthcare system registries.

---

## 12. How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute Modular Analytics Pipeline
```bash
# Data loading, hygiene audit, and cleaning
python -m src.data_loader

# Generate all 5 publication-ready EDA charts
python -m src.eda

# Train and evaluate leakage-free ML models
python -m src.modeling

# Execute risk prioritization and operational rules
python -m src.prescriptive

# Build formal 27-section academic PDF report (report.pdf)
python -m src.report_builder
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

The application opens in your browser at `http://localhost:8501`.
