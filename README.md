🩺 Diabetes Prediction System
![Python](https://img.shields.io/badge/Python-3.x-3B8BD4?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-BA7517?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-1D9E75?style=flat-square)
![Status](https://img.shields.io/badge/Status-Baseline%20Model-E24B4A?style=flat-square)


> A supervised machine learning pipeline using Support Vector Machines to identify diabetes risk from clinical features — supporting early screening and digital health initiatives.


📊 Dataset
Property	Detail
Source	Pima Indians Diabetes Dataset
Samples	768 patients
Features	8 clinical variables
Target	`Outcome` — `0` Non-diabetic · `1` Diabetic


🧠 Project Pipeline

Raw Data → Preprocessing → Train SVM → Evaluate → Predict
  01            02             03          04         05

Step	Description
01 · Load Data	Read `diabetes.csv` using Pandas
02 · Preprocess	Drop target col, apply `StandardScaler`, stratified train/test split
03 · Train SVM	Fit `SVC(kernel='linear')` on scaled training data
04 · Evaluate	Measure accuracy on training and held-out test sets
05 · Predict	Standardize new input, run inference, return outcome label

⚙️ Methodology
Algorithm: Support Vector Machine (SVM) — Linear Kernel  
Chosen for its effectiveness on small-to-medium structured tabular datasets with clear class boundaries.
Preprocessing steps:
Removed `Outcome` target column from feature matrix
Applied `StandardScaler` for feature normalization
Used stratified split to preserve class balance across train/test sets


📈 Model Performance
Split	Accuracy
Training	78.6%
Test	77.3%
> ⚠️ Accuracy alone is not sufficient for clinical deployment. Future iterations will report ROC-AUC, F1-score, and confusion matrix breakdowns.

🔍 Key Predictors
Feature importance based on domain knowledge and SVM weight analysis:
```
Glucose          ████████████████████  High
BMI              ███████████████       Medium
Age              ████████████          Medium
Diabetes Pedigree ██████████           Moderate
Insulin          ████████              Moderate
Blood Pressure   █████                 Low



🔁 Prediction Workflow
Input — `(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigree, Age)`
```python
input_data = (10, 125, 70, 26, 115, 31.1, 0.205, 41)

# Standardize
input_array = np.asarray(input_data).reshape(1, -1)
std_data = scaler.transform(input_array)

# Predict
prediction = classifier.predict(std_data)

Output:

✓ Not Diabetic   (Outcome = 0)
```

⚠️ Limitations
Presence of unrealistic zero values in features like Glucose and BMI
Accuracy alone is insufficient for clinical use
Small dataset size (768 samples) limits generalizability
No model interpretability layer (e.g. SHAP values)
-
🔧 Future Improvements
[ ] Handle missing/zero values using median or KNN imputation
[ ] Implement advanced models — Random Forest, XGBoost
[ ] Add evaluation metrics — ROC-AUC, F1-score, confusion matrix
[ ] Build a Streamlit web app for real-time prediction
[ ] Integrate with retinal imaging AI (multimodal approach)
[ ] Add SHAP interpretability layer for clinical explainability
---
🧰 Tech Stack
![Python](https://img.shields.io/badge/-Python-3B8BD4?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-1D9E75?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-1D9E75?style=flat-square&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-BA7517?style=flat-square&logo=scikit-learn&logoColor=white)
---
📁 Project Structure
```
diabetes-prediction/
├── diabetes.csv              # Pima Indians dataset
├── model_training.ipynb      # Training notebook
├── prediction_script.py      # Standalone prediction script
└── README.md
```
---
🌍 Real-World Relevance
This project aligns with digital health trends and can be extended into:
Early diabetes screening tools for primary care
Clinical decision support systems for practitioners
Population health analytics platforms
---
🏁 Conclusion
This project demonstrates a practical application of machine learning in healthcare. The SVM baseline achieves ~77% test accuracy on a small structured dataset. Further improvements — better preprocessing, ensemble models, and richer evaluation metrics — are required before any real-world clinical use.
---
👤 Author
Prince Kwarteng Amaning  
MS Data Science — University of Michigan–Dearborn  
Background in Dentistry & Public Health
---
Built with Python · Scikit-learn · Pandas · NumPy
