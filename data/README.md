# Data

## `raw/openml_37_diabetes.arff`

| | |
|---|---|
| Dataset | Pima Indians Diabetes Database |
| Original owner | National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) |
| Donor | Vincent Sigillito, Johns Hopkins University Applied Physics Laboratory (UCI, 1990) |
| Source used | OpenML dataset 37, https://www.openml.org/d/37 |
| Direct file | https://openml.org/data/v1/download/37/diabetes.arff |
| Licence | None named. OpenML's licence field reads only "Public", which isn't a named licence, so check the terms before reusing the data elsewhere |
| Downloaded | 2026-10-24 |
| MD5 | `3cbaa3e54586aa88cf6aacb4033e4470` (matches the checksum published in OpenML's metadata) |
| SHA-256 | `4eddd5b2b64679e8888348e306520a393d6a28e1ddc9643cfb76fc5d912d6d40` |
| Size | 768 rows × 9 columns (8 features + `class`) |

The dataset was removed from the current UCI repository site, so OpenML's copy of the UCI file is used. It's value-for-value identical (features and labels) to the widely used Kaggle `diabetes.csv` the project previously used.

**Population:** females at least 21 years old of Pima Indian heritage, near Phoenix, Arizona. **Outcome:** diabetes according to WHO criteria (2-hour post-load plasma glucose ≥ 200 mg/dl at any survey examination, or found during routine medical care).

**Known data issue:** zeros encode missing measurements: Glucose 5, BloodPressure 35, SkinThickness 227, Insulin 374, BMI 11. The analysis converts them to missing values (see the notebook).

**Reference:** Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Symposium on Computer Applications and Medical Care*, 261–265.

There are no identifiers. The data contains only the 8 numeric measurements and the outcome.
