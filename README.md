### ⚠ Model Limitation
The dataset used is highly imbalanced toward lung cancer cases.
To avoid false positives, a rule-based validation layer was added:
If all symptom values are zero, the system predicts "No Lung Cancer"
before applying the ML model.
