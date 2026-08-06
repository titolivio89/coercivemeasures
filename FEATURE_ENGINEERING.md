# Feature engineering documentation

This file documents the feature engineering and preprocessing applied by the pipeline for TRIPOD-AI reporting.

Preprocessing steps (applied to the training set only, then saved and applied to test):

1. Numerical features
   - Missing values imputed with the median of the training set for each feature (SimpleImputer(strategy='median')).
   - Features scaled to zero mean and unit variance using StandardScaler fitted on the training set.

2. Categorical features
   - Missing values (NaN) imputed with the constant string "__missing__" (SimpleImputer(strategy='constant')).
   - One-hot encoding applied using sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore').
     The encoder is fitted on the training set categories and will ignore unseen categories during testing.
   - Note: OneHotEncoder returns dense arrays in the current configuration (sparse=False) for simplicity.

3. Feature selection / engineering
   - The pipeline currently does not apply automated feature selection or creation (e.g., interaction terms).
   - If desired, add transformations in src/preprocessing.py within the ColumnTransformer composition.

Reproducibility
- The preprocessor object is fitted only on the training set and saved (joblib) to outputs/preprocessing/preprocessor.joblib.
- Random seeds (numpy and python random) are fixed and saved in experiment metadata.

Recommendations for TRIPOD-AI
- Report the number of features before and after one-hot encoding in the manuscript; this can be obtained by loading the saved preprocessor and inspecting the transformer output shape.
- For categorical variables with many levels consider alternative encodings (target encoding, embedding) and document the choice and justification.
