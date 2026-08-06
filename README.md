This repository now includes a reproducible machine learning pipeline.

Usage

1. Install dependencies (preferably inside a virtual environment):

   pip install -r requirements.txt

2. Run the pipeline:

   python run_pipeline.py --data path/to/data.csv --target target_column --output outputs/

What it does

- Loads CSV dataset
- Automatic preprocessing (numeric and categorical handling)
- Feature engineering via ColumnTransformer
- Trains Logistic Regression, Linear SVM, Random Forest, XGBoost, CatBoost using nested cross-validation
- Computes ROC-AUC and Balanced Accuracy
- Generates calibration curves and reliability diagrams
- Produces SHAP explainability plots for the best models
- Saves models, predictions, metrics and figures under the output directory

Files added
- run_pipeline.py (top-level runner)
- src/ (modular pipeline code)
- requirements.txt

Notes
- The pipeline expects a tabular CSV. Provide the name of the target column.
- For large datasets or slow models (CatBoost/XGBoost), adjust inner/outer folds and n_jobs.
