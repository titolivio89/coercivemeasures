# Prediction of Coercive Measures Using Privacy-Preserving Large Language Models

## Overview

This repository contains a reproducible machine learning pipeline for predicting coercive measures in psychiatric inpatient care using structured clinical variables extracted from German admission notes by a locally deployed Large Language Model (LLM).

The project accompanies the doctoral research of Guillermo Calvi at the Department of Psychiatry and Psychotherapy, University Hospital Carl Gustav Carus, Technische Universität Dresden.

---

## Objectives

- Reproduce the dissertation analyses
- Benchmark multiple machine learning models
- Improve model generalizability
- Develop an explainable prediction pipeline
- Prepare a publication-ready analysis workflow

---

## Pipeline

Admission Notes

↓

LLM Information Extraction

↓

Structured Clinical Variables

↓

Machine Learning Models

↓

Prediction of Coercive Measures

↓

Evaluation & Explainability

---

## Repository Structure

```
configs/
data/
docs/
figures/
manuscript/
notebooks/
results/
src/

README.md
requirements.txt
environment.yml
run_pipeline.py
```

---

## Planned Models

- Logistic Regression
- Linear SVM
- Random Forest
- XGBoost
- CatBoost

---

## Evaluation

- ROC-AUC
- Balanced Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Calibration
- Feature Importance

---

## Future Work

- Temporal validation
- External validation
- Explainability (SHAP)
- Clinical decision support
