# 🚦 CrashIQ Model Performance Report

Generated at: 2026-09-30 21:37:13

## 1. Executive Summary

The top-performing model is **Random Forest** with **56.67%** accuracy on the stratified holdout test set.

## 2. Model Benchmark Comparison

| Model | Accuracy | Macro F1 | Weighted F1 | Best Performer |
| :--- | :--- | :--- | :--- | :--- |
| Random Forest | 56.67% | 56.90% | 56.90% | ✅ **Best** |
| CatBoost | 43.33% | 41.90% | 41.90% | — |
| XGBoost | 46.67% | 46.30% | 46.30% | — |
| LightGBM | 43.33% | 43.68% | 43.68% | — |

## 3. Best Model Detailed Classification Report

```text
              precision    recall  f1-score   support

 Fatal crash       0.65      0.55      0.59        20
Major injury       0.61      0.55      0.58        20
Minor injury       0.48      0.60      0.53        20

    accuracy                           0.57        60
   macro avg       0.58      0.57      0.57        60
weighted avg       0.58      0.57      0.57        60

```

## 4. Confusion Matrix

```text
[[11  3  6]
 [ 2 11  7]
 [ 4  4 12]]
```

Classes (rows=actual, cols=predicted):
- Class 0: Fatal crash
- Class 1: Major injury
- Class 2: Minor injury
