#!/usr/bin/env python3
"""
CrashIQ: Main Execution & Model Training Pipeline
Trains all 4 machine learning models (Random Forest, XGBoost, LightGBM, CatBoost),
benchmarks their performance, saves the best model artifact, and generates the evaluation report.
"""

import os
import sys
import time
from utils.helper_functions import (
    load_and_preprocess_data,
    get_train_test_split,
    initialize_model,
    evaluate_model,
    save_model,
    predict_severity,
    INV_SEVERITY_MAPPING,
    DEFAULT_FEATURE_COLS
)


def run_pipeline():
    print("=" * 70)
    print("🚦 CrashIQ: Road Crash Severity Prediction & Analysis Pipeline")
    print("=" * 70)

    # 1. Load data
    print("\n[1/5] Loading and preprocessing dataset...")
    df = load_and_preprocess_data()
    print(f"  ✓ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  ✓ Engineered features: 'Over_Speeding', 'Over_Speeding_binary'")
    print(f"  ✓ Target distribution:\n{df['Crash_Severity'].value_counts().to_string()}")

    # 2. Split train/test
    print(f"\n[2/5] Splitting data using top 5 impactful features...")
    print(f"  Selected features: {DEFAULT_FEATURE_COLS}")
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    print(f"  ✓ Train size: {X_train.shape[0]} samples")
    print(f"  ✓ Test size:  {X_test.shape[0]} samples (stratified)")

    # 3. Train models
    print("\n[3/5] Training and benchmarking machine learning models...")
    models_to_train = ['Random Forest', 'CatBoost', 'XGBoost', 'LightGBM']
    results = {}
    fitted_models = {}

    for name in models_to_train:
        t0 = time.time()
        print(f"  → Training {name}...", end=" ", flush=True)
        model = initialize_model(name)
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        eval_res = evaluate_model(model, X_test, y_test)
        results[name] = eval_res
        fitted_models[name] = model
        print(f"Done in {elapsed:.2f}s | Test Accuracy: {eval_res['accuracy']*100:.2f}%")

    # 4. Compare results
    print("\n[4/5] Model Benchmark Summary:")
    print("-" * 70)
    print(f"{'Model':<16} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<12} | {'Status'}")
    print("-" * 70)

    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    for name, res in results.items():
        acc = res['accuracy'] * 100
        macro_f1 = res['report_dict']['macro avg']['f1-score'] * 100
        weighted_f1 = res['report_dict']['weighted avg']['f1-score'] * 100
        is_best = "✅ Best Model" if name == best_model_name else "  Baseline"
        print(f"{name:<16} | {acc:>8.2f}% | {macro_f1:>8.2f}% | {weighted_f1:>10.2f}% | {is_best}")
    print("-" * 70)

    # Save best model
    best_model = fitted_models[best_model_name]
    model_path = "models/best_model.pkl"
    save_model(
        best_model,
        feature_cols=DEFAULT_FEATURE_COLS,
        filepath=model_path,
        metadata={
            'name': best_model_name,
            'accuracy': results[best_model_name]['accuracy'],
            'report': results[best_model_name]['report_dict']
        }
    )
    print(f"\n  ✓ Saved best model ({best_model_name}) to: {model_path}")

    # Generate Markdown Performance Report
    report_path = "reports/model_performance.md"
    generate_markdown_report(results, best_model_name, report_path)
    print(f"  ✓ Saved performance report to: {report_path}")

    # 5. Interactive Test Prediction
    print("\n[5/5] Sample Inference Demonstration:")
    test_cases = [
        {"speed": 115, "limit": 60, "time": 3, "age": 22, "lane_width": 3.1, "scenario": "High Speed at 3 AM (Severe Risk)"},
        {"speed": 55, "limit": 60, "time": 11, "age": 45, "lane_width": 3.4, "scenario": "Within Speed Limit at Midday (Standard)"}
    ]

    best_bundle = {
        'model': best_model,
        'feature_cols': DEFAULT_FEATURE_COLS,
        'inv_severity_mapping': INV_SEVERITY_MAPPING
    }

    for tc in test_cases:
        print(f"\n  Scenario: {tc['scenario']}")
        print(f"    Inputs: Speed={tc['speed']} km/h (Limit: {tc['limit']} km/h), Time={tc['time']}:00, Driver Age={tc['age']}, Lane Width={tc['lane_width']}m")
        pred = predict_severity(
            best_bundle,
            vehicle_speed=tc['speed'],
            speed_limit=tc['limit'],
            crash_time=tc['time'],
            age=tc['age'],
            lane_width=tc['lane_width']
        )
        print(f"    Predicted Severity : {pred['predicted_class']} (Risk Level: {pred['risk_level']})")
        prob_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in pred['probabilities'].items()])
        print(f"    Class Probabilities: {prob_str}")
        if pred['safety_recommendations']:
            print(f"    Recommendations    :")
            for rec in pred['safety_recommendations']:
                print(f"      - {rec}")

    print("\n" + "=" * 70)
    print("✅ CrashIQ Pipeline Execution Completed Successfully!")
    print("=" * 70)
    print("💡 To launch the interactive web dashboard:")
    print("   streamlit run app.py")
    print("=" * 70)


def generate_markdown_report(results, best_model_name, filepath):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write("# 🚦 CrashIQ Model Performance Report\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write(f"The top-performing model is **{best_model_name}** with **{results[best_model_name]['accuracy']*100:.2f}%** accuracy on the stratified holdout test set.\n\n")
        f.write("## 2. Model Benchmark Comparison\n\n")
        f.write("| Model | Accuracy | Macro F1 | Weighted F1 | Best Performer |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for name, res in results.items():
            acc = f"{res['accuracy']*100:.2f}%"
            macro_f1 = f"{res['report_dict']['macro avg']['f1-score']*100:.2f}%"
            weighted_f1 = f"{res['report_dict']['weighted avg']['f1-score']*100:.2f}%"
            is_best = "✅ **Best**" if name == best_model_name else "—"
            f.write(f"| {name} | {acc} | {macro_f1} | {weighted_f1} | {is_best} |\n")
        
        f.write("\n## 3. Best Model Detailed Classification Report\n\n")
        f.write("```text\n")
        f.write(results[best_model_name]['report_str'])
        f.write("\n```\n\n")

        f.write("## 4. Confusion Matrix\n\n")
        f.write("```text\n")
        f.write(f"{results[best_model_name]['confusion_matrix']}\n")
        f.write("```\n\n")
        f.write("Classes (rows=actual, cols=predicted):\n")
        f.write("- Class 0: Fatal crash\n")
        f.write("- Class 1: Major injury\n")
        f.write("- Class 2: Minor injury\n")


if __name__ == '__main__':
    run_pipeline()
