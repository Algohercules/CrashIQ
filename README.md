# 🚦 Crash Severity Prediction


## 📌 Overview

Road accidents are a pressing public safety issue, particularly in high-traffic urban zones. Understanding and predicting the **severity of crashes**—whether a minor injury, major injury, or fatality—can help authorities take targeted action to prevent them.

This project uses **machine learning** to predict crash severity based on a variety of features such as vehicle speed, lane width, time of crash, and more. It provides actionable insights and policy recommendations aimed at enhancing road safety.

---

## 🎯 Objective

- Build a predictive model to classify accident severity into:
  - Minor Injury
  - Major Injury
  - Fatal Crash
- Use insights from the model to **propose interventions** for reducing severe road incidents.

---

## 🧠 Methodology

### 1. 📊 Exploratory Data Analysis (EDA)

- Dataset: 300 rows × 14 features (7 categorical)
- Clean and preprocessed: No missing values or class imbalance
- Key insights:
  - Peak crashes occur at 4 PM and 3 AM
  - Overspeeding is highly associated with fatal crashes
  - Lane width (especially 3.3–3.5 m) correlates with crash severity

### 2. 🔧 Feature Engineering

- **Speed Range Binning** – to analyze impact of overspeeding
- **Age Grouping** – for demographic risk analysis
- **Lane Width Binning** – for understanding structural risks
- **Overspeeding Binary Flags** – useful in behavior modeling

### 3. 🤖 Machine Learning Models

- **Random Forest**
- XGBoost
- LightGBM
- CatBoost

> **Best Performer:** Random Forest – ideal for small, clean datasets.  
> **Hyperparameter tuning:** Done using **Optuna** with Bayesian Optimization (TPE Sampler)

### 4. ⭐ Feature Selection Strategy

Using Random Forest importance scores, the top 5 impactful features were selected:
- Lane Width
- Over Speeding
- Vehicle Speed
- Driver Age
- Crash Time

This helped **improve model accuracy by ~5%** by removing noise.

---

## 📈 Results

| Model         | Accuracy | Comments                                         |
|---------------|----------|--------------------------------------------------|
| Random Forest | ✅ Best   | Stable, accurate, interpretable                  |
| CatBoost      | High     | Slightly overfit, good for categorical handling  |
| XGBoost       | Lower    | Less effective on small dataset                  |
| LightGBM      | Lower    | Similar to XGBoost, not ideal for low volume     |

---

## 🛠 Proposed Solutions Based on Model Insights

1. **Dynamic Speed Limits** – Adjust speed limits based on traffic/time/weather using digital boards
2. **Dedicated Lanes** – Separate lanes for 2-wheelers, trucks, and heavy vehicles
3. **Traffic Monitoring** – AI surveillance for overspeeding & lane misuse
4. **Urban Planning** – Promote public transport, build flyovers, optimize high-density zones

---

## 🔮 Future Scope

- Expand the dataset (driver fatigue, vehicle telemetry, road weather data)
- Deploy predictive tools in emergency services for real-time risk alerts
- Collaborate with urban planners & traffic departments for implementation

---


🏅 Recognized for innovative feature engineering and actionable real-world recommendations.



## 📁 Folder Structure

| Folder/File                   | Description                                                        |
|------------------------------|--------------------------------------------------------------------|
| `README.md`                  | Main documentation file for the project                            |
| `DataQuest_Megalith.ipynb`   | Jupyter notebook with EDA, feature engineering, and ML modeling    |
| `data/`                      | Folder containing dataset files                                    |
| └── `Data_Sheet.csv`         | Cleaned version of the dataset                                     |
| `presentation/`              | Folder for final hackathon presentation                            |
| └── `Resource.pptx`          | PowerPoint presentation with slides and visual insights            |
| `reports/`                   | (Optional) Folder for storing evaluation metrics or summaries      |
| └── `model_performance.md`   | Markdown file summarizing model performance                        |
| `models/`                    | (Optional) Folder for saving trained model artifacts               |
| └── `best_model.pkl`         | Serialized ML model (if saved)                                     |
| `utils/`                     | (Optional) Helper scripts or reusable functions                    |
| └── `helper_functions.py`    | Python script for utilities used in the notebook                   |



For queries or collaborations: [adarshrai1903@gmail.com]

---

> 🚧 *"Together, we can save lives and build safer roads for everyone."*
