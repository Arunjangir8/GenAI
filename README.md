# 🏠 Project 9 — Intelligent Property Price Prediction
### Milestone 1: ML-Based Property Price Prediction

---

## 📌 Overview

This project implements a **machine learning-based rental price prediction system** for Indian residential properties across **Delhi**, **Mumbai**, and **Pune**. It leverages classical ML techniques to estimate monthly rent based on property features, location, and amenities.

---

## 📁 Dataset

Three city-specific CSV datasets are merged into a unified DataFrame:

| File | City |
|------|------|
| `Indian_housing_Delhi_data.csv` | Delhi |
| `Indian_housing_Mumbai_data.csv` | Mumbai |
| `Indian_housing_Pune_data.csv` | Pune |

**Key features used:** Location, City, Size (sq ft), Rooms, Bathrooms, Balconies, Furnishing Status, Property Type, Security Deposit, Negotiability, Latitude/Longitude, Verification Days.

---

## ⚙️ Pipeline Summary

### 1. Data Integration
- Concatenated 3 city datasets using `pd.concat()`

### 2. Data Preparation & Feature Engineering
- Extracted numeric size from `house_size` string → `Size_ft²`
- Computed `Price_per_sqft` = `price_inr / Size_ft²`
- Parsed `house_type` → `rooms_num`, `BHK` (0=RK, 1=BHK), `property_type`
- Cleaned `SecurityDeposit` (removed commas, handled "No Deposit")
- Standardized `isNegotiable` → binary (0/1)
- Converted `verificationDate` strings → numeric `verification_days`
- Label-encoded: `Status`, `city`, `property_type`, `location`
- Filled missing values: `numBalconies` → 0, `numBathrooms` → median

### 3. Feature Discovery
- Correlation heatmap for all features
- Top features correlated with `price_inr`

### 4. Data Splitting
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5. Feature Scaling
- `MinMaxScaler` fitted on training set only; applied to test set

---

## 🤖 Models Trained

| Model | Notes |
|-------|-------|
| **Linear Regression** | Trained on MinMax-scaled features |
| **Random Forest Regressor** | 150 estimators, no scaling required |

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error (₹) |
| **RMSE** | Root Mean Squared Error (₹) |
| **R²** | Coefficient of Determination |

Visualizations included:
- ✅ Actual vs Predicted scatter plots (both models)
- ✅ Residual distribution histograms
- ✅ Train vs Test R² overfitting check
- ✅ Random Forest feature importance bar chart
- ✅ Linear Regression coefficient bar chart

---

## 🔮 Prediction Function

```python
preds = predict_rent(
    size_sqft=1000,
    rooms=2,
    bathrooms=2,
    balconies=1,
    bhk_flag=1,
    property_type_str='Independent Floor',
    city_str='Delhi',
    latitude=28.6139,
    longitude=77.2090,
    security_deposit=25000,
    is_negotiable=0,
    status_str='Furnished',
    location_str='Dwarka'
)
```

Returns predictions from **Linear Regression**, **Random Forest**, and their **Ensemble Average**, formatted as ₹ / month.

---

## 🏙️ Sample Multi-City Predictions

| Property | LR Estimate | RF Estimate | Ensemble |
|----------|-------------|-------------|----------|
| 1BHK Studio, Mumbai | — | — | — |
| 2BHK Floor, Delhi | — | — | — |
| 3BHK House, Pune | — | — | — |
| 4BHK Apt, Mumbai | — | — | — |

*(Values generated at runtime)*

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- **Models:** Linear Regression, Random Forest Regressor
- **Preprocessing:** MinMaxScaler, LabelEncoder
- **Environment:** Google Colab / Jupyter Notebook

---

## 🚀 How to Run

1. Upload the three CSV files to `/content/` in Google Colab (or update paths locally).
2. Run all cells sequentially from top to bottom.
3. Use `predict_rent()` at the end to generate rent estimates for custom properties.

---

## 👥 Team

> Team size: 3 students  
> Team Members: Arun, Mayank Yadav, Rohit Kumar    
> Course: AI/ML Project — Milestone 1 (Mid-Semester Submission)  

---

## 📋 Milestone 1 Deliverables Checklist

- [x] Problem understanding & use-case documentation
- [x] Input–output specification
- [x] Data preprocessing (encoding + scaling)
- [x] ML models: Linear Regression & Random Forest
- [x] Evaluation: MAE, RMSE, R²
- [x] Price driver analysis (feature importance + coefficients)
- [x] Working application with prediction function
- [x] UI (Streamlit/Gradio)
- [x] System architecture diagram

---

> **Note:** Milestone 2 will extend this system into an **Agentic AI Real Estate Advisory Assistant** using LangGraph, RAG (Chroma/FAISS), and LLM-based reasoning.

### Installation
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run app.py
```