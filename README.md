# 📉 Telco Customer Churn Prediction Streamlit App

An interactive machine learning application built with **Streamlit** and **Scikit-Learn** ensemble models to predict customer churn risk, analyze key behavioral indicators, and suggest retention strategies.

---

## 🚀 Features

- **Multi-Model Support**: Select between Decision Tree, KNN, and Logistic Regression Bagging ensembles.
- **Dynamic Thresholding**: Adjust decision threshold dynamically based on business requirements (higher recall vs. precision).
- **Persona Presets**: One-click demo personas (High Risk Churner, Loyal Champion, Moderate Risk) for rapid testing.
- **Key Risk Diagnostics**: Automated extraction of top behavioral churn signals (contract terms, payment methods, add-on services).
- **Prescriptive Retention Actions**: Contextual recommendations for customer success and retention teams.
- **Batch CSV Processing**: Upload a batch of customer profiles and export scored predictions with churn probability and risk tier.

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Shruti-Mandlik-07/ML_Streamlid_App.git
   cd ML_Streamlid_App
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   python -m streamlit run app.py
   ```

---

## 📂 Project Structure

```
├── app.py              # Main Streamlit web application
├── requirements.txt    # Project dependencies
├── bagging_lr.pkl      # Bagging + Logistic Regression model
├── bagging_dt.pkl      # Bagging + Decision Tree model
├── bagging_knn.pkl     # Bagging + KNN model
├── best_model.pkl      # Optimized benchmark model
├── .gitignore          # Ignored cache & system files
└── README.md           # Documentation
```
