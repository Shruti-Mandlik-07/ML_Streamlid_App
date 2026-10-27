import os
import pickle
import warnings
import pandas as pd
import streamlit as st

# Suppress sklearn unpickling warnings
warnings.filterwarnings("ignore")

# Backward-compatibility shim for ColumnTransformer serialized in scikit-learn <= 1.6
import sklearn.compose._column_transformer as ct
if not hasattr(ct, "_RemainderColsList"):
    ct._RemainderColsList = type("_RemainderColsList", (list,), {})

# Set Page Config
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished UI
st.markdown("""
<style>
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 15px;
    }
    .risk-badge-high {
        background-color: #ff4b4b22;
        color: #ff4b4b;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #ff4b4b66;
    }
    .risk-badge-low {
        background-color: #21c35422;
        color: #21c354;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #21c35466;
    }
    .risk-badge-med {
        background-color: #ffa50022;
        color: #ffa500;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #ffa50066;
    }
    .recommendation-box {
        background: rgba(14, 165, 233, 0.08);
        border-left: 4px solid #0ea5e9;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_FILES = {
    "⭐ Best Model (Bagging + Logistic Regression)": "best_model.pkl",
    "Bagging + Decision Tree": "bagging_dt.pkl",
    "Bagging + KNN": "bagging_knn.pkl",
    "Bagging + Logistic Regression": "bagging_lr.pkl",
}

@st.cache_resource
def load_model(filename):
    model_path = os.path.join(os.path.dirname(__file__), filename)
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    model_name = st.selectbox("Choose Machine Learning Model", list(MODEL_FILES.keys()))
    threshold = st.slider("Churn Decision Threshold", 0.10, 0.90, 0.50, 0.05, 
                          help="Lower threshold catches more potential churners (higher recall) at the cost of more false alarms.")
    
    st.markdown("---")
    st.subheader("💡 Quick Presets")
    preset = st.radio(
        "Load Demo Persona",
        ["Custom", "🚨 High Risk Churner", "🛡️ Loyal Champion", "⚖️ Moderate Risk"],
        help="Quickly populate inputs to test model behavior on typical customer profiles."
    )
    st.markdown("---")
    st.caption("Telco Churn AI Studio • Powered by Scikit-Learn Ensembles & Streamlit")

# Main Header
st.title("📉 Telco Customer Churn Intelligence")
st.markdown("Predict customer churn probability, analyze risk factors, and generate retention recommendations.")

# Setup default values based on presets
defaults = {
    "gender": "Female", "senior": "No", "partner": "No", "dependents": "No",
    "tenure": 3, "phone": "Yes", "multi": "No", "internet": "Fiber optic",
    "sec": "No", "backup": "No", "protect": "No", "tech": "No",
    "tv": "Yes", "movies": "Yes", "contract": "Month-to-month",
    "paperless": "Yes", "payment": "Electronic check", "monthly": 85.0, "total": 255.0
}

if preset == "🚨 High Risk Churner":
    defaults = {
        "gender": "Male", "senior": "Yes", "partner": "No", "dependents": "No",
        "tenure": 2, "phone": "Yes", "multi": "Yes", "internet": "Fiber optic",
        "sec": "No", "backup": "No", "protect": "No", "tech": "No",
        "tv": "Yes", "movies": "Yes", "contract": "Month-to-month",
        "paperless": "Yes", "payment": "Electronic check", "monthly": 95.5, "total": 191.0
    }
elif preset == "🛡️ Loyal Champion":
    defaults = {
        "gender": "Female", "senior": "No", "partner": "Yes", "dependents": "Yes",
        "tenure": 60, "phone": "Yes", "multi": "Yes", "internet": "DSL",
        "sec": "Yes", "backup": "Yes", "protect": "Yes", "tech": "Yes",
        "tv": "No", "movies": "No", "contract": "Two year",
        "paperless": "No", "payment": "Credit card (automatic)", "monthly": 55.0, "total": 3300.0
    }
elif preset == "⚖️ Moderate Risk":
    defaults = {
        "gender": "Female", "senior": "No", "partner": "No", "dependents": "No",
        "tenure": 18, "phone": "Yes", "multi": "No", "internet": "Fiber optic",
        "sec": "No", "backup": "Yes", "protect": "No", "tech": "No",
        "tv": "No", "movies": "Yes", "contract": "One year",
        "paperless": "Yes", "payment": "Bank transfer (automatic)", "monthly": 78.0, "total": 1404.0
    }

# Tabs for Single Customer vs Batch CSV
tab1, tab2 = st.tabs(["👤 Single Customer Prediction", "📂 Batch CSV Prediction"])

with tab1:
    st.subheader("1. Customer Profile Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics & Phone**")
        gender = st.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(defaults["gender"]))
        senior = st.selectbox("Senior Citizen", ["No", "Yes"], index=["No", "Yes"].index(defaults["senior"]))
        partner = st.selectbox("Partner", ["No", "Yes"], index=["No", "Yes"].index(defaults["partner"]))
        dependents = st.selectbox("Dependents", ["No", "Yes"], index=["No", "Yes"].index(defaults["dependents"]))
        tenure = st.slider("Tenure (Months)", 0, 72, int(defaults["tenure"]))
        phone = st.selectbox("Phone Service", ["Yes", "No"], index=["Yes", "No"].index(defaults["phone"]))
        multi = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(defaults["multi"]))

    with col2:
        st.markdown("**Internet & Add-on Services**")
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(defaults["internet"]))
        sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["sec"]))
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["backup"]))
        protect = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["protect"]))
        tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["tech"]))
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["tv"]))
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["movies"]))

    with col3:
        st.markdown("**Contract & Financial Details**")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(defaults["contract"]))
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=["Yes", "No"].index(defaults["paperless"]))
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ], index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(defaults["payment"]))
        
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, float(defaults["monthly"]), 0.50)
        suggested_total = round(float(monthly * max(tenure, 1)), 2)
        total = st.number_input("Total Charges ($)", 0.0, 15000.0, float(defaults["total"]), 5.0,
                               help=f"Estimated based on tenure x monthly: ~${suggested_total:,.2f}")

    st.markdown("---")
    predict_btn = st.button("🔮 Calculate Churn Risk", type="primary", use_container_width=True)

    if predict_btn:
        row = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multi,
            "InternetService": internet,
            "OnlineSecurity": sec,
            "OnlineBackup": backup,
            "DeviceProtection": protect,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }])

        try:
            model = load_model(MODEL_FILES[model_name])
            prob = float(model.predict_proba(row)[0, 1])
            churn_predicted = prob >= threshold

            st.markdown("### 2. Risk Assessment Summary")
            r_col1, r_col2, r_col3 = st.columns([1.2, 1.2, 1.6])

            with r_col1:
                st.metric("Churn Probability", f"{prob:.1%}")
                st.progress(min(max(prob, 0.0), 1.0))

            with r_col2:
                if prob >= 0.70:
                    st.markdown("<p style='margin-bottom:4px;'>Risk Classification</p><span class='risk-badge-high'>CRITICAL CHURN RISK</span>", unsafe_allow_html=True)
                elif prob >= threshold:
                    st.markdown("<p style='margin-bottom:4px;'>Risk Classification</p><span class='risk-badge-high'>HIGH CHURN RISK</span>", unsafe_allow_html=True)
                elif prob >= 0.30:
                    st.markdown("<p style='margin-bottom:4px;'>Risk Classification</p><span class='risk-badge-med'>MODERATE RISK</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='margin-bottom:4px;'>Risk Classification</p><span class='risk-badge-low'>LOW RISK (LOYAL)</span>", unsafe_allow_html=True)
                
                status_text = "Predicted to CHURN" if churn_predicted else "Predicted to STAY"
                st.write(f"Decision: **{status_text}** (threshold: {threshold:.0%})")

            with r_col3:
                # Identify key risk indicators from input
                risk_drivers = []
                positive_drivers = []

                if contract == "Month-to-month":
                    risk_drivers.append("Month-to-month contract (low switching barrier)")
                else:
                    positive_drivers.append(f"{contract} contract (committed lock-in)")

                if payment == "Electronic check":
                    risk_drivers.append("Electronic check payment (highest churn correlation)")
                elif "automatic" in payment:
                    positive_drivers.append("Automated payment setup")

                if internet == "Fiber optic" and (sec == "No" or tech == "No"):
                    risk_drivers.append("Fiber optic without Security/Tech Support add-ons")

                if tenure <= 6:
                    risk_drivers.append(f"Early customer lifecycle (tenure = {tenure} months)")
                elif tenure >= 24:
                    positive_drivers.append(f"Established customer tenure ({tenure} months)")

                if monthly > 80.0:
                    risk_drivers.append(f"High monthly bill (${monthly:.2f}/mo)")

                st.markdown("**Top Influencing Signals:**")
                if risk_drivers:
                    for d in risk_drivers[:3]:
                        st.markdown(f"🔴 <small>{d}</small>", unsafe_allow_html=True)
                if positive_drivers:
                    for p in positive_drivers[:2]:
                        st.markdown(f"🟢 <small>{p}</small>", unsafe_allow_html=True)

            # Recommendations Section
            st.markdown("### 3. Prescriptive Retention Action Plan")
            rec_cols = st.columns(3)
            with rec_cols[0]:
                st.markdown("""
                <div class='recommendation-box'>
                    <strong>🎯 Contract Incentive</strong><br>
                    Offer a 10–15% promotional discount in exchange for migrating from Month-to-month to a 1-year or 2-year contract.
                </div>
                """, unsafe_allow_html=True)
            with rec_cols[1]:
                st.markdown("""
                <div class='recommendation-box'>
                    <strong>💳 Payment Optimization</strong><br>
                    Provide a one-time $10 credit to switch from manual electronic checks to automated credit card / bank transfer.
                </div>
                """, unsafe_allow_html=True)
            with rec_cols[2]:
                st.markdown("""
                <div class='recommendation-box'>
                    <strong>🛡️ Value-Add Bundle</strong><br>
                    Bundle complimentary Online Security & Device Protection for 6 months to improve fiber optic customer stickiness.
                </div>
                """, unsafe_allow_html=True)

            with st.expander("🔍 View Raw Features Submitted"):
                st.dataframe(row.T.rename(columns={0: "Submitted Value"}), use_container_width=True)

        except Exception as e:
            st.error(f"Error executing prediction: {str(e)}")

with tab2:
    st.subheader("Batch Customer Processing")
    st.markdown("Upload a CSV file containing multiple customer records to predict churn probabilities en masse.")

    sample_template = pd.DataFrame([{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 75.0, "TotalCharges": 900.0
    }])

    st.download_button(
        label="📥 Download CSV Template",
        data=sample_template.to_csv(index=False),
        file_name="telco_batch_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded **{len(batch_df)}** records.")

            required_cols = [
                "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
                "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"
            ]

            missing = [c for c in required_cols if c not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns in CSV: {', '.join(missing)}")
            else:
                model = load_model(MODEL_FILES[model_name])
                probs = model.predict_proba(batch_df[required_cols])[:, 1]
                
                results_df = batch_df.copy()
                results_df["Churn_Probability"] = (probs * 100).round(2)
                results_df["Churn_Prediction"] = ["Yes (Churn)" if p >= threshold else "No (Stay)" for p in probs]
                results_df["Risk_Category"] = [
                    "Critical" if p >= 0.70 else "High" if p >= threshold else "Moderate" if p >= 0.30 else "Low"
                    for p in probs
                ]

                # Summary Metrics
                churn_rate = (results_df["Churn_Prediction"] == "Yes (Churn)").mean()
                avg_prob = probs.mean()
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Customers Evaluated", len(results_df))
                m2.metric("Projected Churn Rate", f"{churn_rate:.1%}")
                m3.metric("Average Churn Probability", f"{avg_prob:.1%}")

                st.dataframe(results_df, use_container_width=True)

                csv_output = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Scored Customer List (CSV)",
                    data=csv_output,
                    file_name="telco_churn_scored_results.csv",
                    mime="text/csv",
                    type="primary"
                )
        except Exception as ex:
            st.error(f"Failed to process CSV file: {ex}")
