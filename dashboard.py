import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Telecom Customer Churn Predictor")
st.caption("Enter customer details and get a live prediction from the FastAPI model.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Information")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    region = st.selectbox(
        "Region",
        ["Bangalore", "Chennai", "Delhi", "Kolkata", "Mumbai"]
    )

    plan_type = st.selectbox(
        "Plan Type",
        ["Postpaid", "Prepaid"]
    )

    contract_type = st.selectbox(
        "Contract Type",
        ["Monthly", "Annual", "Two-Year"]
    )

    tenure_months = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=120,
        value=12
    )

    device_age_months = st.number_input(
        "Device Age (months)",
        min_value=0,
        max_value=120,
        value=12
    )

    satisfaction_score = st.slider(
        "Satisfaction Score",
        min_value=1,
        max_value=10,
        value=7,
        step=1
    )

with col2:
    st.subheader("Usage & Account Details")

    data_used_gb = st.number_input(
        "Data Used (GB)",
        min_value=0.0,
        max_value=200.0,
        value=10.0
    )

    call_minutes = st.number_input(
        "Call Minutes",
        min_value=0,
        max_value=5000,
        value=200
    )

    calls_made = st.number_input(
        "Calls Made",
        min_value=0,
        max_value=1000,
        value=50
    )

    sms_count = st.number_input(
        "SMS Count",
        min_value=0,
        max_value=2000,
        value=100
    )

    revenue_inr = st.number_input(
        "Revenue (INR)",
        min_value=0.0,
        max_value=100000.0,
        value=500.0
    )

    complaints_6m = st.number_input(
        "Complaints (6 months)",
        min_value=0,
        max_value=50,
        value=0
    )

    network_issues = st.number_input(
        "Network Issues",
        min_value=0,
        max_value=50,
        value=0
    )

    customer_service_calls = st.number_input(
        "Customer Service Calls",
        min_value=0,
        max_value=50,
        value=0
    )

    payment_delay = st.number_input(
        "Payment Delay",
        min_value=0,
        max_value=365,
        value=0
    )

    roaming_usage = st.number_input(
        "Roaming Usage",
        min_value=0,
        max_value=1000,
        value=0
    )

    international_calls = st.number_input(
        "International Calls",
        min_value=0,
        max_value=1000,
        value=0
    )

    auto_payment = st.checkbox(
        "Auto Payment Enabled"
    )


if st.button(
    "🔮 Predict Churn",
    type="primary",
    width="stretch"
):

    payload = {
        "age": age,
        "gender": gender,
        "region": region,
        "plan_type": plan_type,
        "tenure_months": tenure_months,
        "data_used_gb": data_used_gb,
        "call_minutes": call_minutes,
        "calls_made": calls_made,
        "sms_count": sms_count,
        "revenue_inr": revenue_inr,
        "complaints_6m": complaints_6m,
        "network_issues": network_issues,
        "customer_service_calls": customer_service_calls,
        "payment_delay": payment_delay,
        "roaming_usage": roaming_usage,
        "international_calls": international_calls,
        "device_age_months": device_age_months,
        "satisfaction_score": satisfaction_score,
        "contract_type": contract_type,
        "auto_payment": int(auto_payment)
    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=10
        )

        response.raise_for_status()

        result = response.json()

        probability = result["churn_probability"]
        prediction = result["churn_prediction"]

        st.subheader("Prediction Result")

        st.metric(
            "Churn Probability",
            f"{probability * 100:.1f}%"
        )

        if prediction == 1:
            st.error("⚠️ Prediction: LIKELY TO CHURN")
        else:
            st.success("✅ Prediction: LIKELY TO STAY")

    except requests.exceptions.RequestException as e:

        st.error(
            "Could not reach the FastAPI prediction service."
        )

        st.code(str(e))
