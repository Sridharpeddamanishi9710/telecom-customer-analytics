import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD MODELS
# =========================

churn_model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
segmentation_model = joblib.load("models/segmentation_model.pkl")

st.set_page_config(page_title="Telecom Customer Analytics", layout="wide")

st.title("📊 Telecom Customer Analytics System")

menu = ["Churn Prediction", "Customer Segmentation", "Dashboard"]
choice = st.sidebar.selectbox("Select Option", menu)

# =========================
# CHURN PREDICTION
# =========================

if choice == "Churn Prediction":

    st.header("🔮 Customer Churn Prediction")

    tenure = st.number_input("Tenure (Months)", min_value=0.0)
    monthlycharges = st.number_input("Monthly Charges", min_value=0.0)
    totalcharges = st.number_input("Total Charges", min_value=0.0)

    if st.button("Predict Churn"):

        data = pd.DataFrame({
            "tenure": [tenure],
            "MonthlyCharges": [monthlycharges],
            "TotalCharges": [totalcharges]
        })

        data_scaled = scaler.transform(data)

        prediction = churn_model.predict(data_scaled)
        probability = churn_model.predict_proba(data_scaled)

        churn_prob = probability[0][1] * 100

        st.subheader(f"Churn Probability: {churn_prob:.2f}%")

        st.progress(int(churn_prob))

        if prediction[0] == 1:
            st.error("⚠ Customer is likely to Churn")
        else:
            st.success("✅ Customer will Stay")


# =========================
# CUSTOMER SEGMENTATION
# =========================

elif choice == "Customer Segmentation":

    st.header("👥 Customer Segmentation")

    tenure = st.number_input("Tenure")
    monthlycharges = st.number_input("Monthly Charges")
    totalcharges = st.number_input("Total Charges")

    # Segment Meaning Mapping
    segment_meaning = {
        0: "🆕 New Customer",
        1: "⭐ Loyal Customer",
        2: "⚠ At Risk Customer",
        3: "💰 High Value Customer"
    }

    if st.button("Find Segment"):

        data = pd.DataFrame({
            "tenure": [tenure],
            "MonthlyCharges": [monthlycharges],
            "TotalCharges": [totalcharges]
        })

        # Apply same scaling used in training
        data_scaled = scaler.transform(data)

        segment = segmentation_model.predict(data_scaled)

        seg = int(segment[0])

        meaning = segment_meaning.get(seg, "Unknown Segment")

        st.success(f"Customer belongs to Segment {seg}")

        st.info(f"Segment Meaning: {meaning}")


# =========================
# DASHBOARD
# =========================

elif choice == "Dashboard":

    st.header("📈 Telecom Data Dashboard")

    df = pd.read_csv("data/Telco-Customer-Churn.csv")

    # ================= KPI METRICS =================

    st.subheader("Key Business Metrics")

    total_customers = df.shape[0]

    churn_rate = (df["Churn"].value_counts(normalize=True)["Yes"]) * 100

    avg_revenue = df["MonthlyCharges"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", total_customers)
    col2.metric("Churn Rate", f"{churn_rate:.2f}%")
    col3.metric("Avg Monthly Revenue", f"${avg_revenue:.2f}")

    # ================= DATA PREVIEW =================

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # ================= CHARTS =================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Churn Distribution")

        fig, ax = plt.subplots()

        df["Churn"].value_counts().plot(kind="bar", ax=ax)

        ax.set_xlabel("Churn")
        ax.set_ylabel("Count")

        st.pyplot(fig)

    with col2:

        st.subheader("Monthly Charges Distribution")

        fig2, ax2 = plt.subplots()

        sns.histplot(df["MonthlyCharges"], bins=30, ax=ax2)

        st.pyplot(fig2)

    # ================= SCATTER PLOT =================

    st.subheader("Tenure vs Monthly Charges")

    fig3, ax3 = plt.subplots()

    sns.scatterplot(
        x="tenure",
        y="MonthlyCharges",
        hue="Churn",
        data=df,
        ax=ax3
    )

    st.pyplot(fig3)

    # ================= FEATURE IMPORTANCE =================

    st.subheader("Feature Importance for Churn Prediction")

    try:

        importances = churn_model.feature_importances_

        features = ["Tenure", "MonthlyCharges", "TotalCharges"]

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        })

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        )

        fig4, ax4 = plt.subplots()

        sns.barplot(
            x="Importance",
            y="Feature",
            data=importance_df,
            ax=ax4
        )

        st.pyplot(fig4)

    except:

        st.info("Feature importance not available for this model.")