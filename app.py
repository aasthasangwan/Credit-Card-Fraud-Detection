import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load models
rf_model = joblib.load("model_rf.pkl")
nn_model = load_model("model_nn.h5")

st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("🔍 Fraud Detection System")
st.markdown("Upload transaction data and detect potential frauds using ML models.")

uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])
model_choice = st.selectbox("Select Model", ["Random Forest", "Neural Network"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Normalize amount and time
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df['Time'] = scaler.fit_transform(df[['Time']])
    X = df.drop("Class", axis=1) if "Class" in df.columns else df

    st.write("📊 Preview of Uploaded Data:")
    st.dataframe(X.head())

    if st.button("🔎 Predict"):
        if model_choice == "Random Forest":
            preds = rf_model.predict(X)
        else:
            preds = nn_model.predict(X)
            preds = (preds > 0.5).astype("int32").flatten()

        df['Prediction'] = ["Fraud ❌" if p == 1 else "Legit ✅" for p in preds]
        st.success("✅ Prediction Completed!")
        st.dataframe(df[['Amount', 'Time', 'Prediction']].head(20))

        fraud_count = sum(preds)
        st.metric("🚨 Detected Fraudulent Transactions", fraud_count)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results", data=csv, file_name="fraud_results.csv", mime="text/csv")
