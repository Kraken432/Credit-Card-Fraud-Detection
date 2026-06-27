import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open("best_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
name = data["name"]
features = data["features"]

AMOUNT_MEAN, AMOUNT_STD = 88.3496, 250.120109
TIME_MEAN, TIME_STD = 94813.8, 47488.14

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.markdown(f"**Model:** {name}")
st.markdown("---")

amount = st.number_input("Transaction Amount (₹)", min_value=0.0, max_value=999999.0, value=100.0, step=0.01)

if st.button("🔍 Predict"):
    df_orig = pd.read_csv(r"C:\Users\Yasee\Downloads\creditcard.csv")
    df_orig['amount_diff'] = abs(df_orig['Amount'] - amount)
    closest = df_orig.loc[df_orig['amount_diff'].idxmin()]
    actual_label = int(closest['Class'])
    actual_amount = closest['Amount']

    raw = {f"V{i}": closest[f"V{i}"] for i in range(1, 29)}
    amount_scaled = (closest['Amount'] - AMOUNT_MEAN) / AMOUNT_STD
    time_scaled = (closest['Time'] - TIME_MEAN) / TIME_STD
    hour = int(abs(time_scaled) * 24) % 24
    amount_per_sec = amount_scaled / (abs(time_scaled) + 1)
    v1_v2_interaction = raw['V1'] * raw['V2']

    input_dict = {f"V{i}": raw[f"V{i}"] for i in range(1, 29)}
    input_dict.update({"Amount_scaled": amount_scaled, "Time_scaled": time_scaled,
                       "Hour": hour, "Amount_per_sec": amount_per_sec,
                       "V1_V2_interaction": v1_v2_interaction})
    input_array = np.array([[input_dict[f] for f in features]])

    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error("🚨 FRAUDULENT Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    st.metric("Fraud Probability", f"{probability*100:.2f}%")
    st.caption(f"📌 Closest match in dataset: ₹{actual_amount:.2f} — Actual: {'Fraud 🚨' if actual_label == 1 else 'Legit ✅'}")

    if prediction == actual_label:
        st.info("✔️ Prediction matches actual label")
    else:
        st.warning("⚠️ Prediction does not match actual label")