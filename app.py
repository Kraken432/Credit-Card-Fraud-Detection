import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open("best_model.pkl", "rb") as f:
    data = pickle.load(f)
model, features, model_name = data["model"], data["features"], data["name"]

AMOUNT_MEAN, AMOUNT_STD = 88.3496, 250.120109
TIME_MEAN, TIME_STD = 94813.8, 47488.14

@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\Yasee\Downloads\creditcard.csv")

def build_and_predict(amount, time, v_vals):
    amount_scaled = (amount - AMOUNT_MEAN) / AMOUNT_STD
    time_scaled = (time - TIME_MEAN) / TIME_STD
    row = {**v_vals,
           "Amount_scaled": amount_scaled,
           "Time_scaled": time_scaled,
           "Hour": int(abs(time_scaled) * 24) % 24,
           "Amount_per_sec": amount_scaled / (abs(time_scaled) + 1),
           "V1_V2_interaction": v_vals["V1"] * v_vals["V2"]}
    arr = np.array([[row[f] for f in features]])
    pred = model.predict(arr)[0]
    prob = model.predict_proba(arr)[0][1]
    return int(pred), round(prob * 100, 2)

st.set_page_config(page_title="Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")
st.caption(f"Model: {model_name}")
st.divider()

tab1, tab2 = st.tabs(["🎲 Random Transaction", "✏️ Custom Input"])

with tab1:
    col1, col2 = st.columns(2)
    if col1.button("🎲 Pick Random Transaction"):
        st.session_state.row = load_data().sample(1).iloc[0]
    if col2.button("🚨 Pick Random Fraud"):
        df = load_data()
        st.session_state.row = df[df["Class"] == 1].sample(1).iloc[0]

    if "row" in st.session_state:
        row = st.session_state.row
        v_vals = {f"V{i}": row[f"V{i}"] for i in range(1, 29)}
        pred, prob = build_and_predict(row["Amount"], row["Time"], v_vals)

        st.metric("Amount", f"₹{row['Amount']:.2f}")
        st.metric("Fraud Probability", f"{prob}%")

        if pred == 1:
            st.error("🚨 FRAUDULENT Transaction")
        else:
            st.success("✅ LEGITIMATE Transaction")

        actual = int(row["Class"])
        st.info(f"Actual label: {'Fraud 🚨' if actual == 1 else 'Legit ✅'} — {'✔️ Correct' if pred == actual else '❌ Wrong'}")

with tab2:
    amount = st.number_input("Transaction Amount (₹)", min_value=0.01, value=150.0)
    hour = st.slider("Hour of Day", 0, 23, 14)

    if st.button("🔍 Predict"):
        v_vals = {f"V{i}": 0.0 for i in range(1, 29)}  # neutral PCA values
        pred, prob = build_and_predict(amount, float(hour * 3600), v_vals)

        st.metric("Fraud Probability", f"{prob}%")
        if pred == 1:
            st.error("🚨 FRAUDULENT Transaction")
        else:
            st.success("✅ LEGITIMATE Transaction")

        st.caption("Note: V1–V28 are anonymized bank features. Without real data, they default to zero (neutral average).")
