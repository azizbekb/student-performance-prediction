import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model         = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
scaler        = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(BASE_DIR, "feature_names.pkl"), "rb"))
explainer     = shap.TreeExplainer(model)

st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("AI-Based Student Performance Prediction")
st.caption("SHAP Explainability + What-If Simulator | Azizbek Boboqulov | ID: 221198")
st.divider()

st.sidebar.header("Student Parameters")
G1        = st.sidebar.slider("G1 - First period grade",   0, 20, 10)
G2        = st.sidebar.slider("G2 - Second period grade",  0, 20, 10)
absences  = st.sidebar.slider("Absences",                  0, 93,  5)
studytime = st.sidebar.slider("Study time (1-4)",          1,  4,  2)
failures  = st.sidebar.slider("Past failures",             0,  4,  0)
famrel    = st.sidebar.slider("Family relationship (1-5)", 1,  5,  3)
freetime  = st.sidebar.slider("Free time (1-5)",           1,  5,  3)
goout     = st.sidebar.slider("Going out (1-5)",           1,  5,  3)
health    = st.sidebar.slider("Health (1-5)",              1,  5,  3)
Medu      = st.sidebar.slider("Mother education (0-4)",    0,  4,  2)
Fedu      = st.sidebar.slider("Father education (0-4)",    0,  4,  2)

# Build input
default_vals = {f: 0 for f in feature_names}
default_vals.update({
    "G1": G1, "G2": G2, "absences": absences,
    "studytime": studytime, "failures": failures,
    "famrel": famrel, "freetime": freetime,
    "goout": goout, "health": health,
    "Medu": Medu, "Fedu": Fedu
})

input_df = pd.DataFrame([default_vals])[feature_names]
input_sc = scaler.transform(input_df)

# Prediction
proba = model.predict_proba(input_sc)[0][1]
pred  = "PASS" if proba > 0.5 else "FAIL"
risk  = "Low" if proba > 0.7 else "Medium" if proba > 0.5 else "High"

st.markdown(f"### Prediction: {pred}")
st.markdown(f"### Pass Probability: {proba:.1%}")
st.markdown(f"### Risk Level: {risk}")

if proba > 0.5:
    st.success(f"Student is predicted to PASS with {proba:.1%} confidence")
else:
    st.error(f"Student is at risk of FAILING — {proba:.1%} pass probability")

st.divider()
st.subheader("SHAP Explanation — Why this prediction?")

raw = explainer.shap_values(input_sc)
if isinstance(raw, list):
    sv_vals = np.array(raw[1]).flatten()
else:
    sv_vals = np.array(raw).flatten()

if len(sv_vals) != len(feature_names):
    sv_vals = sv_vals[:len(feature_names)]

shap_df = pd.DataFrame({
    "Feature": feature_names,
    "Impact":  sv_vals
}).sort_values("Impact", key=abs, ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = ["#059669" if v > 0 else "#DC2626" for v in shap_df["Impact"]]
ax.barh(shap_df["Feature"], shap_df["Impact"], color=bar_colors)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("SHAP Value")
ax.set_title("Top 10 Feature Impacts", fontweight="bold")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
st.pyplot(fig)
plt.close()

st.divider()
st.subheader("Top 5 Factors")
top5 = pd.DataFrame({
    "Feature": feature_names,
    "Impact":  sv_vals
}).sort_values("Impact", key=abs, ascending=False).head(5)

for _, row in top5.iterrows():
    icon      = "✅" if row["Impact"] > 0 else "⚠️"
    direction = "increases pass chance" if row["Impact"] > 0 else "increases fail risk"
    st.write(f"{icon} **{row['Feature']}**: {row['Impact']:+.3f} — {direction}")
