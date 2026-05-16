dashboard_code = '''
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

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.main{background:#F0F7FB}
.header-box{background:linear-gradient(135deg,#021B2E,#065A82);padding:22px 28px;border-radius:12px;margin-bottom:20px;border-left:6px solid #0FA3B1}
.header-title{color:white;font-size:26px;font-weight:800;margin:0}
.header-sub{color:#9fc8db;font-size:13px;margin-top:5px}
.result-pass{background:linear-gradient(135deg,#059669,#34D399);color:white;border-radius:14px;padding:24px;text-align:center;box-shadow:0 8px 24px rgba(5,150,105,0.3)}
.result-fail{background:linear-gradient(135deg,#DC2626,#F87171);color:white;border-radius:14px;padding:24px;text-align:center;box-shadow:0 8px 24px rgba(220,38,38,0.3)}
.result-label{font-size:54px;font-weight:900;letter-spacing:3px;margin:0;line-height:1}
.result-conf{font-size:17px;margin-top:8px;opacity:0.9}
.metric-card{background:white;border-radius:10px;padding:14px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.07);border-top:3px solid #0FA3B1}
.metric-val{font-size:24px;font-weight:800;color:#065A82}
.metric-lbl{font-size:11px;color:#64748B;margin-top:3px}
.section-title{font-size:16px;font-weight:700;color:#065A82;margin-bottom:10px;padding-bottom:5px;border-bottom:2px solid #D4EEF7}
</style>
""", unsafe_allow_html=True)

PROFILES = {
    "🔴  Ali — At Risk Student": {
        "desc": "Low grades, frequent absences, 2 past failures",
        "G1":6,"G2":7,"absences":18,"studytime":1,"failures":2,"famrel":2
    },
    "🟡  Jasur — Borderline Student": {
        "desc": "Average grades, some absences, 1 past failure",
        "G1":10,"G2":11,"absences":8,"studytime":2,"failures":1,"famrel":3
    },
    "🟢  Barno — Strong Student": {
        "desc": "High grades, low absences, no failures",
        "G1":15,"G2":16,"absences":2,"studytime":3,"failures":0,"famrel":5
    },
    "✏️  Custom Student": {
        "desc": "Set your own values using the sliders",
        "G1":10,"G2":10,"absences":5,"studytime":2,"failures":0,"famrel":3
    },
}

KEY_LABELS = {
    "G1":"G1 — 1st Period Grade","G2":"G2 — 2nd Period Grade",
    "absences":"Absences (days)","studytime":"Study Time (1-4)",
    "failures":"Past Failures","famrel":"Family Relations (1-5)"
}

st.markdown("""
<div class="header-box">
    <p class="header-title">🎓 AI-Based Student Performance Predictor</p>
    <p class="header-sub">SHAP Explainability + What-If Simulator &nbsp;|&nbsp; Azizbek Boboqulov · ID: 221198 &nbsp;|&nbsp; CAU · May 2026</p>
</div>
""", unsafe_allow_html=True)

col_left, col_main = st.columns([1, 2.4])

with col_left:
    st.markdown('<p class="section-title">👤 Student Profile</p>', unsafe_allow_html=True)
    selected = st.selectbox("Select student:", list(PROFILES.keys()), label_visibility="collapsed")
    profile  = PROFILES[selected]
    st.info(f"**{selected[:3]}** {profile['desc']}")

    st.markdown('<p class="section-title" style="margin-top:14px">🎛️ Adjust Parameters</p>', unsafe_allow_html=True)
    G1        = st.slider("G1 — 1st Period Grade",         0, 20, profile["G1"])
    G2        = st.slider("G2 — 2nd Period Grade",         0, 20, profile["G2"])
    absences  = st.slider("Absences (days)",               0, 93, profile["absences"])
    studytime = st.slider("Study Time (1=<2h · 4=>10h)",  1,  4, profile["studytime"])
    failures  = st.slider("Past Failures",                 0,  4, profile["failures"])
    famrel    = st.slider("Family Relations (1=bad · 5=excellent)", 1, 5, profile["famrel"])

    st.markdown("---")
    st.markdown('<p class="section-title">💡 Quick Alerts</p>', unsafe_allow_html=True)
    if absences > 15: st.warning("⚠️ High absences — major risk")
    if failures > 0:  st.error(f"🔴 {failures} past failure(s) — high risk")
    if G2 >= 14:      st.success("✅ Strong G2 — positive factor")
    if studytime == 1:st.info("📚 Low study time — try to increase")

with col_main:
    default_vals = {f: 0 for f in feature_names}
    default_vals.update({"G1":G1,"G2":G2,"absences":absences,"studytime":studytime,"failures":failures,"famrel":famrel})
    input_df = pd.DataFrame([default_vals])[feature_names]
    input_sc = scaler.transform(input_df)

    proba   = model.predict_proba(input_sc)[0][1]
    pred    = "PASS" if proba > 0.5 else "FAIL"
    risk    = "🟢 Low Risk" if proba > 0.75 else "🟡 Medium Risk" if proba > 0.5 else "🔴 High Risk"
    css_cls = "result-pass" if proba > 0.5 else "result-fail"

    st.markdown('<p class="section-title">📊 Prediction Result</p>', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f"""
        <div class="{css_cls}">
            <p class="result-label">{pred}</p>
            <p class="result-conf">{proba:.1%} confidence</p>
            <p style="font-size:14px;margin-top:4px;opacity:0.85;font-weight:600">{risk}</p>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{proba:.1%}</div><div class="metric-lbl">Pass Probability</div></div>', unsafe_allow_html=True)
    with r3:
        st.markdown('<div class="metric-card"><div class="metric-val">87.34%</div><div class="metric-lbl">Model Accuracy</div></div>', unsafe_allow_html=True)
    with r4:
        st.markdown('<div class="metric-card"><div class="metric-val">0.94</div><div class="metric-lbl">AUC-ROC Score</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    shap_vals = explainer.shap_values(input_sc)
    sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

    KEY_FEATS = ["G1","G2","absences","studytime","failures","famrel"]
    shap_dict = {f: sv[feature_names.index(f)] for f in KEY_FEATS if f in feature_names}
    shap_sorted = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    KEY_VALUES  = {"G1":G1,"G2":G2,"absences":absences,"studytime":studytime,"failures":failures,"famrel":famrel}

    col_shap, col_summary = st.columns([1.6, 1])

    with col_shap:
        st.markdown('<p class="section-title">🔍 SHAP — Why This Prediction?</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        fig.patch.set_facecolor("#F0F7FB")
        ax.set_facecolor("#F0F7FB")
        labels = [KEY_LABELS[f] for f, _ in shap_sorted]
        values = [v for _, v in shap_sorted]
        colors = ["#059669" if v > 0 else "#DC2626" for v in values]
        bars   = ax.barh(labels, values, color=colors, height=0.52, edgecolor="white", linewidth=1)
        ax.axvline(x=0, color="#021B2E", linewidth=1.5)
        ax.set_xlabel("SHAP Value  (Green = PASS · Red = FAIL risk)", fontsize=10.5, color="#021B2E")
        ax.set_title("Feature Impact on Prediction", fontsize=12, fontweight="bold", color="#065A82", pad=8)
        ax.tick_params(axis="y", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, values):
            sign = "+" if val >= 0 else ""
            ax.text(val + (0.002 if val >= 0 else -0.002),
                    bar.get_y() + bar.get_height()/2,
                    f"{sign}{val:.3f}", va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=10, fontweight="bold",
                    color="#059669" if val > 0 else "#DC2626")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_summary:
        st.markdown('<p class="section-title">📋 Input Summary</p>', unsafe_allow_html=True)
        for feat, val in shap_sorted:
            label = KEY_LABELS[feat]
            fval  = KEY_VALUES[feat]
            sign  = "+" if val > 0 else ""
            color = "#059669" if val > 0 else "#DC2626"
            arrow = "▲" if val > 0 else "▼"
            st.markdown(f"""
            <div style="background:white;border-radius:8px;padding:9px 12px;margin-bottom:7px;
                        border-left:4px solid {color};box-shadow:0 1px 4px rgba(0,0,0,0.06)">
                <div style="font-size:12px;font-weight:700;color:#021B2E">{label}</div>
                <div style="display:flex;justify-content:space-between;margin-top:3px">
                    <span style="font-size:17px;font-weight:800;color:#065A82">{fval}</span>
                    <span style="font-size:12px;font-weight:700;color:{color}">{arrow} {sign}{val:.3f}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">📝 Educator Recommendation</p>', unsafe_allow_html=True)
    top_pos = [(f,v) for f,v in shap_sorted if v > 0][:2]
    top_neg = [(f,v) for f,v in shap_sorted if v < 0][:2]
    explain = f"**Prediction: {pred}** with **{proba:.1%} confidence**. "
    if top_pos:
        explain += "Strengths: " + ", ".join([f"**{KEY_LABELS[f]}** (+{v:.3f})" for f,v in top_pos]) + ". "
    if top_neg:
        explain += "Risk factors: " + ", ".join([f"**{KEY_LABELS[f]}** ({v:.3f})" for f,v in top_neg]) + ". "
    if pred == "FAIL":
        explain += "⚠️ **Action needed:** Address the risk factors above immediately."
    else:
        explain += "✅ **Good standing:** Continue monitoring red factors."
    st.info(explain)
'''

with open("dashboard.py", "w") as f:
    f.write(dashboard_code)

from google.colab import files
files.download("dashboard.py")
print("✅ dashboard.py downloaded! Now upload to GitHub.")
