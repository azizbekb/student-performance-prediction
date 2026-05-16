dashboard_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model         = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"),  "rb"))
scaler        = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(BASE_DIR, "feature_names.pkl"), "rb"))
explainer     = shap.TreeExplainer(model)

st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
.main { background: #F0F7FB; }
.block-container { padding: 1.2rem 2rem 2rem 2rem; max-width: 1200px; }
.app-header { background: linear-gradient(135deg, #021B2E 0%, #065A82 100%); padding: 20px 28px 16px 28px; border-radius: 14px; border-left: 7px solid #0FA3B1; margin-bottom: 20px; }
.app-header h1 { color: #fff; font-size: 24px; font-weight: 800; margin: 0 0 4px 0; }
.app-header p  { color: #9fc8db; font-size: 13px; margin: 0; }
.card { background: white; border-radius: 12px; padding: 18px 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.07); margin-bottom: 14px; }
.card-title { font-size: 14px; font-weight: 700; color: #065A82; text-transform: uppercase; letter-spacing: .6px; border-bottom: 2px solid #D4EEF7; padding-bottom: 7px; margin-bottom: 14px; }
.profile-info { background: #F0F7FB; border-radius: 8px; padding: 10px 14px; border-left: 4px solid #0FA3B1; font-size: 13px; color: #1a1a2e; }
.res-pass { background: linear-gradient(135deg, #059669 0%, #10B981 100%); border-radius: 14px; padding: 22px 18px 18px 18px; text-align: center; box-shadow: 0 6px 20px rgba(5,150,105,.3); color: white; }
.res-fail { background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%); border-radius: 14px; padding: 22px 18px 18px 18px; text-align: center; box-shadow: 0 6px 20px rgba(220,38,38,.3); color: white; }
.res-verdict { font-size: 50px; font-weight: 900; letter-spacing: 3px; line-height: 1; }
.res-conf { font-size: 17px; opacity: .9; margin-top: 6px; }
.res-risk { font-size: 13px; font-weight: 700; margin-top: 5px; opacity: .85; }
.metric { background: white; border-radius: 10px; padding: 12px 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,.06); border-top: 3px solid #0FA3B1; }
.metric-v { font-size: 22px; font-weight: 800; color: #065A82; }
.metric-l { font-size: 11px; color: #64748B; margin-top: 2px; }
.feat-row { display: flex; align-items: center; justify-content: space-between; background: white; border-radius: 8px; padding: 9px 13px; margin-bottom: 7px; border-left: 4px solid #ccc; box-shadow: 0 1px 4px rgba(0,0,0,.05); }
.feat-name  { font-size: 12.5px; font-weight: 600; color: #1a1a2e; }
.feat-value { font-size: 18px; font-weight: 800; color: #065A82; }
.feat-shap  { font-size: 12px; font-weight: 700; }
.alert-box  { border-radius: 8px; padding: 9px 14px; font-size: 13px; font-weight: 500; margin-bottom: 8px; }
.alert-warn  { background:#FEF9C3; color:#854D0E; border-left:4px solid #EAB308; }
.alert-error { background:#FEE2E2; color:#991B1B; border-left:4px solid #DC2626; }
.alert-good  { background:#DCFCE7; color:#166534; border-left:4px solid #059669; }
.alert-info  { background:#E0F2FE; color:#075985; border-left:4px solid #0FA3B1; }
.rec-box { background: #EFF6FF; border: 1.5px solid #BFDBFE; border-radius: 10px; padding: 14px 18px; font-size: 13.5px; color: #1e3a5f; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

KEY_FEATS  = ["G1", "G2", "absences", "studytime", "failures"]
KEY_LABELS = {
    "G1":"G1 — 1st Period Grade (0–20)",
    "G2":"G2 — 2nd Period Grade (0–20)",
    "absences":"School Absences (days)",
    "studytime":"Weekly Study Time (1=<2h, 4=>10h)",
    "failures":"Number of Past Failures",
}
KEY_SHORT = {"G1":"G1 Grade","G2":"G2 Grade","absences":"Absences","studytime":"Study Time","failures":"Failures"}
PROFILES = {
    "🔴  Ali — High Risk":      {"desc":"Very low grades · 18 absences · 2 past failures · rarely studies","G1":6,"G2":7,"absences":18,"studytime":1,"failures":2},
    "🟡  Jasur — Borderline":   {"desc":"Average grades · 8 absences · 1 past failure · moderate study","G1":10,"G2":11,"absences":8,"studytime":2,"failures":1},
    "🟢  Barno — Strong":       {"desc":"High grades · 2 absences · no failures · studies regularly","G1":15,"G2":16,"absences":2,"studytime":3,"failures":0},
    "✏️  Custom Student":       {"desc":"Set your own values using the sliders below","G1":10,"G2":10,"absences":5,"studytime":2,"failures":0},
}

def get_shap(input_sc):
    raw = explainer.shap_values(input_sc)
    if isinstance(raw, list):
        return np.array(raw[1]).flatten()
    return np.array(raw).flatten()

def build_input(vals):
    d = {f: 0 for f in feature_names}
    d.update(vals)
    df = pd.DataFrame([d])[feature_names]
    return scaler.transform(df)

st.markdown("""
<div class="app-header">
  <h1>🎓 AI-Based Student Performance Predictor</h1>
  <p>SHAP Explainability &nbsp;+&nbsp; What-If Simulator &nbsp;|&nbsp; Azizbek Boboqulov · ID: 221198 &nbsp;|&nbsp; CAU Engineering School · May 2026</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 2.2], gap="large")

with left:
    st.markdown('<div class="card"><div class="card-title">👤 Student Profile</div>', unsafe_allow_html=True)
    sel     = st.selectbox("", list(PROFILES.keys()), label_visibility="collapsed")
    profile = PROFILES[sel]
    st.markdown(f\'<div class="profile-info">📌 {profile["desc"]}</div>\', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(\'<div class="card"><div class="card-title">🎛️ Adjust Parameters</div>\', unsafe_allow_html=True)
    G1        = st.slider("G1 — 1st Period Grade",             0, 20, profile["G1"])
    G2        = st.slider("G2 — 2nd Period Grade",             0, 20, profile["G2"])
    absences  = st.slider("School Absences (days)",            0, 93, profile["absences"])
    studytime = st.slider("Study Time  (1=<2h · 4=>10h)",     1,  4, profile["studytime"])
    failures  = st.slider("Past Failures",                     0,  4, profile["failures"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(\'<div class="card"><div class="card-title">💡 Quick Alerts</div>\', unsafe_allow_html=True)
    alerts = []
    if absences > 15:   alerts.append(("error", f"⚠️ {absences} absences — major risk factor"))
    if failures > 1:    alerts.append(("error", f"🔴 {failures} past failures — significant risk"))
    elif failures == 1: alerts.append(("warn",  "🟡 1 past failure — moderate risk"))
    if G2 < 10:         alerts.append(("error", f"📉 G2 = {G2} — below passing threshold"))
    elif G2 >= 14:      alerts.append(("good",  f"✅ G2 = {G2} — strong positive predictor"))
    if studytime == 1:  alerts.append(("info",  "📚 Increase study time to improve prediction"))
    if G1 > 0 and G2 > G1 + 2: alerts.append(("good", f"📈 Grade improving: G1={G1} → G2={G2}"))
    if not alerts:      alerts.append(("info",  "ℹ️ No critical risk factors detected"))
    css_map = {"error":"alert-error","warn":"alert-warn","good":"alert-good","info":"alert-info"}
    for kind, msg in alerts:
        st.markdown(f\'<div class="alert-box {css_map[kind]}">{msg}</div>\', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    vals     = {"G1":G1,"G2":G2,"absences":absences,"studytime":studytime,"failures":failures}
    input_sc = build_input(vals)
    proba    = float(model.predict_proba(input_sc)[0][1])
    pred     = "PASS" if proba > 0.5 else "FAIL"
    css_cls  = "res-pass" if proba > 0.5 else "res-fail"
    risk_lbl = ("🟢 Low Risk" if proba > 0.75 else "🟡 Medium Risk" if proba > 0.5 else "🟠 High Risk" if proba > 0.3 else "🔴 Very High Risk")

    r1, r2, r3, r4 = st.columns([1.4, 1, 1, 1])
    with r1:
        st.markdown(f\'\'\'<div class="{css_cls}"><div class="res-verdict">{pred}</div><div class="res-conf">{proba:.1%} confidence</div><div class="res-risk">{risk_lbl}</div></div>\'\'\', unsafe_allow_html=True)
    with r2:
        st.markdown(f\'<div class="metric"><div class="metric-v">{proba:.1%}</div><div class="metric-l">Pass Probability</div></div>\', unsafe_allow_html=True)
    with r3:
        st.markdown(\'<div class="metric"><div class="metric-v">87.34%</div><div class="metric-l">Model Accuracy</div></div>\', unsafe_allow_html=True)
    with r4:
        st.markdown(\'<div class="metric"><div class="metric-v">0.94</div><div class="metric-l">AUC-ROC</div></div>\', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    sv_all     = get_shap(input_sc)
    shap_dict  = {}
    for f in KEY_FEATS:
        if f in feature_names:
            idx = feature_names.index(f)
            if idx < len(sv_all):
                shap_dict[f] = float(sv_all[idx])
    shap_sorted = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    col_chart, col_sum = st.columns([1.6, 1], gap="medium")

    with col_chart:
        st.markdown(\'<div class="card"><div class="card-title">🔍 SHAP — Why This Prediction?</div>\', unsafe_allow_html=True)
        if shap_sorted:
            labels = [KEY_SHORT[f] for f,_ in shap_sorted]
            values = [v for _,v in shap_sorted]
            colors = ["#059669" if v>0 else "#DC2626" for v in values]
            fig, ax = plt.subplots(figsize=(6.5, 3.5))
            fig.patch.set_facecolor("white"); ax.set_facecolor("white")
            bars = ax.barh(labels, values, color=colors, height=0.5, edgecolor="white", linewidth=1.5)
            ax.axvline(0, color="#021B2E", linewidth=1.5, zorder=5)
            xlim = max(abs(v) for v in values) * 1.55
            ax.set_xlim(-xlim, xlim)
            ax.set_xlabel("SHAP Value  (← FAIL  |  PASS →)", fontsize=9.5, color="#4A5568")
            ax.set_title("Feature Contributions to Prediction", fontsize=12, fontweight="bold", color="#065A82", pad=8)
            ax.tick_params(axis="y", labelsize=11, colors="#1a1a2e")
            ax.tick_params(axis="x", labelsize=9,  colors="#4A5568")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#E2E8F0"); ax.spines["bottom"].set_color("#E2E8F0")
            ax.set_axisbelow(True); ax.xaxis.grid(True, color="#F0F4F8", linewidth=0.8)
            for bar, val in zip(bars, values):
                sign = "+" if val >= 0 else ""
                off  = xlim * 0.04
                ax.text(val+(off if val>=0 else -off), bar.get_y()+bar.get_height()/2,
                        f"{sign}{val:.3f}", va="center", ha="left" if val>=0 else "right",
                        fontsize=10.5, fontweight="bold", color="#059669" if val>0 else "#DC2626")
            gp = mpatches.Patch(color="#059669", label="Increases PASS chance")
            rp = mpatches.Patch(color="#DC2626", label="Increases FAIL risk")
            ax.legend(handles=[gp,rp], fontsize=9, loc="lower right", framealpha=0.9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_sum:
        st.markdown(\'<div class="card"><div class="card-title">📋 Feature Summary</div>\', unsafe_allow_html=True)
        for feat, sv in shap_sorted:
            fval  = vals.get(feat,"—")
            color = "#059669" if sv>0 else "#DC2626"
            sign  = "+" if sv>0 else ""
            arrow = "▲" if sv>0 else "▼"
            st.markdown(f\'\'\'
            <div class="feat-row" style="border-left-color:{color}">
                <div><div class="feat-name">{KEY_SHORT[feat]}</div><div class="feat-value">{fval}</div></div>
                <div class="feat-shap" style="color:{color}">{arrow} {sign}{sv:.3f}</div>
            </div>\'\'\', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(\'<div class="card"><div class="card-title">📝 Educator Recommendation</div>\', unsafe_allow_html=True)
    pos_f = [(f,v) for f,v in shap_sorted if v>0]
    neg_f = [(f,v) for f,v in shap_sorted if v<0]
    icon  = "✅" if pred=="PASS" else "⚠️"
    rec   = f"{icon} <b>Prediction: {pred}</b> with <b>{proba:.1%} confidence</b> ({risk_lbl})<br><br>"
    if pos_f:
        rec += "<b>Strengths:</b> " + " · ".join([f\'<span style="color:#059669;font-weight:700">{KEY_SHORT[f]}</span> (+{v:.3f})\' for f,v in pos_f]) + "<br>"
    if neg_f:
        rec += "<b>Risk factors:</b> " + " · ".join([f\'<span style="color:#DC2626;font-weight:700">{KEY_SHORT[f]}</span> ({v:.3f})\' for f,v in neg_f]) + "<br><br>"
    if pred=="FAIL":
        top = neg_f[0][0] if neg_f else None
        rec += f\'🔴 <b>Action required:</b> Prioritize improving <b>{KEY_SHORT[top] if top else "key factors"}</b> — it has the largest negative impact.\' if top else "🔴 Immediate academic support recommended."
    else:
        if neg_f:
            rec += f\'🟡 <b>Monitor:</b> Despite passing, <b>{KEY_SHORT[neg_f[0][0]]}</b> is still a risk factor. Address it to improve confidence.\'
        else:
            rec += "🟢 <b>Well on track.</b> All factors are positive. Encourage the student to maintain current performance."
    st.markdown(f\'<div class="rec-box">{rec}</div>\', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
'''

with open("dashboard.py", "w") as f:
    f.write(dashboard_code)

from google.colab import files
files.download("dashboard.py")
print("✅ Professional dashboard ready! Upload to GitHub.")
