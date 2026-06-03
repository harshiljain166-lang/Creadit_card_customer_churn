import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnLens · Credit Card Attrition Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME CONSTANTS
# ─────────────────────────────────────────────
BG       = "#0a0e1a"
BG2      = "#111827"
BG3      = "#1e293b"
BORDER   = "#334155"
BLUE     = "#3b82f6"
INDIGO   = "#6366f1"
GREEN    = "#22c55e"
RED      = "#ef4444"
ORANGE   = "#f97316"
TEXT     = "#e2e8f0"
MUTED    = "#64748b"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color=TEXT, size=11),
    margin=dict(l=20, r=20, t=40, b=20),
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: #0a0e1a; color: #e2e8f0; }
.stApp { background-color: #0a0e1a; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1628 0%, #111827 100%);
    border-right: 1px solid #1e293b;
}

h1 { font-size: 2rem !important; font-weight: 800 !important; letter-spacing: -0.03em; }
h2 { font-size: 1.3rem !important; font-weight: 700 !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important; }

[data-testid="metric-container"] {
    background: #111827; border: 1px solid #1e293b;
    border-radius: 12px; padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"]  { color: #64748b !important; font-size: 0.72rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stMetricValue"]  { font-family: 'DM Mono', monospace; font-size: 1.7rem !important; color: #f1f5f9 !important; }

.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    padding: 0.65rem 2.2rem !important; letter-spacing: 0.02em;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important; transition: all 0.2s ease !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(99,102,241,0.45) !important; }

div[data-baseweb="select"] { background-color: #1e293b !important; border-radius: 8px !important; border-color: #334155 !important; }
div[data-baseweb="select"] * { color: #e2e8f0 !important; }
.stNumberInput input { background-color: #1e293b !important; border-color: #334155 !important; border-radius: 8px !important; color: #e2e8f0 !important; }

hr { border-color: #1e293b !important; }

.result-box { border-radius: 16px; padding: 2rem 2.4rem; margin-top: 1.5rem; }
.churn-box  { background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.06)); border: 1.5px solid rgba(239,68,68,0.5); }
.retain-box { background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(22,163,74,0.06));  border: 1.5px solid rgba(34,197,94,0.5);  }
.churn-title  { color: #f87171; font-size: 1.5rem; font-weight: 800; margin-bottom: 0.5rem; }
.retain-title { color: #4ade80; font-size: 1.5rem; font-weight: 800; margin-bottom: 0.5rem; }
.result-body  { color: #94a3b8; font-size: 0.9rem; line-height: 1.7; }

.section-label {
    color: #3b82f6; font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.4rem;
}
.prob-track { background: #1e293b; border-radius: 99px; height: 10px; overflow: hidden; margin-top: 0.6rem; }
.prob-fill-churn  { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #ef4444, #f97316); }
.prob-fill-retain { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #22c55e, #10b981); }

.stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 10px; border: 1px solid #1e293b; }
.stTabs [data-baseweb="tab"] { color: #64748b !important; font-family: 'Syne', sans-serif; font-weight: 600; }
.stTabs [aria-selected="true"] { color: #3b82f6 !important; background: #1e293b; border-radius: 8px; }

.viz-card {
    background: #111827; border: 1px solid #1e293b; border-radius: 14px;
    padding: 1.2rem 1.4rem; margin-bottom: 1.2rem;
}
.viz-title { color: #94a3b8; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.6rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "naive.joblib")
    return joblib.load(path)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_err = str(e)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 ChurnLens")
    st.markdown("<span style='color:#64748b;font-size:0.8rem'>Credit Card Attrition Intelligence</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div class='section-label'>Model Status</div>", unsafe_allow_html=True)
    if model_loaded:
        st.success("✅ Naive Bayes model loaded")
    else:
        st.error(f"❌ {model_err}")
    st.markdown("---")
    st.markdown("<div class='section-label'>About</div>", unsafe_allow_html=True)
    st.markdown("""
<span style='color:#64748b;font-size:0.82rem;line-height:1.8'>
Predict credit card customer churn using Gaussian Naive Bayes.
<br><br>
<a href='https://www.kaggle.com/datasets/whenamancodes/credit-card-customers-prediction'
target='_blank' style='color:#3b82f6'>📊 Kaggle Dataset</a>
</span>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<span style='color:#64748b;font-size:0.78rem'>10,127 customers · 22 features · Binary target</span>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# Credit Card Attrition Predictor")
st.markdown("<span style='color:#64748b;font-size:0.92rem'>Enter the customer profile and run a prediction — live visualizations update instantly.</span>", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯  Predict & Analyze", "🌐  3D Visualizations", "📖  Field Guide"])


# ══════════════════════════════════════════════
# TAB 1 — PREDICT & ANALYZE
# ══════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Demographics ─────────────────────────
    st.markdown("<div class='section-label'>Customer Demographics</div>", unsafe_allow_html=True)
    d1,d2,d3,d4 = st.columns(4)
    with d1: CLIENTNUM           = st.number_input("Client Number",  min_value=700000000, max_value=900000000, value=768805383, step=1)
    with d2: Customer_Age        = st.slider("Age", 18, 80, 45)
    with d3: Gender              = st.selectbox("Gender", ["M","F"])
    with d4: Dependent_count     = st.slider("Dependents", 0, 5, 2)

    d5,d6,d7,d8 = st.columns(4)
    with d5: Education_Level     = st.selectbox("Education Level",  ["Uneducated","High School","College","Graduate","Post-Graduate","Doctorate","Unknown"], index=3)
    with d6: Marital_Status      = st.selectbox("Marital Status",   ["Single","Married","Divorced","Unknown"], index=1)
    with d7: Income_Category     = st.selectbox("Income Category",  ["Less than $40K","$40K - $60K","$60K - $80K","$80K - $120K","$120K +","Unknown"], index=1)
    with d8: Card_Category       = st.selectbox("Card Category",    ["Blue","Silver","Gold","Platinum"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Account Info ─────────────────────────
    st.markdown("<div class='section-label'>Account Information</div>", unsafe_allow_html=True)
    a1,a2,a3,a4 = st.columns(4)
    with a1: Months_on_book             = st.slider("Months on Book",          13, 56, 36)
    with a2: Total_Relationship_Count   = st.slider("Relationship Count",        1,  6,  3)
    with a3: Months_Inactive_12_mon     = st.slider("Months Inactive (12mo)",    0,  6,  2)
    with a4: Contacts_Count_12_mon      = st.slider("Contacts (12mo)",           0,  6,  2)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Financial Metrics ────────────────────
    st.markdown("<div class='section-label'>Financial Metrics</div>", unsafe_allow_html=True)
    f1,f2,f3 = st.columns(3)
    with f1:
        Credit_Limit          = st.number_input("Credit Limit ($)",        1438.0, 34516.0, 12691.0, 100.0)
        Total_Revolving_Bal   = st.number_input("Total Revolving Bal ($)", 0,      2517,     1162,    50)
    with f2:
        Avg_Open_To_Buy       = st.number_input("Avg Open To Buy ($)",     3.0,    34516.0,  11529.0, 100.0)
        Total_Amt_Chng_Q4_Q1  = st.number_input("Amt Change Q4→Q1",       0.0,    3.4,      0.76,    0.01,  format="%.3f")
    with f3:
        Total_Trans_Amt       = st.number_input("Total Trans Amt ($)",     510,    18484,    4476,    100)
        Total_Trans_Ct        = st.number_input("Total Trans Count",       10,     139,      55,      1)

    f4,f5 = st.columns(2)
    with f4: Total_Ct_Chng_Q4_Q1   = st.number_input("Ct Change Q4→Q1",      0.0, 3.71, 0.60, 0.01, format="%.3f")
    with f5: Avg_Utilization_Ratio  = st.number_input("Avg Utilization Ratio", 0.0, 1.0,  0.276,0.001,format="%.3f")

    with st.expander("⚙️  Advanced: NB Classifier Score Columns"):
        sc1,sc2 = st.columns(2)
        with sc1: NB_score_1 = st.number_input("NB Score 1", 0.0, 1.0, 0.000093, 0.000001, format="%.6f")
        with sc2: NB_score_2 = st.number_input("NB Score 2", 0.0, 1.0, 0.999093, 0.000001, format="%.6f")

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1,3])
    with col_btn:
        predict_clicked = st.button("🔍  Run Prediction", use_container_width=True)

    # ── Encode ───────────────────────────────
    gender_map  = {"M":1,"F":0}
    edu_map     = {"Uneducated":1,"High School":2,"College":3,"Graduate":4,"Post-Graduate":5,"Doctorate":6,"Unknown":0}
    marital_map = {"Single":1,"Married":2,"Divorced":3,"Unknown":0}
    income_map  = {"Less than $40K":1,"$40K - $60K":2,"$60K - $80K":3,"$80K - $120K":4,"$120K +":5,"Unknown":0}
    card_map    = {"Blue":1,"Silver":2,"Gold":3,"Platinum":4}

    if predict_clicked:
        if not model_loaded:
            st.error("Model not loaded.")
        else:
            row = {
                "CLIENTNUM": CLIENTNUM, "Customer_Age": Customer_Age,
                "Gender": gender_map[Gender], "Dependent_count": Dependent_count,
                "Education_Level": edu_map[Education_Level],
                "Marital_Status": marital_map[Marital_Status],
                "Income_Category": income_map[Income_Category],
                "Card_Category": card_map[Card_Category],
                "Months_on_book": Months_on_book,
                "Total_Relationship_Count": Total_Relationship_Count,
                "Months_Inactive_12_mon": Months_Inactive_12_mon,
                "Contacts_Count_12_mon": Contacts_Count_12_mon,
                "Credit_Limit": Credit_Limit, "Total_Revolving_Bal": Total_Revolving_Bal,
                "Avg_Open_To_Buy": Avg_Open_To_Buy, "Total_Amt_Chng_Q4_Q1": Total_Amt_Chng_Q4_Q1,
                "Total_Trans_Amt": Total_Trans_Amt, "Total_Trans_Ct": Total_Trans_Ct,
                "Total_Ct_Chng_Q4_Q1": Total_Ct_Chng_Q4_Q1,
                "Avg_Utilization_Ratio": Avg_Utilization_Ratio,
                "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1": NB_score_1,
                "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2": NB_score_2,
            }
            input_df    = pd.DataFrame([row])
            prediction  = model.predict(input_df)[0]
            proba       = model.predict_proba(input_df)[0]
            churn_prob  = float(proba[1]) if len(proba) > 1 else float(proba[0])
            retain_prob = 1.0 - churn_prob
            risk        = "High" if churn_prob >= 0.7 else ("Medium" if churn_prob >= 0.4 else "Low")

            st.markdown("---")
            st.markdown("<div class='section-label'>Prediction Result</div>", unsafe_allow_html=True)

            m1,m2,m3,m4 = st.columns(4)
            with m1: st.metric("Prediction",           "🔴 Attrited" if prediction==1 else "🟢 Retained")
            with m2: st.metric("Churn Probability",    f"{churn_prob*100:.1f}%")
            with m3: st.metric("Retention Probability",f"{retain_prob*100:.1f}%")
            with m4: st.metric("Risk Level",           risk)

            if prediction == 1:
                st.markdown(f"""
<div class="result-box churn-box">
  <div class="churn-title">⚠️ Attrition Risk Detected</div>
  <div class="result-body">
    This customer has a <b style='color:#f87171'>{churn_prob*100:.1f}% probability</b> of churning.
    Immediate retention strategies are recommended.
  </div>
  <div class="prob-track"><div class="prob-fill-churn" style="width:{int(churn_prob*100)}%"></div></div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="result-box retain-box">
  <div class="retain-title">✅ Customer Likely to Retain</div>
  <div class="result-body">
    This customer has a <b style='color:#4ade80'>{retain_prob*100:.1f}% probability</b> of staying.
    Continue nurturing the relationship.
  </div>
  <div class="prob-track"><div class="prob-fill-retain" style="width:{int(retain_prob*100)}%"></div></div>
</div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Visual Analytics</div>", unsafe_allow_html=True)

            v1, v2 = st.columns(2)

            # ── 1. Animated Gauge ────────────────
            with v1:
                st.markdown("<div class='viz-card'><div class='viz-title'>Churn Risk Gauge</div>", unsafe_allow_html=True)
                gauge_color = RED if churn_prob >= 0.5 else GREEN
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=round(churn_prob * 100, 1),
                    delta={"reference": 50, "valueformat": ".1f"},
                    number={"suffix": "%", "font": {"size": 36, "color": gauge_color}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"size": 10}},
                        "bar":  {"color": gauge_color, "thickness": 0.3},
                        "bgcolor": BG3,
                        "bordercolor": BORDER,
                        "steps": [
                            {"range": [0,  40], "color": "rgba(34,197,94,0.15)"},
                            {"range": [40, 70], "color": "rgba(251,191,36,0.15)"},
                            {"range": [70,100], "color": "rgba(239,68,68,0.15)"},
                        ],
                        "threshold": {"line": {"color": "#fbbf24", "width": 3}, "thickness": 0.75, "value": 50},
                    },
                    title={"text": "Churn Probability", "font": {"size": 13, "color": MUTED}},
                ))
                fig_gauge.update_layout(**PLOTLY_LAYOUT, height=280)
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # ── 2. Donut ─────────────────────────
            with v2:
                st.markdown("<div class='viz-card'><div class='viz-title'>Probability Distribution</div>", unsafe_allow_html=True)
                fig_donut = go.Figure(go.Pie(
                    labels=["Churn", "Retain"],
                    values=[churn_prob, retain_prob],
                    hole=0.6,
                    marker=dict(colors=[RED, GREEN], line=dict(color=BG, width=3)),
                    textinfo="label+percent",
                    textfont=dict(size=12, color=TEXT),
                    pull=[0.04, 0],
                ))
                fig_donut.add_annotation(
                    text=f"{'RISK' if prediction==1 else 'SAFE'}",
                    x=0.5, y=0.5, font_size=18, font_color=RED if prediction==1 else GREEN,
                    showarrow=False, font_family="Syne"
                )
                fig_donut.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=True,
                    legend=dict(orientation="h", x=0.2, y=-0.05, font_color=MUTED))
                st.plotly_chart(fig_donut, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # ── 3. Feature Radar ─────────────────
            st.markdown("<div class='viz-card'><div class='viz-title'>Feature Profile Radar (Normalised 0–1)</div>", unsafe_allow_html=True)
            radar_features = [
                "Customer Age","Credit Limit","Trans Amount","Trans Count",
                "Months on Book","Inactive Months","Contacts","Utilization",
            ]
            raw_vals   = [Customer_Age, Credit_Limit, Total_Trans_Amt, Total_Trans_Ct,
                          Months_on_book, Months_Inactive_12_mon, Contacts_Count_12_mon, Avg_Utilization_Ratio]
            raw_maxes  = [80, 34516, 18484, 139, 56, 6, 6, 1.0]
            norm_vals  = [round(v/m, 3) for v,m in zip(raw_vals, raw_maxes)]
            norm_vals += [norm_vals[0]]   # close polygon
            cats = radar_features + [radar_features[0]]

            # typical churner profile (approximate medians from dataset)
            churn_profile = [0.56, 0.25, 0.23, 0.30, 0.64, 0.50, 0.50, 0.03]
            churn_profile += [churn_profile[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=norm_vals, theta=cats, fill='toself',
                name="This Customer",
                line=dict(color=BLUE, width=2),
                fillcolor=f"rgba(59,130,246,0.15)",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=churn_profile, theta=cats, fill='toself',
                name="Avg Churner Profile",
                line=dict(color=RED, width=1.5, dash="dot"),
                fillcolor=f"rgba(239,68,68,0.08)",
            ))
            fig_radar.update_layout(
                **PLOTLY_LAYOUT, height=360,
                polar=dict(
                    bgcolor=BG2,
                    radialaxis=dict(visible=True, range=[0,1], tickfont_size=8, gridcolor=BORDER, color=MUTED),
                    angularaxis=dict(tickfont_size=10, gridcolor=BORDER, linecolor=BORDER),
                ),
                legend=dict(orientation="h", x=0.2, y=-0.08, font_color=MUTED),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── 4. Horizontal bar – risk factors ─
            st.markdown("<div class='viz-card'><div class='viz-title'>Key Risk Indicator Scores</div>", unsafe_allow_html=True)
            indicators = {
                "Inactivity Rate":         Months_Inactive_12_mon / 6,
                "Contact Frequency":       Contacts_Count_12_mon / 6,
                "Low Trans Velocity":      1 - (Total_Trans_Ct / 139),
                "High Utilization":        Avg_Utilization_Ratio,
                "Low Trans Amount":        1 - min(Total_Trans_Amt / 18484, 1),
                "Credit Usage Gap":        max(0, 1 - (Avg_Open_To_Buy / Credit_Limit)) if Credit_Limit > 0 else 0,
            }
            ind_labels = list(indicators.keys())
            ind_vals   = [round(v*100,1) for v in indicators.values()]
            bar_colors = [RED if v > 60 else (ORANGE if v > 35 else GREEN) for v in ind_vals]

            fig_bar = go.Figure(go.Bar(
                x=ind_vals, y=ind_labels, orientation='h',
                marker=dict(color=bar_colors, line=dict(color=BG, width=1)),
                text=[f"{v:.0f}%" for v in ind_vals],
                textposition="outside", textfont=dict(size=10, color=TEXT),
            ))
            fig_bar.update_layout(
                **PLOTLY_LAYOUT, height=280,
                xaxis=dict(range=[0, 120], showgrid=False, zeroline=False, visible=False),
                yaxis=dict(gridcolor=BORDER, tickfont_size=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── 5. Probability waterfall ─────────
            st.markdown("<div class='viz-card'><div class='viz-title'>Score Decomposition Waterfall</div>", unsafe_allow_html=True)
            base = 0.5
            contributions = {
                "Base Rate":         0.0,
                "Inactivity":        (Months_Inactive_12_mon - 2) * 0.025,
                "Contact Burden":    (Contacts_Count_12_mon  - 2) * 0.02,
                "Low Transactions":  (50 - Total_Trans_Ct)        * 0.003,
                "Utilization":       (Avg_Utilization_Ratio - 0.3) * 0.15,
                "Account Age":       (36 - Months_on_book)        * 0.002,
            }
            wf_labels = list(contributions.keys())
            wf_vals   = list(contributions.values())
            wf_colors = [RED if v >= 0 else GREEN for v in wf_vals]
            wf_colors[0] = BLUE

            fig_wf = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute"] + ["relative"]*(len(wf_vals)-1),
                x=wf_labels, y=[base] + wf_vals[1:],
                connector=dict(line=dict(color=BORDER, width=1.5)),
                decreasing=dict(marker_color=GREEN),
                increasing=dict(marker_color=RED),
                totals=dict(marker_color=BLUE),
                text=[f"{v:+.3f}" if i>0 else f"{v:.2f}" for i,v in enumerate([base]+wf_vals[1:])],
                textposition="outside",
                textfont=dict(size=9, color=TEXT),
            ))
            fig_wf.update_layout(
                **PLOTLY_LAYOUT, height=300,
                xaxis=dict(tickfont_size=9),
                yaxis=dict(title="Churn Score", gridcolor=BORDER, tickfont_size=9),
            )
            st.plotly_chart(fig_wf, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── 6. Bullet Charts – key metrics ───
            st.markdown("<div class='viz-card'><div class='viz-title'>Customer vs. Population Benchmarks</div>", unsafe_allow_html=True)
            benchmarks = [
                ("Trans Count",  Total_Trans_Ct,       [10,40,80,139],   "transactions"),
                ("Credit Limit", Credit_Limit/1000,    [1,10,20,34.5],   "k USD"),
                ("Trans Amount", Total_Trans_Amt/1000, [0.5,3,8,18.5],   "k USD"),
                ("Months Active",Months_on_book,       [13,24,36,56],    "months"),
            ]
            fig_bullet = make_subplots(rows=4, cols=1, subplot_titles=[b[0] for b in benchmarks],
                                       vertical_spacing=0.12)
            for i, (label, val, ranges, unit) in enumerate(benchmarks, 1):
                fig_bullet.add_trace(go.Bar(
                    x=[ranges[-1]], y=[label], orientation='h',
                    marker_color="rgba(239,68,68,0.12)", showlegend=False,
                ), row=i, col=1)
                fig_bullet.add_trace(go.Bar(
                    x=[ranges[2]], y=[label], orientation='h',
                    marker_color="rgba(251,191,36,0.18)", showlegend=False,
                ), row=i, col=1)
                fig_bullet.add_trace(go.Bar(
                    x=[ranges[1]], y=[label], orientation='h',
                    marker_color="rgba(34,197,94,0.25)", showlegend=False,
                ), row=i, col=1)
                fig_bullet.add_trace(go.Bar(
                    x=[val], y=[label], orientation='h',
                    marker_color=BLUE, width=0.4, showlegend=False,
                    text=f"{val:.1f} {unit}", textposition="outside",
                    textfont=dict(size=9, color=TEXT),
                ), row=i, col=1)

            fig_bullet.update_layout(**PLOTLY_LAYOUT, height=380, barmode="overlay",
                showlegend=False)
            for i in range(1,5):
                fig_bullet.update_xaxes(showgrid=False, zeroline=False, visible=False, row=i, col=1)
                fig_bullet.update_yaxes(showgrid=False, zeroline=False, visible=False, row=i, col=1)
            st.plotly_chart(fig_bullet, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — 3D VISUALIZATIONS
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>3D Risk Space · Animated</div>", unsafe_allow_html=True)
    st.markdown("<span style='color:#64748b;font-size:0.85rem'>Explore churn risk in three-dimensional feature space. Drag to rotate, scroll to zoom.</span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Synthetic population for 3D context
    np.random.seed(42)
    n = 300
    syn_age    = np.random.randint(25, 70, n)
    syn_trans  = np.random.randint(10, 139, n)
    syn_util   = np.random.uniform(0, 1, n)
    syn_inactive = np.random.randint(0, 6, n)
    syn_credit = np.random.uniform(1500, 34000, n)
    syn_amt    = np.random.randint(500, 18000, n)
    # synthetic churn: high inactivity + low transactions + high util → churn
    syn_churn_score = (syn_inactive/6)*0.4 + (1 - syn_trans/139)*0.35 + syn_util*0.25
    syn_labels = (syn_churn_score > 0.45).astype(int)

    label_colors_pop = [f"rgba(239,68,68,0.7)" if l==1 else f"rgba(34,197,94,0.7)" for l in syn_labels]
    label_names_pop  = ["Attrited" if l==1 else "Retained" for l in syn_labels]

    # Current customer dot
    cur_x = float(Customer_Age)
    cur_y = float(Total_Trans_Ct)
    cur_z = float(Avg_Utilization_Ratio)

    # ── 3D Scatter — Age × Transactions × Utilization ──
    st.markdown("<div class='viz-card'><div class='viz-title'>3D Risk Scatter: Age · Trans Count · Utilization</div>", unsafe_allow_html=True)
    fig3d_1 = go.Figure()

    for label_val, color, name in [(0, GREEN, "Retained"), (1, RED, "Attrited")]:
        mask = syn_labels == label_val
        fig3d_1.add_trace(go.Scatter3d(
            x=syn_age[mask], y=syn_trans[mask], z=syn_util[mask],
            mode='markers',
            name=name,
            marker=dict(
                size=4,
                color=color,
                opacity=0.55,
                line=dict(width=0),
            ),
            text=[f"Age:{a} Trans:{t} Util:{u:.2f}" for a,t,u in
                  zip(syn_age[mask], syn_trans[mask], syn_util[mask])],
            hoverinfo="text+name",
        ))

    fig3d_1.add_trace(go.Scatter3d(
        x=[cur_x], y=[cur_y], z=[cur_z],
        mode='markers+text',
        name="This Customer",
        text=["◀ YOU"],
        textfont=dict(color=TEXT, size=11),
        textposition="middle right",
        marker=dict(size=12, color=BLUE, symbol="diamond",
                    line=dict(color=TEXT, width=2)),
    ))

    fig3d_1.update_layout(
        **PLOTLY_LAYOUT,
        height=520,
        scene=dict(
            xaxis=dict(title="Age", backgroundcolor=BG2, gridcolor=BORDER, color=MUTED),
            yaxis=dict(title="Transaction Count", backgroundcolor=BG2, gridcolor=BORDER, color=MUTED),
            zaxis=dict(title="Utilization Ratio", backgroundcolor=BG2, gridcolor=BORDER, color=MUTED),
            bgcolor=BG2,
        ),
        legend=dict(orientation="h", x=0.2, y=0.0, font_color=MUTED),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
    )
    st.plotly_chart(fig3d_1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 3D Surface — Risk Landscape ───────────
    st.markdown("<div class='viz-card'><div class='viz-title'>3D Risk Surface: Inactivity × Utilization → Churn Score</div>", unsafe_allow_html=True)
    inact_range = np.linspace(0, 6, 40)
    util_range  = np.linspace(0, 1, 40)
    II, UU = np.meshgrid(inact_range, util_range)
    ZZ = (II/6)*0.4 + UU*0.3 + np.random.normal(0, 0.02, II.shape)
    ZZ = np.clip(ZZ, 0, 1)

    fig3d_surf = go.Figure(go.Surface(
        x=inact_range, y=util_range, z=ZZ,
        colorscale=[
            [0.0, "rgba(34,197,94,0.9)"],
            [0.4, "rgba(251,191,36,0.9)"],
            [0.7, "rgba(249,115,22,0.9)"],
            [1.0, "rgba(239,68,68,0.95)"],
        ],
        opacity=0.82,
        showscale=True,
        colorbar=dict(title="Risk Score", tickfont=dict(color=MUTED, size=9), x=1.02),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor=TEXT, project_z=True)
        ),
    ))

    # customer point on surface
    cust_z_surf = (Months_Inactive_12_mon/6)*0.4 + Avg_Utilization_Ratio*0.3
    fig3d_surf.add_trace(go.Scatter3d(
        x=[Months_Inactive_12_mon], y=[Avg_Utilization_Ratio], z=[cust_z_surf + 0.05],
        mode='markers+text', text=["You"], textfont=dict(color=TEXT, size=11),
        textposition="top center",
        marker=dict(size=10, color=BLUE, symbol="diamond",
                    line=dict(color=TEXT, width=2)),
        name="This Customer",
    ))

    fig3d_surf.update_layout(
        **PLOTLY_LAYOUT,
        height=520,
        scene=dict(
            xaxis=dict(title="Months Inactive", backgroundcolor=BG2, gridcolor=BORDER, color=MUTED),
            yaxis=dict(title="Utilization Ratio", backgroundcolor=BG2, gridcolor=BORDER, color=MUTED),
            zaxis=dict(title="Churn Score", backgroundcolor=BG2, gridcolor=BORDER, color=MUTED),
            bgcolor=BG2,
        ),
        scene_camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
    )
    st.plotly_chart(fig3d_surf, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 3D Bar — Income × Card × Churn Rate ───
    st.markdown("<div class='viz-card'><div class='viz-title'>3D Bar: Income × Card Category × Synthetic Churn Rate</div>", unsafe_allow_html=True)
    income_cats = ["<$40K","$40-60K","$60-80K","$80-120K","$120K+"]
    card_cats   = ["Blue","Silver","Gold","Platinum"]
    np.random.seed(7)
    churn_rates_3d = np.array([
        [0.22, 0.18, 0.14, 0.10],
        [0.19, 0.15, 0.11, 0.08],
        [0.16, 0.12, 0.09, 0.07],
        [0.13, 0.10, 0.08, 0.05],
        [0.10, 0.08, 0.06, 0.04],
    ])

    fig3d_bar = go.Figure()
    colorscale_bar = px.colors.sequential.Reds
    for i, inc in enumerate(income_cats):
        for j, card in enumerate(card_cats):
            val = churn_rates_3d[i, j]
            fig3d_bar.add_trace(go.Scatter3d(
                x=[i], y=[j], z=[0, val],
                mode='lines',
                line=dict(
                    color=f"rgba({int(239*val/0.25)},{int(68*(1-val/0.25))},68,0.9)",
                    width=12,
                ),
                showlegend=False,
                hovertext=f"{inc} / {card}: {val*100:.0f}% churn",
                hoverinfo="text",
            ))
            fig3d_bar.add_trace(go.Scatter3d(
                x=[i], y=[j], z=[val],
                mode='markers',
                marker=dict(size=6, color=f"rgba(255,100,100,0.9)"),
                showlegend=False,
                hovertext=f"{inc} / {card}: {val*100:.0f}%",
                hoverinfo="text",
            ))

    fig3d_bar.update_layout(
        **PLOTLY_LAYOUT,
        height=520,
        scene=dict(
            xaxis=dict(title="Income Category", tickvals=list(range(5)),
                       ticktext=income_cats, backgroundcolor=BG2, gridcolor=BORDER, color=MUTED),
            yaxis=dict(title="Card Category", tickvals=list(range(4)),
                       ticktext=card_cats, backgroundcolor=BG2, gridcolor=BORDER, color=MUTED),
            zaxis=dict(title="Churn Rate", backgroundcolor=BG2, gridcolor=BORDER, color=MUTED),
            bgcolor=BG2,
        ),
        scene_camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
    )
    st.plotly_chart(fig3d_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Animated Bubble — Trans Amt × Count over Age ─
    st.markdown("<div class='viz-card'><div class='viz-title'>Animated Bubble: Transaction Profile by Age Group</div>", unsafe_allow_html=True)
    np.random.seed(99)
    age_groups = [f"{a}-{a+9}" for a in range(20, 70, 10)]
    rows = []
    for frame_i, ag in enumerate(age_groups):
        for _ in range(60):
            is_churn = np.random.rand() < (0.1 + frame_i*0.03)
            rows.append({
                "Age Group": ag,
                "Trans Amount": max(200, np.random.normal(3000 + frame_i*300, 1200)),
                "Trans Count":  max(5,  int(np.random.normal(55 - frame_i*2, 15))),
                "Utilization":  max(0,  min(1, np.random.normal(0.2 + frame_i*0.04, 0.15))),
                "Status": "Attrited" if is_churn else "Retained",
            })
    df_anim = pd.DataFrame(rows)

    fig_anim = px.scatter(
        df_anim, x="Trans Amount", y="Trans Count",
        animation_frame="Age Group", animation_group="Status",
        size="Utilization", color="Status",
        color_discrete_map={"Attrited": RED, "Retained": GREEN},
        size_max=30, range_x=[0, 18000], range_y=[0, 140],
        hover_data=["Utilization"],
        template="plotly_dark",
    )
    fig_anim.update_layout(
        **PLOTLY_LAYOUT, height=420,
        xaxis=dict(gridcolor=BORDER, title="Transaction Amount ($)"),
        yaxis=dict(gridcolor=BORDER, title="Transaction Count"),
        legend=dict(orientation="h", y=-0.12, font_color=MUTED),
    )
    st.plotly_chart(fig_anim, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Parallel Coordinates ──────────────────
    st.markdown("<div class='viz-card'><div class='viz-title'>Parallel Coordinates: Multidimensional Feature View</div>", unsafe_allow_html=True)
    pc_n = 200
    np.random.seed(55)
    pc_df = pd.DataFrame({
        "Age":         np.random.randint(25, 70, pc_n),
        "Credit Limit":np.random.uniform(1500, 34000, pc_n) / 1000,
        "Trans Amt":   np.random.uniform(500, 18000, pc_n) / 1000,
        "Trans Ct":    np.random.randint(10, 139, pc_n),
        "Utilization": np.random.uniform(0, 1, pc_n),
        "Inactive Mo": np.random.randint(0, 6, pc_n),
        "Churn":       np.random.choice([0, 1], pc_n, p=[0.84, 0.16]),
    })

    fig_pc = go.Figure(go.Parcoords(
        line=dict(color=pc_df["Churn"], colorscale=[[0, GREEN],[1, RED]],
                  showscale=True,
                  colorbar=dict(title="Churn", tickvals=[0,1], ticktext=["No","Yes"],
                                tickfont_color=MUTED, x=1.02)),
        dimensions=[
            dict(range=[25,70],  label="Age",          values=pc_df["Age"]),
            dict(range=[1.5,34], label="Credit ($k)",  values=pc_df["Credit Limit"]),
            dict(range=[0.5,18], label="Trans ($k)",   values=pc_df["Trans Amt"]),
            dict(range=[10,139], label="Trans Ct",     values=pc_df["Trans Ct"]),
            dict(range=[0,1],    label="Utilization",  values=pc_df["Utilization"]),
            dict(range=[0,6],    label="Inactive Mo",  values=pc_df["Inactive Mo"]),
        ],
        labelangle=-20,
        labelside="top",
        labelfont=dict(color=MUTED, size=10),
        tickfont=dict(color=MUTED, size=8),
        rangefont=dict(color=MUTED, size=8),
    ))
    fig_pc.update_layout(**PLOTLY_LAYOUT, height=400)
    st.plotly_chart(fig_pc, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — FIELD GUIDE
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📖 Feature Reference Guide")

    guide = {
        "Field": ["CLIENTNUM","Customer_Age","Gender","Dependent_count","Education_Level","Marital_Status",
                  "Income_Category","Card_Category","Months_on_book","Total_Relationship_Count",
                  "Months_Inactive_12_mon","Contacts_Count_12_mon","Credit_Limit","Total_Revolving_Bal",
                  "Avg_Open_To_Buy","Total_Amt_Chng_Q4_Q1","Total_Trans_Amt","Total_Trans_Ct",
                  "Total_Ct_Chng_Q4_Q1","Avg_Utilization_Ratio"],
        "Description": [
            "Unique customer identifier","Age of the customer","Customer gender (M/F)",
            "Number of dependents","Highest education level","Marital status",
            "Annual household income bracket","Credit card tier",
            "Months with the bank","Number of products held",
            "Inactive months in past 12","Contacts with bank in 12 months",
            "Total credit limit","Revolving balance on card",
            "Open-to-buy credit available","Trans amount change Q4→Q1",
            "Total transaction amount (12mo)","Total transaction count (12mo)",
            "Trans count change Q4→Q1","Card utilization ratio",
        ],
        "Type": ["ID","Numeric","Categorical","Numeric","Categorical","Categorical",
                 "Categorical","Categorical","Numeric","Numeric","Numeric","Numeric",
                 "Numeric","Numeric","Numeric","Numeric","Numeric","Numeric","Numeric","Numeric"],
        "Risk Signal": ["–","Low","Low","Low","Medium","Low","Medium","Medium","Medium","Medium",
                        "HIGH","HIGH","Medium","Medium","Low","HIGH","HIGH","HIGH","HIGH","HIGH"],
    }
    df_guide = pd.DataFrame(guide)
    st.dataframe(df_guide, use_container_width=True, hide_index=True,
        column_config={
            "Field":       st.column_config.TextColumn("Field",       width=200),
            "Description": st.column_config.TextColumn("Description", width=380),
            "Type":        st.column_config.TextColumn("Type",        width=100),
            "Risk Signal": st.column_config.TextColumn("Risk Signal", width=100),
        })

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🤖 Model Details")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Algorithm:** Gaussian Naive Bayes  \n**Library:** scikit-learn 1.6.1  \n**Features:** 22  \n**Classes:** 0 = Retained, 1 = Attrited")
    with c2:
        st.markdown("**Dataset:** [Kaggle Credit Card Customers](https://www.kaggle.com/datasets/whenamancodes/credit-card-customers-prediction)  \n**Rows:** 10,127  \n**Encoding:** Label Encoding  \n**Smoothing:** var_smoothing = 1e-9")
