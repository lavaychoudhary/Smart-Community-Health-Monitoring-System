import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Community Health Monitoring",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg-base:        #040d1a;
    --bg-card:        #071528;
    --bg-card2:       #0b1e35;
    --accent-cyan:    #00d4ff;
    --accent-teal:    #00f5c4;
    --border:         rgba(0, 212, 255, 0.15);
    --text-primary:   #e8f4fd;
    --text-secondary: #7fa8c9;
    --text-muted:     #3d6080;
    --success:        #00f5c4;
    --danger:         #ff4d6d;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}
.stApp { background: var(--bg-base) !important; }

.main-header {
    background: linear-gradient(135deg, #071528 0%, #0a1f3a 50%, #071528 100%);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.main-header::before {
    content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 50%, rgba(0,212,255,0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(0,245,196,0.04) 0%, transparent 50%);
    pointer-events: none;
}
.main-header h1 {
    font-family: 'Syne', sans-serif !important; font-size: 2.4rem !important; font-weight: 800 !important;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-teal));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0 !important; letter-spacing: -0.5px;
}
.main-header p { color: var(--text-secondary); font-size: 1rem; margin: 0; font-weight: 300; }
.header-badge {
    display: inline-block; background: rgba(0,212,255,0.12); border: 1px solid rgba(0,212,255,0.3);
    color: var(--accent-cyan); font-size: 0.72rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase; padding: 4px 12px;
    border-radius: 20px; margin-bottom: 0.8rem;
}
.metric-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.25rem 1.5rem; position: relative; overflow: hidden;
}
.metric-card::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-teal));
    border-radius: 14px 14px 0 0;
}
.metric-label { font-size: 0.75rem; font-weight: 500; letter-spacing: 1px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.4rem; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 700; color: var(--accent-cyan); line-height: 1; }
.metric-sub   { font-size: 0.78rem; color: var(--text-secondary); margin-top: 0.3rem; }

.section-title {
    font-family: 'Syne', sans-serif; font-size: 1.25rem; font-weight: 700;
    color: var(--text-primary); margin-bottom: 1rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
}
.card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.5rem; margin-bottom: 1rem;
}
.alert-danger {
    background: rgba(255,77,109,0.1); border: 1px solid rgba(255,77,109,0.4);
    border-left: 4px solid var(--danger); border-radius: 10px;
    padding: 1.2rem 1.5rem; color: #ffb3c1;
}
.alert-success {
    background: rgba(0,245,196,0.08); border: 1px solid rgba(0,245,196,0.3);
    border-left: 4px solid var(--success); border-radius: 10px;
    padding: 1.2rem 1.5rem; color: #a0f5e0;
}
.alert-title { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; margin-bottom: 0.3rem; }
.risk-tag {
    display: inline-block; background: rgba(255,107,53,0.15); border: 1px solid rgba(255,107,53,0.4);
    color: #ffb399; font-size: 0.8rem; font-weight: 500; padding: 4px 12px; border-radius: 20px; margin: 3px;
}

/* ── Unsafe condition cards ── */
.unsafe-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 12px;
    margin-bottom: 1.5rem;
}
.unsafe-card {
    background: var(--bg-card);
    border: 1px solid rgba(255,77,109,0.25);
    border-left: 3px solid #ff4d6d;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    position: relative;
}
.unsafe-card.warn {
    border-color: rgba(255,209,102,0.3);
    border-left-color: #ffd166;
}
.unsafe-icon { font-size: 1.4rem; margin-bottom: 0.4rem; display: block; }
.unsafe-param {
    font-family: 'Syne', sans-serif; font-size: 0.88rem; font-weight: 700;
    color: #e8f4fd; margin-bottom: 0.15rem;
}
.unsafe-threshold {
    font-size: 0.75rem; color: #ff9db3; font-weight: 500; margin-bottom: 0.5rem;
}
.unsafe-card.warn .unsafe-threshold { color: #ffd166; }
.unsafe-disease { font-size: 0.8rem; color: var(--text-secondary); line-height: 1.55; }
.unsafe-disease strong { color: #ffb3c1; font-weight: 500; }
.unsafe-card.warn .unsafe-disease strong { color: #ffd166; }
.unsafe-stat {
    position: absolute; top: 0.8rem; right: 1rem;
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: #ff4d6d;
}
.unsafe-card.warn .unsafe-stat { color: #ffd166; }
.unsafe-stat span {
    display: block; font-family: 'DM Sans', sans-serif; font-size: 0.65rem;
    font-weight: 400; color: var(--text-muted); text-align: right;
}
.conditions-intro {
    background: linear-gradient(135deg, rgba(255,77,109,0.07) 0%, rgba(255,107,53,0.04) 100%);
    border: 1px solid rgba(255,77,109,0.18);
    border-radius: 12px; padding: 1.1rem 1.4rem; margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 1rem;
}
.conditions-intro-text h3 {
    font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700;
    color: #ffb3c1; margin: 0 0 0.2rem 0;
}
.conditions-intro-text p { font-size: 0.82rem; color: var(--text-secondary); margin: 0; line-height: 1.5; }

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important; border-radius: 12px !important;
    border: 1px solid var(--border) !important; padding: 4px !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--text-secondary) !important;
    border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important; font-size: 0.9rem !important; padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] { background: rgba(0,212,255,0.15) !important; color: var(--accent-cyan) !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

.stNumberInput input, div[data-baseweb="select"] {
    background: var(--bg-card2) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text-primary) !important;
}
.stNumberInput label, .stSelectbox label { color: var(--text-secondary) !important; font-size: 0.82rem !important; }
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #00f5c4) !important; color: #040d1a !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important; padding: 0.65rem 2rem !important;
}
.filter-bar {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem 1.5rem; margin-bottom: 1.5rem;
}
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Plotly layout helper ───────────────────────────────────────────────────────
# _BASE defines xaxis/yaxis once. lay() merges everything cleanly — no duplicate keys.
_XAXIS = dict(gridcolor="rgba(0,212,255,0.08)", linecolor="rgba(0,212,255,0.15)", zerolinecolor="rgba(0,212,255,0.1)")
_YAXIS = dict(gridcolor="rgba(0,212,255,0.08)", linecolor="rgba(0,212,255,0.15)", zerolinecolor="rgba(0,212,255,0.1)")

_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#7fa8c9", size=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#7fa8c9")),
    colorway=["#00d4ff", "#00f5c4", "#ff6b35", "#ffd166", "#ff4d6d", "#a78bfa"],
    margin=dict(t=50, b=30, l=10, r=10),
)

def lay(title_text, height=300, xaxis_title=None, yaxis_title=None, **kw):
    """
    Build a complete layout dict. xaxis/yaxis are handled here explicitly
    so callers can never accidentally pass a duplicate key.
    """
    xa = dict(**_XAXIS)
    ya = dict(**_YAXIS)
    if xaxis_title is not None:
        xa['title'] = xaxis_title
    if yaxis_title is not None:
        ya['title'] = yaxis_title
    return dict(
        **_BASE,
        title=dict(text=title_text, font=dict(family="Syne", size=15, color="#e8f4fd")),
        height=height,
        xaxis=xa,
        yaxis=ya,
        **kw,
    )


# ─── Load model & data ──────────────────────────────────────────────────────────
@st.cache_resource
def load_ml():
    return joblib.load('water_potability_model.pkl'), joblib.load('scaler.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv('Smart_health_prediction_dataset_1.csv', encoding='latin1')
    # Normalise Lead column name (encoding artefact varies by OS)
    lead_col = [c for c in df.columns if 'Lead' in c][0]
    df = df.rename(columns={lead_col: 'Lead Concentration (µg/L)'})
    return df

model, scaler = load_ml()
df = load_data()


# ─── Unsafe conditions definition ──────────────────────────────────────────────
CONDITIONS = [
    {
        "param":     "Lead Concentration",
        "col":       "Lead Concentration (µg/L)",
        "threshold": "> 10 µg/L  (WHO limit)",
        "op":        lambda s: s > 10,
        "icon":      "🧪",
        "severity":  "danger",
        "disease":   "Causes chronic <strong>lead poisoning</strong>, irreversible neurological damage, kidney failure, and developmental disorders in children.",
    },
    {
        "param":     "Bacteria Count",
        "col":       "Bacteria Count (CFU/mL)",
        "threshold": "> 500 CFU/mL",
        "op":        lambda s: s > 500,
        "icon":      "🦠",
        "severity":  "danger",
        "disease":   "High risk of <strong>Cholera, Typhoid fever</strong>, Dysentery, and acute Diarrheal illness.",
    },
    {
        "param":     "pH — Acidic Water",
        "col":       "pH Level",
        "threshold": "< 6.5 pH",
        "op":        lambda s: s < 6.5,
        "icon":      "⚗️",
        "severity":  "danger",
        "disease":   "Acidic water corrodes pipes, leaching metals. Causes <strong>gastrointestinal irritation</strong> and tooth enamel erosion.",
    },
    {
        "param":     "pH — Alkaline Water",
        "col":       "pH Level",
        "threshold": "> 8.5 pH",
        "op":        lambda s: s > 8.5,
        "icon":      "⚗️",
        "severity":  "warn",
        "disease":   "Bitter taste, skin and eye irritation. May indicate <strong>industrial or agricultural contamination</strong>.",
    },
    {
        "param":     "Nitrate Level",
        "col":       "Nitrate Level (mg/L)",
        "threshold": "> 50 mg/L  (WHO limit)",
        "op":        lambda s: s > 50,
        "icon":      "🌿",
        "severity":  "warn",
        "disease":   "Causes <strong>Methemoglobinemia (Blue Baby Syndrome)</strong> in infants; long-term cancer risk in adults.",
    },
    {
        "param":     "Turbidity",
        "col":       "Turbidity (NTU)",
        "threshold": "> 5 NTU  (WHO limit)",
        "op":        lambda s: s > 5,
        "icon":      "🌊",
        "severity":  "warn",
        "disease":   "Cloudy water conceals pathogens. Linked to <strong>Cryptosporidiosis, Giardia</strong>, and Hepatitis A outbreaks.",
    },
    {
        "param":     "Dissolved Oxygen",
        "col":       "Dissolved Oxygen (mg/L)",
        "threshold": "< 3 mg/L",
        "op":        lambda s: s < 3,
        "icon":      "💨",
        "severity":  "danger",
        "disease":   "Oxygen-depleted stagnant water fosters <strong>anaerobic bacteria</strong> and produces toxic Hydrogen Sulphide gas.",
    },
    {
        "param":     "Contaminant Level",
        "col":       "Contaminant Level (ppm)",
        "threshold": "> 10 ppm",
        "op":        lambda s: s > 10,
        "icon":      "☣️",
        "severity":  "danger",
        "disease":   "General toxic contamination. Depending on substance, causes <strong>organ damage, carcinogenesis</strong>, or acute poisoning.",
    },
]


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="header-badge">AI-Powered Monitoring</div>
    <h1>💧 Smart Community Health Monitoring</h1>
    <p>Real-time water safety analysis &amp; community health surveillance platform</p>
</div>
""", unsafe_allow_html=True)

# ─── KPI strip ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
for col, label, value, sub in zip(
    [k1, k2, k3, k4],
    ["Potable Water Rate", "Avg Bacteria Count", "Average pH Level", "Total Records"],
    [f"{df['Potability'].mean()*100:.1f}%", f"{int(df['Bacteria Count (CFU/mL)'].mean()):,}",
     f"{df['pH Level'].mean():.2f}", f"{len(df):,}"],
    ["of all samples safe", "CFU/mL across dataset", "ideal range 6.5 – 8.5", "community data points"],
):
    with col:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔴  Live Prediction & Alerts", "📊  Community Dashboard"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Prediction + Unsafe Conditions Reference
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Sensor input ──
     
    st.markdown('<div class="section-title">Manual Sensor Data Entry</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:var(--text-secondary);margin-bottom:1.5rem;font-size:0.9rem;">Enter current water parameters to evaluate disease risk and potability.</p>', unsafe_allow_html=True)

    with st.container():
     col1, col2 = st.columns(2)

    with col1:
        ph          = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        turbidity   = st.number_input("Turbidity (NTU)", min_value=0.0, value=2.0, step=0.1)
        bacteria    = st.number_input("Bacteria Count (CFU/mL)", min_value=0.0, value=150.0, step=10.0)
        lead        = st.number_input("Lead Concentration (µg/L)", min_value=0.0, value=5.0, step=0.5)

    with col2:
        nitrate     = st.number_input("Nitrate Level (mg/L)", min_value=0.0, value=10.0, step=1.0)
        do_val      = st.number_input("Dissolved Oxygen (mg/L)", min_value=0.0, value=6.5, step=0.1)
        contaminant = st.number_input("Contaminant Level (ppm)", min_value=0.0, value=2.0, step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)
    evaluate = st.button("🔍  Evaluate Water Safety", use_container_width=True)



    # ── Prediction result ──
    if evaluate:
        inp   = np.array([[contaminant, ph, turbidity, do_val, nitrate, lead, bacteria]])
        pred  = model.predict(scaler.transform(inp))[0]
        proba = model.predict_proba(scaler.transform(inp))[0]

        st.markdown("<br>", unsafe_allow_html=True)
        if pred == 0:
            st.markdown("""<div class="alert-danger">
                <div class="alert-title">🚨 URGENT ALERT — WATER IS NOT POTABLE</div>
                High risk of waterborne disease detected. Immediate remediation required.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="alert-success">
                <div class="alert-title">✅ SYSTEM NORMAL — WATER IS POTABLE</div>
                All parameters within acceptable safety thresholds for consumption.
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        g1, g2 = st.columns(2, gap="large")

        with g1:
            danger_score = round(proba[0] * 100, 1)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=danger_score,
                number=dict(suffix="%", font=dict(color="#e8f4fd", size=30, family="Syne")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#7fa8c9", tickfont=dict(color="#7fa8c9")),
                    bar=dict(color="#ff4d6d" if pred == 0 else "#00f5c4"),
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="rgba(0,212,255,0.2)",
                    steps=[
                        dict(range=[0,  40], color="rgba(0,245,196,0.12)"),
                        dict(range=[40, 70], color="rgba(255,209,102,0.12)"),
                        dict(range=[70,100], color="rgba(255,77,109,0.12)"),
                    ],
                    threshold=dict(line=dict(color="#ffd166", width=2), thickness=0.75, value=70),
                ),
            ))
            fig_gauge.update_layout(**lay("Contamination Risk Score", height=290))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with g2:
            params  = ["pH", "Turbidity", "Bacteria", "Lead", "Nitrate", "DO", "Contaminant"]
            maxvals = [14, 20, 1000, 50, 50, 15, 10]
            vals    = [ph, turbidity, bacteria, lead, nitrate, do_val, contaminant]
            norm    = [min(v / m * 100, 100) for v, m in zip(vals, maxvals)]

            fig_radar = go.Figure(go.Scatterpolar(
                r=norm + [norm[0]], theta=params + [params[0]],
                fill='toself', fillcolor='rgba(0,212,255,0.1)',
                line=dict(color="#00d4ff", width=2), marker=dict(color="#00d4ff", size=5),
            ))
            fig_radar.update_layout(
                **lay("Parameter Profile (normalised %)", height=290),
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 100], color="#3d6080", gridcolor="rgba(0,212,255,0.1)"),
                    angularaxis=dict(color="#7fa8c9", gridcolor="rgba(0,212,255,0.1)"),
                ),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        risks = []
        if bacteria > 500:        risks.append("High Bacterial Contamination — Cholera / Typhoid risk")
        if lead > 10:             risks.append("Toxic Lead Levels — heavy metal poisoning risk")
        if ph < 6.5 or ph > 8.5: risks.append("Dangerous pH Imbalance — corrosion / chemical risk")
        if nitrate > 45:          risks.append("Elevated Nitrate — methemoglobinemia risk")
        if turbidity > 10:        risks.append("High Turbidity — pathogen concealment risk")

        if risks:
            st.markdown('<div class="section-title" style="margin-top:1rem;">⚠️ Identified Health Risks</div>', unsafe_allow_html=True)
            st.markdown("".join(f'<span class="risk-tag">{r}</span>' for r in risks), unsafe_allow_html=True)

        if pred == 0:
            st.info("📲 Automated SMS warning dispatched to local community officials.")

    # ── Unsafe Conditions Reference (always visible in Tab 1) ──────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔬 WHO Safety Thresholds & Health Risks</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="conditions-intro">
        <span style="font-size:2rem;">⚠️</span>
        <div class="conditions-intro-text">
            <h3>Know the Unsafe Limits — Backed by Real Data</h3>
            <p>Each card shows an internationally recognised safety threshold, the percentage of dataset samples that breach it, and the diseases or health consequences linked to that contamination. This evidence helps communities understand exactly why the water quality alerts matter.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cards — stats computed from the full dataset
    cards_html = '<div class="unsafe-grid">'
    for cond in CONDITIONS:
        flagged   = cond["op"](df[cond["col"]]).sum()
        pct       = round(flagged / len(df) * 100, 1)
        cls       = "" if cond["severity"] == "danger" else " warn"
        cards_html += f"""
        <div class="unsafe-card{cls}">
            <div class="unsafe-stat">{pct}%<span>of samples</span></div>
            <span class="unsafe-icon">{cond['icon']}</span>
            <div class="unsafe-param">{cond['param']}</div>
            <div class="unsafe-threshold">Unsafe if {cond['threshold']}</div>
            <div class="unsafe-disease">{cond['disease']}</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # Summary horizontal bar chart
    labels   = [c["param"] for c in CONDITIONS]
    pcts     = [round(c["op"](df[c["col"]]).sum() / len(df) * 100, 1) for c in CONDITIONS]
    bar_clrs = ["#ff4d6d" if c["severity"] == "danger" else "#ffd166" for c in CONDITIONS]

    fig_unsafe = go.Figure(go.Bar(
        x=pcts, y=labels, orientation='h',
        marker=dict(color=bar_clrs, line=dict(width=0)),
        text=[f"{p}%" for p in pcts],
        textposition='outside',
        textfont=dict(color="#7fa8c9", size=11),
        hovertemplate="%{y}<br>%{x}% of samples breach this threshold<extra></extra>",
    ))
    # Build layout manually — no xaxis/yaxis kwargs to avoid duplicate-key error
    unsafe_layout = lay(
    "% of Dataset Samples Breaching Each Safety Threshold",
    height=340,
    xaxis_title="% of samples"
    )

    unsafe_layout["xaxis"]["range"] = [0, max(pcts) * 1.18]
    unsafe_layout["margin"] = dict(t=50, b=30, l=10, r=60)
    fig_unsafe.update_layout(**unsafe_layout)
    st.plotly_chart(fig_unsafe, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Community Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1: sel_region = st.selectbox("Region",       ["All"] + sorted(df['Region'].unique().tolist()))
    with f2: sel_year   = st.selectbox("Year",         ["All"] + sorted(df['Year'].unique().tolist()))
    with f3: sel_source = st.selectbox("Water Source", ["All"] + sorted(df['Water Source Type'].unique().tolist()))
    st.markdown('</div>', unsafe_allow_html=True)

    fdf = df.copy()
    if sel_region != "All": fdf = fdf[fdf['Region'] == sel_region]
    if sel_year   != "All": fdf = fdf[fdf['Year']   == sel_year]
    if sel_source != "All": fdf = fdf[fdf['Water Source Type'] == sel_source]

    st.markdown(
        f'<p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:1.5rem;">'
        f'Showing <strong style="color:#00d4ff;">{len(fdf):,}</strong> records</p>',
        unsafe_allow_html=True,
    )

    # Row 1 ── disease bar + potability donut
    r1, r2 = st.columns(2, gap="large")
    with r1:
        dd = fdf.groupby('Region')[
            ['Cholera Cases per 100,000 people', 'Typhoid Cases per 100,000 people', 'Diarrheal Cases per 100,000 people']
        ].mean().reset_index()
        fig1 = go.Figure()
        for col, lbl, clr in zip(
            ['Cholera Cases per 100,000 people', 'Typhoid Cases per 100,000 people', 'Diarrheal Cases per 100,000 people'],
            ["Cholera", "Typhoid", "Diarrheal"], ["#00d4ff", "#ff6b35", "#ffd166"],
        ):
            fig1.add_trace(go.Bar(name=lbl, x=dd['Region'], y=dd[col].round(1),
                                  marker_color=clr, marker_line_width=0, opacity=0.85))
        fig1.update_layout(**lay("Disease Cases by Region (avg per 100k)", height=320,
                                 yaxis_title="Cases per 100k", barmode='group'))
        st.plotly_chart(fig1, use_container_width=True)

    with r2:
        pot = fdf['Potability'].value_counts().reset_index()
        pot.columns = ['Potability', 'Count']
        pot['Label'] = pot['Potability'].map({1: 'Potable', 0: 'Not Potable'})
        fig2 = go.Figure(go.Pie(
            labels=pot['Label'], values=pot['Count'], hole=0.55,
            marker=dict(colors=["#00f5c4", "#ff4d6d"], line=dict(color="#040d1a", width=2)),
            textfont=dict(color="#e8f4fd", family="DM Sans"),
            hovertemplate="%{label}<br>Count: %{value:,}<br>%{percent}<extra></extra>",
        ))
        fig2.update_layout(
            **lay("Potability Distribution", height=320),
            annotations=[dict(text=f"{len(fdf):,}", x=0.5, y=0.5, showarrow=False,
                              font=dict(size=18, color="#e8f4fd", family="Syne"))],
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2 ── bacteria bar + source donut
    r3, r4 = st.columns(2, gap="large")
    with r3:
        sd = (fdf.groupby('Water Source Type')['Bacteria Count (CFU/mL)']
              .mean().reset_index().sort_values('Bacteria Count (CFU/mL)'))
        fig3 = go.Figure(go.Bar(
            x=sd['Bacteria Count (CFU/mL)'].round(0), y=sd['Water Source Type'], orientation='h',
            marker=dict(color=sd['Bacteria Count (CFU/mL)'],
                        colorscale=[[0, "#00f5c4"], [0.5, "#ffd166"], [1, "#ff4d6d"]], line=dict(width=0)),
            text=sd['Bacteria Count (CFU/mL)'].round(0).astype(int),
            textposition='outside', textfont=dict(color="#7fa8c9", size=11),
        ))
        fig3.update_layout(**lay("Avg Bacteria Count by Water Source", height=320,
                                 xaxis_title="CFU/mL"))
        st.plotly_chart(fig3, use_container_width=True)

    with r4:
        src = fdf['Water Source Type'].value_counts().reset_index()
        src.columns = ['Source', 'Count']
        fig4 = go.Figure(go.Pie(
            labels=src['Source'], values=src['Count'], hole=0.45,
            marker=dict(colors=["#00d4ff", "#00f5c4", "#ffd166", "#ff6b35", "#ff4d6d", "#a78bfa"],
                        line=dict(color="#040d1a", width=2)),
            textfont=dict(color="#e8f4fd", family="DM Sans", size=11),
            hovertemplate="%{label}<br>Count: %{value:,}<br>%{percent}<extra></extra>",
        ))
        fig4.update_layout(**lay("Water Source Type Distribution", height=320))
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3 ── trend line
    trend = fdf.groupby('Year')[
        ['Cholera Cases per 100,000 people', 'Typhoid Cases per 100,000 people']
    ].mean().reset_index()
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=trend['Year'], y=trend['Cholera Cases per 100,000 people'].round(2),
        mode='lines+markers', name='Cholera',
        line=dict(color="#00d4ff", width=2.5), marker=dict(size=5),
        fill='tozeroy', fillcolor='rgba(0,212,255,0.06)',
    ))
    fig5.add_trace(go.Scatter(
        x=trend['Year'], y=trend['Typhoid Cases per 100,000 people'].round(2),
        mode='lines+markers', name='Typhoid',
        line=dict(color="#ff6b35", width=2.5), marker=dict(size=5),
        fill='tozeroy', fillcolor='rgba(255,107,53,0.06)',
    ))
    fig5.update_layout(**lay("Disease Trend Over Years", height=300,
                             xaxis_title="Year", yaxis_title="Cases per 100k"))
    st.plotly_chart(fig5, use_container_width=True)

    # Row 4 ── pH histogram + scatter
    r5, r6 = st.columns(2, gap="large")
    with r5:
        fig6 = go.Figure(go.Histogram(
            x=fdf['pH Level'], nbinsx=40,
            marker=dict(color="#00d4ff", opacity=0.75, line=dict(color="#040d1a", width=0.5)),
        ))
        fig6.add_vline(x=6.5, line=dict(color="#ffd166", dash="dash", width=1.5))
        fig6.add_vline(x=8.5, line=dict(color="#ffd166", dash="dash", width=1.5))
        fig6.update_layout(**lay("pH Level Distribution (safe zone: 6.5–8.5)", height=280,
                                 xaxis_title="pH", yaxis_title="Frequency"))
        st.plotly_chart(fig6, use_container_width=True)

    with r6:
        samp = fdf.sample(min(3000, len(fdf)), random_state=42)
        fig7 = go.Figure(go.Scatter(
            x=samp['Turbidity (NTU)'], y=samp['Bacteria Count (CFU/mL)'], mode='markers',
            marker=dict(
                color=samp['Potability'], colorscale=[[0, "#ff4d6d"], [1, "#00f5c4"]],
                size=4, opacity=0.6,
                colorbar=dict(
                    title=dict(text="Potable", font=dict(color="#7fa8c9", size=12)),
                    tickvals=[0, 1], ticktext=["No", "Yes"],
                    tickfont=dict(color="#7fa8c9"),
                ),
            ),
        ))
        fig7.update_layout(**lay("Turbidity vs Bacteria Count", height=280,
                                 xaxis_title="Turbidity (NTU)", yaxis_title="Bacteria (CFU/mL)"))
        st.plotly_chart(fig7, use_container_width=True)

    # Row 5 ── box plots + scatter
    r7, r8 = st.columns(2, gap="large")
    with r7:
        fig8 = go.Figure()
        for region in sorted(fdf['Region'].unique()):
            sub = fdf[fdf['Region'] == region]['Sanitation Coverage (% of Population)']
            fig8.add_trace(go.Box(
                y=sub, name=region, line=dict(color="#00d4ff"),
                fillcolor="rgba(0,212,255,0.1)",
                marker=dict(size=3, color="rgba(0,212,255,0.5)"),
                boxpoints='outliers',
            ))
        fig8.update_layout(**lay("Sanitation Coverage by Region (%)", height=300,
                                 yaxis_title="%", showlegend=False))
        st.plotly_chart(fig8, use_container_width=True)

    with r8:
        scat = fdf.sample(min(2000, len(fdf)), random_state=1)
        fig9 = go.Figure()
        palette = ["#00d4ff", "#00f5c4", "#ff6b35", "#ffd166", "#ff4d6d", "#a78bfa"]
        for i, region in enumerate(sorted(scat['Region'].unique())):
            s = scat[scat['Region'] == region]
            fig9.add_trace(go.Scatter(
                x=s['Healthcare Access Index (0-100)'],
                y=s['Infant Mortality Rate (per 1,000 live births)'],
                mode='markers', name=region,
                marker=dict(size=4, color=palette[i % len(palette)], opacity=0.6),
            ))
        fig9.update_layout(**lay("Healthcare Access vs Infant Mortality", height=300,
                                 xaxis_title="Healthcare Access Index",
                                 yaxis_title="Infant Mortality Rate"))
        st.plotly_chart(fig9, use_container_width=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin-top:3rem;">
<p style="text-align:center;color:#3d6080;font-size:0.78rem;padding-bottom:1rem;">
     Smart Community Health Monitoring · AI-Powered Early Warning System
</p>
""", unsafe_allow_html=True)