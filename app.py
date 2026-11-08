"""
MBR Permeability Forecaster  -  IESE, NUST
===========================================
Streamlit app that loads the frozen best model (Random Forest) trained in
MBR_ML_Model_NUST.ipynb and forecasts NEXT-DAY membrane permeability P(t+1)
from routine plant measurements. It also converts the forecast into the TMP
expected tomorrow at a planned flux (TMP = Flux / P).

Run:   streamlit run app.py      ->  opens http://localhost:8501
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- page setup
st.set_page_config(page_title="MBR Permeability Forecaster | NUST",
                   page_icon="💧", layout="wide")

INK, AQUA, FOAM, SLUDGE, ALARM, AMBER, SLATE = (
    "#0E2F3B", "#1FA698", "#F2F6F5", "#8C6A3E", "#C2483D", "#D99A2B", "#5C7380")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600;700&family=Source+Sans+3:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, label {{ font-family: 'Source Sans 3', sans-serif; }}
h1, h2, h3, .readout, .stTabs button p {{ font-family: 'Barlow Semi Condensed', sans-serif !important; }}
.block-container {{ padding-top: 1.6rem; max-width: 1250px; }}
h1 {{ color: {INK}; font-weight: 700; font-size: 2.3rem; margin-bottom: 0; letter-spacing: -0.01em; }}
.sub {{ color: {SLATE}; font-size: 1.05rem; margin-top: .2rem; }}
.panel {{ background: white; border: 1px solid #DCE6E4; border-left: 5px solid {AQUA};
          border-radius: 6px; padding: 14px 18px; height: 100%; }}
.panel.warn {{ border-left-color: {AMBER}; }} .panel.alarm {{ border-left-color: {ALARM}; }}
.label {{ color: {SLATE}; font-size: .9rem; }}
.readout {{ color: {INK}; font-size: 2.2rem; font-weight: 700; line-height: 1.1; }}
.unit {{ color: {SLATE}; font-size: .95rem; font-family: 'Source Sans 3', sans-serif; font-weight: 400; }}
.delta-up {{ color: {AQUA}; font-weight: 600; }} .delta-down {{ color: {ALARM}; font-weight: 600; }}
.status {{ border-radius: 6px; padding: 14px 18px; color: white; font-size: 1.05rem; }}
section[data-testid="stSidebar"] {{ background: {INK}; }}
section[data-testid="stSidebar"] * {{ color: #E6F0EE; }}
section[data-testid="stSidebar"] input {{ color: {INK} !important; }}
section[data-testid="stSidebar"] [data-baseweb="select"] * {{ color: {INK} !important; }}
section[data-testid="stSidebar"] h3 {{ color: #9FE0D6; font-size: 1.15rem; margin: .6rem 0 .2rem; }}
.note {{ color: {SLATE}; font-size: .88rem; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- load assets
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(HERE, "best_model.joblib"))
    with open(os.path.join(HERE, "model_metadata.json")) as f:
        meta = json.load(f)
    return model, meta


@st.cache_data
def load_history():
    h = pd.read_csv(os.path.join(HERE, "mbr_history_clean.csv"), parse_dates=["datetime"])
    days, last = [], None
    for t, ce in zip(h["datetime"], h["Cleaning_Event"]):
        if ce == 1:
            last = t
        days.append(np.nan if last is None else (t - last).days)
    h["Days_since_cleaning"] = days
    return h


model, meta = load_model()
hist = load_history()
FEATURES = meta["features"]
R = meta["feature_ranges"]

# ---------------------------------------------------------------- defaults & prefill
FIELDS = ["F0", "T0", "F1", "T1", "F2", "T2", "MLSS", "MLVSS", "COD_in", "COD_out",
          "pH", "Turbidity", "TDS", "Conductivity", "clean_today", "days_clean", "F_plan"]


def values_for_row(i):
    """Return the sidebar values for a recorded day (needs the two days before it)."""
    r, r1, r2 = hist.iloc[i], hist.iloc[i - 1], hist.iloc[i - 2]
    v = {"F0": r.Flux, "T0": r.TMP, "F1": r1.Flux, "T1": r1.TMP, "F2": r2.Flux, "T2": r2.TMP,
         "MLSS": r.MLSS, "MLVSS": r.MLVSS, "COD_in": r.COD_in, "COD_out": r.COD_out,
         "pH": r.pH, "Turbidity": r.Turbidity, "TDS": r.TDS, "Conductivity": r.Conductivity,
         "clean_today": bool(r.Cleaning_Event == 1), "days_clean": int(r.Days_since_cleaning),
         "F_plan": r.Flux}
    return {k: (round(float(x), 2) if isinstance(x, (float, np.floating)) else x) for k, x in v.items()}


valid_idx = [i for i in range(2, len(hist))
             if hist.iloc[i - 2:i + 1][["Flux", "TMP"]].notna().all().all()]
date_labels = {hist.iloc[i]["datetime"].strftime("%d %b %Y"): i for i in valid_idx}

if "init" not in st.session_state:
    st.session_state.update(values_for_row(valid_idx[-1]))
    st.session_state["day_pick"] = list(date_labels)[-1]
    st.session_state["init"] = True


def apply_day():
    st.session_state.update(values_for_row(date_labels[st.session_state["day_pick"]]))


# ---------------------------------------------------------------- sidebar inputs
with st.sidebar:
    st.markdown("## Plant readings")
    st.selectbox("Load a recorded day", list(date_labels), key="day_pick", on_change=apply_day,
                 help="Fills every field with that day's measurements. Edit any value afterwards.")

    st.markdown("### Membrane: flux and TMP")
    c1, c2 = st.columns(2)
    c1.number_input("Flux today (LMH)", 5.0, 40.0, step=0.1, key="F0")
    c2.number_input("TMP today (kPa)", 1.0, 40.0, step=0.1, key="T0")
    c1.number_input("Flux yesterday", 5.0, 40.0, step=0.1, key="F1")
    c2.number_input("TMP yesterday", 1.0, 40.0, step=0.1, key="T1")
    c1.number_input("Flux 2 days ago", 5.0, 40.0, step=0.1, key="F2")
    c2.number_input("TMP 2 days ago", 1.0, 40.0, step=0.1, key="T2")

    st.markdown("### Cleaning cycle")
    st.toggle("Membrane cleaned today", key="clean_today")
    st.number_input("Days since last cleaning", 0, 120, step=1, key="days_clean")

    st.markdown("### Biomass")
    st.number_input("MLSS (mg/L)", 1000.0, 15000.0, step=10.0, key="MLSS")
    st.number_input("MLVSS (mg/L)", 500.0, 12000.0, step=10.0, key="MLVSS")

    st.markdown("### Organic load")
    st.number_input("COD influent (mg/L)", 10.0, 2000.0, step=1.0, key="COD_in")
    st.number_input("COD permeate (mg/L)", 0.0, 500.0, step=0.5, key="COD_out")

    st.markdown("### Water quality")
    st.number_input("pH", 4.0, 10.0, step=0.01, key="pH")
    st.number_input("Turbidity (NTU)", 0.0, 50.0, step=0.05, key="Turbidity")
    st.number_input("TDS (mg/L)", 50.0, 3000.0, step=1.0, key="TDS")
    st.number_input("Conductivity (µS/cm)", 100.0, 6000.0, step=1.0, key="Conductivity")

    st.markdown("### Tomorrow's plan")
    st.number_input("Planned flux tomorrow (LMH)", 5.0, 40.0, step=0.1, key="F_plan",
                    help="Used to convert the permeability forecast into expected TMP.")

s = st.session_state


# ---------------------------------------------------------------- feature engineering (same as notebook)
def make_features(v):
    P0, P1, P2 = v["F0"] / v["T0"], v["F1"] / v["T1"], v["F2"] / v["T2"]
    row = {"Flux": v["F0"], "TMP": v["T0"], "Permeability": P0, "Perm_lag1": P1, "Perm_lag2": P2,
           "TMP_change": v["T0"] - v["T1"], "Flux_change": v["F0"] - v["F1"],
           "MLSS": v["MLSS"], "MLVSS": v["MLVSS"], "MLVSS_MLSS": v["MLVSS"] / v["MLSS"],
           "COD_in": v["COD_in"], "COD_out": v["COD_out"],
           "COD_removal": (v["COD_in"] - v["COD_out"]) / v["COD_in"] * 100,
           "pH": v["pH"], "Turbidity": v["Turbidity"], "TDS": v["TDS"], "Conductivity": v["Conductivity"],
           "Cleaning_Event": int(v["clean_today"]),
           "Days_since_cleaning": 0 if v["clean_today"] else v["days_clean"]}
    return pd.DataFrame([row])[FEATURES]


X = make_features(s)
P_today = float(X["Permeability"].iloc[0])
P_next = float(model.predict(X)[0])
TMP_next = s["F_plan"] / P_next
change = (P_next - P_today) / P_today * 100

if P_next >= 1.60:
    status_cls, status_col, status_txt = "", AQUA, "Membrane in good condition. Permeability is in the post-cleaning range."
elif P_next >= 1.50:
    status_cls, status_col, status_txt = "warn", AMBER, "Watch. Permeability is declining through the cleaning cycle."
else:
    status_cls, status_col, status_txt = "alarm", ALARM, "Plan a cleaning. Permeability is at levels seen just before past cleanings."

out_of_range = [c for c in ["Flux", "TMP", "MLSS", "MLVSS", "COD_in", "COD_out", "pH", "Turbidity", "TDS", "Conductivity"]
                if not (R[c]["min"] * 0.95 <= float(X[c].iloc[0]) <= R[c]["max"] * 1.05)]

# ---------------------------------------------------------------- header + treatment-train hero
st.markdown("<h1>MBR permeability forecaster</h1>", unsafe_allow_html=True)
st.markdown(f"<div class='sub'>Next-day membrane permeability from routine monitoring, "
            f"using a {meta['model_name']} model trained on six months of plant data. IESE, NUST.</div>",
            unsafe_allow_html=True)


def train_svg(p_now, p_next, colour):
    stages = [("Raw sewage", f"COD {s['COD_in']:.0f} mg/L", SLUDGE),
              ("Equalisation", "buffer tank", SLATE),
              ("Bioreactor", f"MLSS {s['MLSS']:.0f} mg/L", SLUDGE),
              ("Membrane", f"P {p_now:.3f} now", colour),
              ("Permeate", f"COD {s['COD_out']:.0f} mg/L", AQUA)]
    w, gap, x0 = 170, 38, 10
    parts = [f'<svg viewBox="0 0 {x0*2 + 5*w + 4*gap} 150" width="100%" xmlns="http://www.w3.org/2000/svg" '
             f'style="font-family:Barlow Semi Condensed, sans-serif">']
    parts.append(f'<line x1="{x0}" y1="62" x2="{x0 + 5*w + 4*gap}" y2="62" stroke="#BFD3D0" stroke-width="6"/>')
    for i, (name, val, col) in enumerate(stages):
        x = x0 + i * (w + gap)
        big = name == "Membrane"
        fill = col if big else "white"
        txt = "white" if big else INK
        parts.append(f'<rect x="{x}" y="{22 if big else 30}" width="{w}" height="{80 if big else 64}" rx="6" '
                     f'fill="{fill}" stroke="{col}" stroke-width="2.5"/>')
        if big:  # hollow-fibre hint
            for k in range(7):
                fx = x + 22 + k * 21
                parts.append(f'<line x1="{fx}" y1="30" x2="{fx}" y2="94" stroke="white" stroke-opacity=".16" stroke-width="5"/>')
        parts.append(f'<text x="{x + w/2}" y="{54 if big else 58}" text-anchor="middle" font-size="19" '
                     f'font-weight="700" fill="{txt}">{name}</text>')
        parts.append(f'<text x="{x + w/2}" y="{78 if big else 80}" text-anchor="middle" font-size="15" fill="{txt}">{val}</text>')
        if i < 4:
            ax = x + w + gap / 2
            parts.append(f'<polygon points="{ax-6},55 {ax+7},62 {ax-6},69" fill="{SLATE}"/>')
    mx = x0 + 3 * (w + gap) + w / 2
    parts.append(f'<text x="{mx}" y="128" text-anchor="middle" font-size="17" font-weight="700" fill="{colour}">'
                 f'Tomorrow {p_next:.3f} LMH/kPa</text>')
    parts.append('</svg>')
    return "".join(parts)


st.markdown(train_svg(P_today, P_next, status_col), unsafe_allow_html=True)

# ---------------------------------------------------------------- tabs
tab1, tab2, tab3 = st.tabs(["Forecast", "Plant history", "Model card"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    arrow = "▲" if change >= 0 else "▼"
    dcls = "delta-up" if change >= 0 else "delta-down"
    c1.markdown(f"<div class='panel'><div class='label'>Permeability today</div>"
                f"<div class='readout'>{P_today:.3f}</div><div class='unit'>LMH/kPa</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='panel {status_cls}'><div class='label'>Forecast for tomorrow</div>"
                f"<div class='readout'>{P_next:.3f}</div><div class='unit'>LMH/kPa "
                f"<span class='{dcls}'>{arrow} {abs(change):.1f}%</span></div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='panel {status_cls}'><div class='label'>Expected TMP tomorrow</div>"
                f"<div class='readout'>{TMP_next:.2f}</div><div class='unit'>kPa at {s['F_plan']:.1f} LMH</div></div>",
                unsafe_allow_html=True)
    c4.markdown(f"<div class='panel'><div class='label'>Cleaning cycle</div>"
                f"<div class='readout'>Day {int(X['Days_since_cleaning'].iloc[0])}</div>"
                f"<div class='unit'>past cleanings came every 31 days</div></div>", unsafe_allow_html=True)

    st.markdown(f"<div class='status' style='background:{status_col};margin-top:14px'>{status_txt}</div>",
                unsafe_allow_html=True)
    if out_of_range:
        st.warning("Outside the range the model was trained on: " + ", ".join(out_of_range)
                   + ". Treat this forecast with caution.")

    g1, g2 = st.columns([1, 1.35])
    with g1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=P_next,
            number={"valueformat": ".3f", "suffix": " LMH/kPa", "font": {"size": 30, "color": INK}},
            delta={"reference": P_today, "valueformat": ".3f"},
            title={"text": "Forecast permeability vs cleaning-cycle range", "font": {"size": 15, "color": SLATE}},
            gauge={"axis": {"range": [1.35, 1.80]},
                   "bar": {"color": INK, "thickness": 0.25},
                   "steps": [{"range": [1.35, 1.50], "color": "#F2C9C4"},
                             {"range": [1.50, 1.60], "color": "#F6E3BC"},
                             {"range": [1.60, 1.80], "color": "#C6EBE5"}],
                   "threshold": {"line": {"color": SLATE, "width": 3}, "value": P_today}}))
        gauge.update_layout(height=300, margin=dict(l=25, r=25, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, width="stretch")
        st.markdown("<div class='note'>Red band: levels seen just before past cleanings. "
                    "Green band: typical after cleaning. Grey tick: today.</div>", unsafe_allow_html=True)

    with g2:
        days = ["2 days ago", "Yesterday", "Today", "Tomorrow"]
        perms = [float(X["Perm_lag2"].iloc[0]), float(X["Perm_lag1"].iloc[0]), P_today]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days[:3], y=perms, mode="lines+markers", name="Measured",
                                 line=dict(color=INK, width=3), marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=days[2:], y=[P_today, P_next], mode="lines+markers", name="Forecast",
                                 line=dict(color=status_col, width=3, dash="dash"),
                                 marker=dict(size=[10, 16], symbol=["circle", "diamond"])))
        rmse = meta["test_metrics"][meta["model_name"]]["RMSE"]
        fig.add_trace(go.Scatter(x=["Tomorrow", "Tomorrow"], y=[P_next - rmse, P_next + rmse], mode="lines",
                                 line=dict(color=status_col, width=8), opacity=0.3, name=f"± test RMSE ({rmse:.3f})"))
        fig.update_layout(title="Permeability: last three days and tomorrow", height=330,
                          yaxis_title="LMH/kPa", plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
        fig.update_yaxes(gridcolor="#E3ECEA")
        st.plotly_chart(fig, width="stretch")

    st.markdown("### What if TMP changes today?")
    st.markdown("<div class='note'>Holding every other reading fixed, how the forecast responds to today's TMP. "
                "A steeper fall means the model is more sensitive to rising pressure.</div>", unsafe_allow_html=True)
    tmp_grid = np.linspace(R["TMP"]["min"], R["TMP"]["max"], 40)
    rows = []
    for t in tmp_grid:
        v = s.to_dict(); v["T0"] = t
        rows.append(make_features(v))
    preds = model.predict(pd.concat(rows))
    wf = go.Figure()
    wf.add_trace(go.Scatter(x=tmp_grid, y=preds, mode="lines", line=dict(color=AQUA, width=3),
                            fill="tozeroy", fillcolor="rgba(31,166,152,0.08)", name="Forecast P(t+1)"))
    wf.add_trace(go.Scatter(x=[s["T0"]], y=[P_next], mode="markers", marker=dict(size=14, color=INK),
                            name="Current input"))
    wf.update_layout(height=300, xaxis_title="TMP today (kPa)", yaxis_title="Forecast P(t+1), LMH/kPa",
                     plot_bgcolor="white", margin=dict(l=10, r=10, t=20, b=10),
                     yaxis=dict(range=[min(preds) - 0.03, max(preds) + 0.03], gridcolor="#E3ECEA"))
    st.plotly_chart(wf, width="stretch")

    st.markdown("### What drives this forecast")
    try:
        import shap
        expl = shap.TreeExplainer(model.named_steps["model"])
        sv = pd.Series(expl.shap_values(X)[0], index=FEATURES)
        top = sv.reindex(sv.abs().sort_values(ascending=False).index[:8])[::-1]
        bar = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                               marker_color=[AQUA if x > 0 else ALARM for x in top.values]))
        bar.update_layout(height=320, xaxis_title="Contribution to forecast (LMH/kPa)",
                          plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(bar, width="stretch")
        st.markdown("<div class='note'>SHAP contributions for this input: green pushes tomorrow's permeability up, "
                    "red pushes it down. These are predictive associations, not causes.</div>", unsafe_allow_html=True)
    except Exception:
        imp = pd.Series(meta["feature_importance"]).head(8)[::-1]
        bar = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=AQUA))
        bar.update_layout(height=320, xaxis_title="Global importance", plot_bgcolor="white",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(bar, width="stretch")

with tab2:
    options = {"Permeability (LMH/kPa)": "Permeability", "TMP (kPa)": "TMP", "Flux (LMH)": "Flux",
               "MLSS (mg/L)": "MLSS", "MLVSS (mg/L)": "MLVSS", "COD influent (mg/L)": "COD_in",
               "COD permeate (mg/L)": "COD_out", "COD removal (%)": "COD_removal",
               "MLVSS/MLSS": "MLVSS_MLSS", "Turbidity (NTU)": "Turbidity", "TDS (mg/L)": "TDS",
               "Conductivity (µS/cm)": "Conductivity", "pH": "pH"}
    pick = st.multiselect("Variables to plot", list(options), default=["Permeability (LMH/kPa)"])
    norm = st.toggle("Scale each variable to 0–1 so they share one axis", value=len(pick) > 1)
    hf = go.Figure()
    colours = [AQUA, ALARM, INK, SLUDGE, AMBER, SLATE, "#4C8FBF", "#7A9E3B"]
    for k, label in enumerate(pick):
        y = hist[options[label]]
        if norm:
            y = (y - y.min()) / (y.max() - y.min())
        hf.add_trace(go.Scatter(x=hist["datetime"], y=y, name=label, mode="lines",
                                line=dict(color=colours[k % len(colours)], width=2)))
    for d in hist.loc[hist["Cleaning_Event"] == 1, "datetime"]:
        hf.add_vline(x=d, line=dict(color=SLATE, dash="dot", width=1))
    hf.update_layout(height=430, plot_bgcolor="white", hovermode="x unified",
                     title="Historical monitoring data, Jan–Jun 2026 (dotted lines: membrane cleaning)",
                     margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15))
    hf.update_yaxes(gridcolor="#E3ECEA")
    st.plotly_chart(hf, width="stretch")

    cycle = hist.dropna(subset=["Permeability"])
    sc = go.Figure(go.Scatter(x=cycle["Days_since_cleaning"], y=cycle["Permeability"], mode="markers",
                              marker=dict(color=cycle["datetime"].dt.month, colorscale="Teal", size=8,
                                          colorbar=dict(title="Month"))))
    sc.update_layout(height=360, title="Permeability across the cleaning cycle",
                     xaxis_title="Days since last cleaning", yaxis_title="Permeability (LMH/kPa)",
                     plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(sc, width="stretch")

with tab3:
    st.markdown(f"### {meta['model_name']}: next-day permeability forecaster")
    tm = pd.DataFrame(meta["test_metrics"]).T[["R2", "RMSE", "MAE"]]
    mc = go.Figure()
    for metric, col in zip(["R2", "RMSE", "MAE"], [AQUA, ALARM, AMBER]):
        mc.add_trace(go.Bar(name=metric, x=tm.index, y=tm[metric], marker_color=col,
                            text=tm[metric].round(3), textposition="outside"))
    mc.update_layout(barmode="group", height=360, plot_bgcolor="white",
                     title="Performance on the unseen test month (June 2026)",
                     margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(mc, width="stretch")
    st.dataframe(tm.round(4), width="stretch")
    st.markdown(f"""
**Target:** {meta['target']}
**Training data:** {meta['train_period']}. Split by time: {meta['split']['train']}, {meta['split']['validation']}, {meta['split']['test']}.
**Hyper-parameters:** `{meta['hyperparameters']}`
**Inputs ({len(FEATURES)}):** {', '.join(FEATURES)}

**Use responsibly**
- The model forecasts permeability decline (hydraulic deterioration). It does not identify fouling mechanisms or causes.
- It learned this plant's 31-day cleaning rhythm. If cleaning dates change, enter them correctly, because unscheduled cleanings are not predictable.
- Valid for the studied MBR only. TMP is assumed to be in kPa and flux in LMH; confirm the units before operational use.
- Decision support only. It does not replace laboratory monitoring.
""")
