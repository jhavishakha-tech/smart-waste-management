"""Smart Waste Segregation & Collection - interactive demo dashboard.   streamlit run app.py"""
import os, time, json
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

import config as C
import routing as R
import forecasting as F
from simulator import SmartWasteSystem, run_simulation
from classifier.waste_classifier import (WasteClassifier, CLASSES, COMPARTMENT, COMPARTMENTS,
                                         CONF_THRESHOLD, route_item)

st.set_page_config(page_title="Smart Waste System - Bengaluru", page_icon="♻️", layout="wide")
RES = os.path.join(os.path.dirname(__file__), "results")
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ROUTE_COLORS = ["#1f77b4", "#9467bd", "#2ca02c", "#e377c2", "#17becf", "#8c564b", "#bcbd22", "#ff7f0e"]


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading forecasting model...")
def get_forecaster():
    return F.load_or_train()


@st.cache_resource
def get_classifier():
    return WasteClassifier()


def sim_clock(t):
    return f"{DAYS[(t // 24) % 7]}, day {t // 24 + 1}, {t % 24:02d}:00"


def new_system(n, seed):
    s = SmartWasteSystem(n, seed, hours=24 * 40)
    fc = get_forecaster()
    while s.t < s.warm + C.DECISION_HOUR - 1 - 0:                    # warm-up: 2 days of daily collection
        s.step()
        if s.is_decision_time():
            s.collect(s.select("fixed_daily")[0], 0.1)
    s.reset_metrics(); s.last_route = None
    return s


def advance(s, hours, policy, theta, auto):
    fc = get_forecaster()
    for _ in range(hours):
        s.step()
        if s.is_decision_time() and auto:
            sel, _ = s.select(policy, fc, theta)
            s.last_route = s.collect(sel, 0.3)
            s.last_route["t"] = s.t


def predictions(s, theta):
    fc = get_forecaster()
    pred = fc.predict(s.ghat, s.t, s.field)
    df = s.field.df[["bin_id", "area", "lat", "lon"]].copy()
    df["fill_%"] = (s.obs * 100).round(1)
    df["pred_+24h_%"] = ((s.obs + pred) * 100).round(1)
    rate = np.maximum(pred / 24.0, 1e-4)
    df["hrs_to_full"] = np.where(s.obs >= 0.99, 0.0, np.minimum((1 - s.obs) / rate, 99)).round(0)
    df["alert"] = (s.obs + pred >= theta)
    return df


def colour(f):
    return "#7f0000" if f >= 0.99 else "#d62728" if f >= 0.8 else "#ff9f1c" if f >= 0.5 else "#2ca02c"


def make_map(df, route=None, field=None, height=560):
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=11, tiles="OpenStreetMap")
    folium.Marker([C.DEPOT["lat"], C.DEPOT["lon"]], tooltip=C.DEPOT["name"],
                  icon=folium.Icon(color="black", icon="truck", prefix="fa")).add_to(m)
    for _, r in df.iterrows():
        f = r["fill_%"] / 100
        folium.CircleMarker([r.lat, r.lon], radius=5 + 6 * f, color=colour(f), fill=True, fill_opacity=.85, weight=1,
                            tooltip=f"<b>{r.bin_id}</b> - {r.area}<br>Fill {r['fill_%']:.0f}% | +24h {r['pred_+24h_%']:.0f}%"
                                    f"<br>~{r.hrs_to_full:.0f} h to full").add_to(m)
    if route and field is not None:
        for k, stops in enumerate(route["routes"]):
            pts = [[C.DEPOT["lat"], C.DEPOT["lon"]]] + [[field.lat[i], field.lon[i]] for i in stops] + [[C.DEPOT["lat"], C.DEPOT["lon"]]]
            folium.PolyLine(pts, color=ROUTE_COLORS[k % len(ROUTE_COLORS)], weight=3, opacity=.8,
                            tooltip=f"Truck {k+1}: {len(stops)} bins").add_to(m)
    return m


def plan_preview(s, policy, theta):
    """Plan (but do not execute) a collection for the current state."""
    sel, _ = s.select(policy, get_forecaster(), theta)
    if len(sel) == 0:
        return dict(routes=[], distance_km=0.0, sel=sel, greedy_km=0.0)
    nodes = np.r_[0, sel + 1]
    D = s.field.dist[np.ix_(nodes, nodes)]
    dem = np.r_[0, np.maximum(np.ceil(np.clip(s.obs[sel], 0, 1) * C.BIN_VOLUME_L), 1)].astype(int)
    res = R.solve_vrp(D, dem, time_limit_s=0.8)
    gr = R.solve_greedy(D, dem)
    routes = [[int(sel[k - 1]) for k in r] for r in res["routes"]]
    return dict(routes=routes, distance_km=res["distance_m"] / 1000, sel=sel, greedy_km=gr["distance_m"] / 1000)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("♻️ Smart Waste - Bengaluru")
n_bins = st.sidebar.select_slider("Number of smart bins", [20, 50, 100, 200], value=50)
seed = st.sidebar.number_input("Random seed", 0, 999, 1)
policy = st.sidebar.selectbox("Collection policy", C.POLICIES, index=3, format_func=lambda p: C.POLICY_LABEL[p])
theta = st.sidebar.slider("Proposed: dispatch threshold θ", 0.6, 1.0, C.THETA, 0.05,
                          help="Collect a bin if sensed fill + forecast 24 h accumulation ≥ θ")
auto = st.sidebar.checkbox("Auto-dispatch trucks at 06:00", True)
if st.sidebar.button("🔄 Reset system") or "sys" not in st.session_state \
        or st.session_state.get("cfg") != (n_bins, seed):
    with st.spinner("Initialising bins & 2-day sensor history..."):
        st.session_state.sys = new_system(n_bins, seed)
        st.session_state.cfg = (n_bins, seed)
        st.session_state.bin_station = {c: 0 for c in COMPARTMENTS}
        st.session_state.feed = []
s = st.session_state.sys

st.sidebar.markdown("---")
st.sidebar.caption("Simulation-based prototype. Fill levels are synthetic (assumptions in config.py); "
                   "the classifier uses real image datasets (Phase 2).")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Live Dashboard", "🤖 Smart Bin (AI)", "🚛 Route Optimizer",
                                         "🧪 Policy Simulation", "📊 Paper Results"])

# ------------------------------------------------------------------ TAB 1
with tab1:
    st.subheader(f"Simulated time: {sim_clock(s.t)}")
    c = st.columns(4)
    if c[0].button("⏱ +1 hour"): advance(s, 1, policy, theta, auto)
    if c[1].button("⏱ +6 hours"): advance(s, 6, policy, theta, auto)
    if c[2].button("⏱ +24 hours"): advance(s, 24, policy, theta, auto)
    if c[3].button("⏱ +7 days"):
        with st.spinner("Simulating a week..."): advance(s, 24 * 7, policy, theta, auto)
    df = predictions(s, theta)
    k = st.columns(5)
    k[0].metric("Avg fill", f"{df['fill_%'].mean():.0f}%")
    k[1].metric("Bins > 80 %", int((df["fill_%"] > 80).sum()))
    k[2].metric("Bins full now", int((s.fill >= 0.99).sum()))
    sm = s.summary()
    k[3].metric("Distance so far", f"{sm['distance_km']:.0f} km")
    k[4].metric("Overflow bin-hours", f"{sm['overflow_pct']:.2f}%")
    left, right = st.columns([3, 2])
    with left:
        st_folium(make_map(df, getattr(s, "last_route", None), s.field), height=520, width=None, returned_objects=[])
        st.caption("🟢 <50 %  🟠 50-80 %  🔴 >80 %  ⚫ overflowing — lines: last dispatched truck routes "
                   "(straight-line view; distances use ×1.35 road factor)")
    with right:
        st.markdown(f"**⚠️ Forecast alerts** (sensed + predicted ≥ {theta:.0%} by next dispatch)")
        al = df[df.alert].sort_values("pred_+24h_%", ascending=False)[["bin_id", "area", "fill_%", "pred_+24h_%", "hrs_to_full"]]
        st.dataframe(al, height=230, hide_index=True)
        lr = getattr(s, "last_route", None)
        if lr:
            st.markdown(f"**Last dispatch ({sim_clock(lr['t'])}):** {len(lr['bins'])} bins, {lr['trips']} trucks, "
                        f"{lr['distance_km']:.1f} km")
        hist = pd.DataFrame(s.obs_hist[max(0, s.t - 72):s.t + 1] * 100, columns=s.field.df.bin_id)
        pick = st.multiselect("Fill history (last 72 h)", list(hist.columns), default=list(hist.columns[:3]))
        if pick: st.line_chart(hist[pick], height=200)

# ------------------------------------------------------------------ TAB 2
with tab2:
    clf = get_classifier()
    st.subheader("AI waste segregation at the smart-bin station")
    if clf.available:
        st.success(f"Trained classifier loaded ({clf.kind}).")
    else:
        st.warning("**Demo mode:** no trained model in `classifier/` yet (Phase 2: train in Colab, drop the file in). "
                   "Pick the class manually to demonstrate the rest of the pipeline - confidence-based rejection and bin updates.")
    a, b = st.columns(2)
    with a:
        img = st.file_uploader("Upload an item image", type=["jpg", "jpeg", "png"])
        if img is not None:
            from PIL import Image
            im = Image.open(img); st.image(im, width=260)
        if clf.available and img is not None:
            label, conf, probs = clf.predict(im)
            st.write(f"**Prediction:** {label} ({conf:.1%})")
            st.bar_chart(pd.Series(probs, index=clf.classes))
        else:
            label = st.selectbox("Item class (simulated classifier output)", CLASSES, index=0)
            conf = st.slider("Simulated confidence", 0.0, 1.0, 0.9, 0.01)
        thr = st.slider("Rejection threshold", 0.3, 0.95, CONF_THRESHOLD, 0.05)
        if st.button("➕ Drop item into bin", type="primary"):
            comp = route_item(label, conf, thr)
            st.session_state.bin_station[comp] += 1
            st.session_state.feed.insert(0, dict(item=label, confidence=round(conf, 2), compartment=comp))
    with b:
        cap = 20
        st.markdown("**Bin compartments** (capacity 20 items each, demo scale)")
        for comp in COMPARTMENTS:
            n_it = st.session_state.bin_station[comp]
            st.progress(min(n_it / cap, 1.0), text=f"{comp}: {n_it}/{cap}" + ("  - manual check needed" if comp == "Reject" and n_it else ""))
        payload = dict(bin_id="STATION-01", ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       fill={c_: round(st.session_state.bin_station[c_] / cap, 2) for c_ in COMPARTMENTS})
        st.markdown("**MQTT payload** → `smartwaste/bin/STATION-01/telemetry`")
        st.code(json.dumps(payload, indent=2), language="json")
        if st.session_state.feed:
            st.dataframe(pd.DataFrame(st.session_state.feed).head(8), hide_index=True)

# ------------------------------------------------------------------ TAB 3
with tab3:
    st.subheader("Route optimisation for the current system state")
    st.caption("All policies use the same OR-Tools capacitated-VRP solver; they differ only in WHICH bins are collected.")
    if s.t < s.warm:
        st.info("Initialising...")
    else:
        if st.button("🚛 Generate routes for all policies", type="primary"):
            rows, plans = [], {}
            for p in C.POLICIES:
                pl = plan_preview(s, p, theta); plans[p] = pl
                rows.append({"Policy": C.POLICY_LABEL[p], "Bins to collect": len(pl["sel"]), "Trucks": len(pl["routes"]),
                             "OR-Tools distance (km)": round(pl["distance_km"], 1),
                             "Nearest-neighbour (km)": round(pl["greedy_km"], 1),
                             "Fuel (L)": round(pl["distance_km"] * C.FUEL_L_PER_KM, 1)})
            st.session_state.plans = plans; st.session_state.plan_tbl = pd.DataFrame(rows)
        if "plans" in st.session_state and st.session_state.get("cfg") == (n_bins, seed):
            st.dataframe(st.session_state.plan_tbl, hide_index=True)
            show = st.selectbox("Show route map for", C.POLICIES, index=3, format_func=lambda p: C.POLICY_LABEL[p], key="showpol")
            pl = st.session_state.plans[show]
            st_folium(make_map(predictions(s, theta), dict(routes=pl["routes"]), s.field, 520), height=520, width=None, returned_objects=[])
        st.caption("Snapshot = bins' state at the simulated time shown on the Live Dashboard. Savings of 'proposed' show up over "
                   "multiple days (Policy Simulation tab) because bins left uncollected keep filling.")

# ------------------------------------------------------------------ TAB 4
with tab4:
    st.subheader("Multi-day policy simulation (fast-forward)")
    c = st.columns(3)
    sim_n = c[0].select_slider("Bins", [20, 50, 100], value=50, key="simn")
    sim_days = c[1].slider("Days", 5, 21, 10)
    sim_seed = c[2].number_input("Seed", 0, 999, 1, key="sims")
    if st.button("▶ Run all four policies", type="primary"):
        fc = get_forecaster(); rows = []; prog = st.progress(0.0)
        for i, p in enumerate(C.POLICIES):
            r = run_simulation(sim_n, p, int(sim_seed), days=sim_days, forecaster=fc, theta=theta,
                               time_limit_s=0.3 if sim_n <= 50 else 0.5)
            rows.append({"Policy": C.POLICY_LABEL[p], "Distance (km/day)": r["distance_km_per_day"],
                         "Trucks/day": r["trucks_per_day"], "Bins collected/day": r["bins_collected_per_day"],
                         "Overflow (% bin-h)": r["overflow_pct"], "Avg fill at collection": r["avg_fill_at_collection"],
                         "CO₂ (kg)": r["co2_kg"]})
            prog.progress((i + 1) / 4)
        d = pd.DataFrame(rows).set_index("Policy")
        st.dataframe(d.round(2))
        a, b = st.columns(2)
        a.markdown("**Distance (km/day)**"); a.bar_chart(d["Distance (km/day)"])
        b.markdown("**Overflow (% bin-hours)**"); b.bar_chart(d["Overflow (% bin-h)"])
        base = d.iloc[0]["Distance (km/day)"]; pr = d.iloc[3]
        st.success(f"Proposed vs fixed-daily: distance {100*(pr['Distance (km/day)']-base)/base:+.0f}%, "
                   f"overflow {pr['Overflow (% bin-h)']:.2f}% (fixed-daily {d.iloc[0]['Overflow (% bin-h)']:.2f}%).")

# ------------------------------------------------------------------ TAB 5
with tab5:
    st.subheader("Results from the full experiment suite (10 seeds, mean ± std)")
    if not os.path.exists(os.path.join(RES, "main_summary.csv")):
        st.info("Run `python experiments.py all` to generate paper results.")
    else:
        for fig, cap in [("fig_main.png", "Main comparison across fleet sizes"),
                         ("fig_pareto.png", "Distance vs overflow trade-off (100 bins; τ/θ sweeps)"),
                         ("fig_forecast.png", "24 h fill forecasting models"),
                         ("fig_robust.png", "Robustness to sensor noise and packet loss")]:
            p = os.path.join(RES, fig)
            if os.path.exists(p): st.image(p, caption=cap)
        for csv, cap in [("stat_tests.csv", "Paired Wilcoxon tests (proposed vs baselines)"),
                         ("forecast_models.csv", "Forecasting model comparison")]:
            p = os.path.join(RES, csv)
            if os.path.exists(p):
                st.markdown(f"**{cap}**"); st.dataframe(pd.read_csv(p).round(3), hide_index=True)
