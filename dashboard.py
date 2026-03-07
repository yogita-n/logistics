# dashboard.py
import streamlit as st
import json, os
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.manifold import MDS

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RouteAI — Last Mile Optimizer",
    page_icon="🚚",
    layout="wide"
)

# ── Defaults ──────────────────────────────────────────────────────────────────
run_btn  = False
day      = 1
scenario = "mostlikely"
num_veh  = 5

# ── Colors for each vehicle ───────────────────────────────────────────────────
VEHICLE_COLORS = ["blue", "red", "green", "purple", "orange",
                  "darkred", "cadetblue", "darkgreen", "darkpurple", "gray"]

# ── Load data helpers ─────────────────────────────────────────────────────────
@st.cache_data
def load_data(day, scenario):
    with open(f"data/processed/day{day}_{scenario}.json") as f:
        return json.load(f)

@st.cache_data
def load_result(day):
    with open(f"data/results/day{day}_result.json") as f:
        return json.load(f)

@st.cache_data
def get_coords(day, scenario):
    data = load_data(day, scenario)
    dist = np.array(data["dist_matrix"])
    dist = (dist + dist.T) / 2   # symmetrize

    mds = MDS(n_components=2, metric=True, n_init=1,
              random_state=42, normalized_stress="auto")
    coords_2d = mds.fit_transform(dist)

    lat_center, lng_center = 37.97, 23.73
    scale = 0.08
    lat_range = coords_2d[:, 1].max() - coords_2d[:, 1].min()
    lng_range = coords_2d[:, 0].max() - coords_2d[:, 0].min()

    lats = lat_center + (coords_2d[:, 1] - coords_2d[:, 1].mean()) / lat_range * scale
    lngs = lng_center + (coords_2d[:, 0] - coords_2d[:, 0].mean()) / lng_range * scale
    return list(zip(lats.tolist(), lngs.tolist()))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/delivery--v1.png", width=80)
    st.title("RouteAI Controls")
    st.markdown("---")

    day      = st.selectbox("📅 Select Day", list(range(1, 10)), index=0)
    scenario = st.selectbox("🚦 Traffic Scenario",
                            ["mostlikely", "optimistic", "pessimistic"])
    num_veh  = st.slider("🚚 Number of Vehicles", 1, 8, 5)
    st.markdown("---")

    available = [d for d in range(1, 10)
                 if os.path.exists(f"data/results/day{d}_result.json")]
    st.caption(f"✅ Results ready: Days {available}")
    st.caption(f"📌 Viewing: Day {day} | {scenario}")
    st.markdown("---")

    run_btn = st.button("🧠 Run Optimization", use_container_width=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚚 RouteAI — Last Mile Delivery Optimizer")
st.caption("AI-powered route optimization using VRPTW + Dynamic Priority-Penalty Re-weighting (DPPR)")
st.markdown("---")

# ── Run optimizer ─────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"🧠 Optimizing Day {day} ({scenario})... please wait ~10 sec"):
        import subprocess
        my_env = os.environ.copy()
        my_env["PYTHONIOENCODING"] = "utf-8"
        res = subprocess.run(
            ["python", "optimizer.py",
             f"--day={day}",
             f"--scenario={scenario}",
             f"--vehicles={num_veh}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=my_env
        )
    if res.returncode == 0:
        st.success(f"✅ Day {day} optimized successfully!")
        st.cache_data.clear()
        st.rerun()
    else:
        st.error(f"❌ Optimizer error:\n{res.stderr}")
        st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    data   = load_data(day, scenario)
    result = load_result(day)
    coords = get_coords(day, scenario)
except FileNotFoundError:
    st.warning(f"⚠️ No result for Day {day}. Click **🧠 Run Optimization** in the sidebar.")
    st.stop()

depot_coord = coords[0]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: METRICS CARDS
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Performance Metrics")

# FIX: handle negative savings gracefully in metric display
time_saved = result['time_saved_min']
dist_saved = result['dist_saved_km']
cost_saved = result['cost_saved_inr']
eff_gain   = result['efficiency_gain_pct']

c1, c2, c3, c4, c5 = st.columns(5)

# FIX: delta sign/color now correctly reflects positive=good, negative=bad
c1.metric(
    "🕐 Time Saved",
    f"{abs(time_saved)} min",
    delta=f"{'−' if time_saved < 0 else '−'}{eff_gain}%" if time_saved >= 0
          else f"+{abs(eff_gain)}% slower",
    delta_color="normal" if time_saved >= 0 else "inverse"
)
c2.metric(
    "📏 Distance Saved",
    f"{dist_saved} km",
    delta=f"{'saved' if dist_saved >= 0 else 'extra'}"
)
c3.metric(
    "💰 Cost Saved",
    f"₹{cost_saved}",
    delta="saved" if cost_saved >= 0 else "extra cost"
)
c4.metric("🚚 Vehicles Used", f"{result['num_vehicles_used']}")
c5.metric(
    "📦 Stops Served",
    f"{sum(r['num_stops'] for r in result['optimized_routes'])}/{result['num_stops']}"
)

# FIX: add a callout explaining the comparison methodology
st.info(
    "ℹ️ **Comparison note:** Naive = 1 vehicle visits all stops sequentially by tightest deadline. "
    "Optimized = time until the *last* vehicle returns (makespan), so multiple vehicles "
    "working in parallel is fairly compared to the naive single-vehicle elapsed time."
)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: MAPS — Naive vs Optimized
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🗺 Route Visualization")
col_naive, col_opt = st.columns(2)

# ── NAIVE MAP ─────────────────────────────────────────────────────────────────
with col_naive:
    st.markdown("### 🔴 Naive Route")
    st.caption("Single vehicle — stops sorted by tightest deadline (earliest LAT) only")

    m_naive = folium.Map(location=depot_coord, zoom_start=13,
                         tiles="CartoDB positron")
    folium.Marker(depot_coord, popup="🏭 DEPOT",
                  icon=folium.Icon(color="black", icon="home",
                                   prefix="fa")).add_to(m_naive)

    # FIX: sort by 'latest_arrival' (renamed from 'lat')
    naive_nodes = [s["id"] for s in sorted(data["stops"],
                                            key=lambda s: s["latest_arrival"])]
    naive_line  = [depot_coord] + [coords[n] for n in naive_nodes] + [depot_coord]
    folium.PolyLine(naive_line, color="red", weight=2.5,
                    opacity=0.8, tooltip="Naive Route").add_to(m_naive)

    for seq, node in enumerate(naive_nodes):
        stop = next(s for s in data["stops"] if s["id"] == node)
        folium.CircleMarker(
            coords[node], radius=6, color="red",
            fill=True, fill_opacity=0.8,
            popup=folium.Popup(
                # FIX: use correct renamed field names in popup
                f"<b>Stop {node}</b><br>"
                f"Seq: {seq + 1}<br>"
                f"Window: {stop['earliest_arrival']}–{stop['latest_arrival']} min<br>"
                f"Priority: {stop['priority']}<br>"
                f"Weight: {stop['weight_kg']} kg",
                max_width=220)
        ).add_to(m_naive)

    st_folium(m_naive, width=600, height=450, key=f"naive_map_{day}")
    st.info(f"⏱ {result['naive_time_min']} min  |  📏 {result['naive_dist_km']} km")

# ── OPTIMIZED MAP ─────────────────────────────────────────────────────────────
with col_opt:
    st.markdown("### ✅ AI Optimized Routes")
    st.caption(f"{result['num_vehicles_used']} vehicles — VRPTW + DPPR algorithm")

    m_opt = folium.Map(location=depot_coord, zoom_start=13,
                       tiles="CartoDB positron")
    folium.Marker(depot_coord, popup="🏭 DEPOT",
                  icon=folium.Icon(color="black", icon="home",
                                   prefix="fa")).add_to(m_opt)

    for route in result["optimized_routes"]:
        color = VEHICLE_COLORS[(route["vehicle_id"] - 1) % len(VEHICLE_COLORS)]
        nodes = route["stop_sequence"]
        line  = [depot_coord] + [coords[n] for n in nodes] + [depot_coord]
        folium.PolyLine(line, color=color, weight=3, opacity=0.9,
                        tooltip=f"Vehicle {route['vehicle_id']} | "
                                f"{route['num_stops']} stops").add_to(m_opt)
        for seq, node in enumerate(nodes):
            stop = next(s for s in data["stops"] if s["id"] == node)
            folium.CircleMarker(
                coords[node], radius=6, color=color,
                fill=True, fill_opacity=0.9,
                popup=folium.Popup(
                    # FIX: correct field names in popup
                    f"<b>Stop {node}</b><br>"
                    f"Vehicle: {route['vehicle_id']}<br>"
                    f"Seq: {seq + 1}<br>"
                    f"Window: {stop['earliest_arrival']}–{stop['latest_arrival']} min<br>"
                    f"Priority: {stop['priority']}<br>"
                    f"Weight: {stop['weight_kg']} kg",
                    max_width=220)
            ).add_to(m_opt)

    st_folium(m_opt, width=600, height=450, key=f"opt_map_{day}")
    st.success(f"⏱ {result['total_opt_time_min']} min (makespan)  |  "
               f"📏 {result['total_opt_dist_km']} km (total)")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: PER-VEHICLE TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📋 Detailed Route Breakdown")

rows = []
for r in result["optimized_routes"]:
    rows.append({
        "Vehicle":       f"🚚 Vehicle {r['vehicle_id']}",
        "Stops":         r["num_stops"],
        "Travel Time":   f"{r['travel_time_min']} min",
        "Distance":      f"{r['distance_km']} km",
        "Stop Sequence": str(r["stop_sequence"])
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: SIMULATION — All 9 Days
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📈 Performance Simulation — All 9 Days")

sim_rows = []
for d in range(1, 10):
    path = f"data/results/day{d}_result.json"
    if os.path.exists(path):
        with open(path) as f:
            r = json.load(f)
        sim_rows.append({
            "Day":                   f"Day {d}",
            "Stops Served":          sum(x["num_stops"] for x in r["optimized_routes"]),
            "Naive Time (min)":      r["naive_time_min"],
            # FIX: label clarifies this is makespan (not sum of all vehicles)
            "Optimized Makespan (min)": r["total_opt_time_min"],
            "Time Saved (min)":      r["time_saved_min"],
            "Dist Saved (km)":       r["dist_saved_km"],
            "Cost Saved (₹)":        r["cost_saved_inr"],
            "Efficiency Gain":       f"{r['efficiency_gain_pct']}%"
        })

if sim_rows:
    sim_df = pd.DataFrame(sim_rows)
    st.dataframe(sim_df, use_container_width=True, hide_index=True)
    st.bar_chart(
        sim_df.set_index("Day")[["Naive Time (min)", "Optimized Makespan (min)"]],
        use_container_width=True
    )
else:
    st.info("Run optimization for all 9 days using the sidebar to populate this chart.")

st.markdown("---")
st.caption("Built with OR-Tools VRPTW + DPPR | RouteAI v1.0 | © 2026 Yogita Babu Naik")