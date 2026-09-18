"""
CIVL 6962 — Homework 1: Traffic Data Dashboard
================================================
Dataset: PeMS04 (Caltrans PeMS, District 4 — San Francisco Bay Area)
         307 loop detectors, 5-minute intervals, 2018-01-01 to 2018-02-28.
         Channels: flow (veh/5min), occupancy (fraction 0-1), speed (mph).

This dashboard has THREE charts. A sidebar "View" switch shows one chart at a
time, in this order, and ONLY the controls that belong to the chart on
screen — no dead widgets sitting around for a chart you're not looking at:

  1. Time Series        — classic per-sensor time series. Controls:
                          sensor, number of days.
  2. Topology Map        — a virtual node-link layout of the road network
                          (recovered from the published adjacency graph,
                          NOT real geographic coordinates — see the
                          blind-spot panel). Controls: metric, time window.
  3. Anomaly Screening  — flags "speed dropped while occupancy stayed flat"
                          events (a leading indicator of developing
                          congestion), then ranks locations by how often
                          it happens. Controls: speed threshold, occupancy
                          threshold, alarm level (1-5, controls color
                          sensitivity only).

Run:
    streamlit run app.py
"""

import os

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="PeMS04 Traffic Dashboard", page_icon="🚦", layout="wide")

# Use a path relative to this file, not the process's working directory —
# Streamlit Community Cloud runs the app from the repo root, not from this
# script's own folder, so a bare "data" would only work when run locally
# from inside HW1/.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
N_SENSORS = 307


# ============================================================== loaders
@st.cache_data
def load_raw():
    """Load the raw (time, sensor, feature) array. Cached because the file
    is ~33MB and re-parsing it on every widget interaction would make the
    whole app feel sluggish (the course's own v3 lesson)."""
    raw = np.load(f"{DATA_DIR}/pems04.npz")["data"]
    time_index = pd.date_range("2018-01-01", periods=raw.shape[0], freq="5min")
    return raw, time_index


@st.cache_data
def load_topology():
    """Load the published adjacency graph (from,to,cost) and lay it out as
    a virtual node-link diagram. Cached because kamada-kawai layout on a
    237-node graph is the single slowest step in the whole app."""
    dist = pd.read_csv(f"{DATA_DIR}/pems04_distance.csv")
    seq = pd.read_csv(f"{DATA_DIR}/sensor_sequence.csv").set_index("node_id")

    G = nx.DiGraph()
    for _, r in dist.iterrows():
        G.add_edge(int(r["from"]), int(r["to"]), weight=r["cost"])

    comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    pos = {}

    big_nodes = comps[0]
    big_pos = nx.kamada_kawai_layout(G.subgraph(big_nodes), weight="weight")
    xs = np.array([p[0] for p in big_pos.values()])
    ys = np.array([p[1] for p in big_pos.values()])
    for n, (x, y) in big_pos.items():
        pos[n] = (
            (x - xs.min()) / (xs.max() - xs.min() + 1e-9) * 10,
            (y - ys.min()) / (ys.max() - ys.min() + 1e-9) * 6 + 2.5,
        )

    row_y, col_x, row_gap = 1.2, 0.0, 0.85
    for comp in comps[1:]:
        ordered = sorted(comp, key=lambda n: seq.loc[n, "seq_pos"] if n in seq.index else 0)
        span = max(len(ordered) - 1, 1)
        for i, n in enumerate(ordered):
            pos[n] = (col_x + i * (10.0 / max(span, 8)), row_y)
        col_x_end = col_x + span * (10.0 / max(span, 8))
        row_y -= row_gap
        if row_y < -8:
            row_y, col_x = 1.2, col_x + 11.5
        else:
            col_x = 0.0 if col_x_end < 10.5 else col_x

    return G, pos


@st.cache_data
def sensor_malfunction_mask(_raw):
    """Flag (time, sensor) cells that look like device faults rather than
    real traffic: occupancy stuck at 0 for 15+ minutes, or speed repeating
    the exact same value for 25+ minutes. Excluded from anomaly screening
    so a broken detector doesn't get mistaken for a real event."""
    occ, speed = _raw[:, :, 1], _raw[:, :, 2]
    bad = np.zeros_like(occ, dtype=bool)
    for s in range(N_SENSORS):
        z = occ[:, s] == 0
        run_id = (~z).cumsum()
        run_len = pd.Series(z).groupby(run_id).transform("sum") * z
        bad[:, s] |= run_len.values >= 3

        same = pd.Series(speed[:, s]).diff() == 0
        run_id2 = (~same).cumsum()
        run_len2 = same.groupby(run_id2).transform("sum") * same
        bad[:, s] |= run_len2.values >= 5
    return bad


@st.cache_data
def metric_by_segment(_raw, metric: str, segment: str):
    """Median value per sensor for one metric, restricted to one time-of-day
    window. `metric` in {'speed','occupancy'}; `segment` in
    {'overall','morning_peak','evening_peak','daytime','nighttime'}."""
    col = {"speed": 2, "occupancy": 1}[metric]
    vals = _raw[:, :, col]
    hours = pd.date_range("2018-01-01", periods=_raw.shape[0], freq="5min").hour.values
    windows = {
        "overall": np.ones_like(hours, dtype=bool),
        "morning_peak": (hours >= 7) & (hours < 9),
        "evening_peak": (hours >= 17) & (hours < 19),
        "daytime": (hours >= 6) & (hours < 18),
        "nighttime": (hours >= 18) | (hours < 6),
    }
    mask = windows[segment]
    return np.nanmedian(vals[mask], axis=0)


@st.cache_data
def onset_events(_raw, speed_thresh: float, occ_thresh: float):
    """Detect 'speed dropped while occupancy stayed flat' onsets, relative
    to a rolling 15-minute anchor, then classify each onset by how long the
    drop persists (capped at 1 hour of tracking):

      - transient (<=15 min): resolves quickly, likely noise/minor braking
      - sustained (15-60 min): worth flagging for attention
      - long-running (hits the 1hr cap): worth flagging, cause unresolved

    This is a screening tool, not an incident detector: flagged points are
    candidates for human follow-up, not confirmed accidents.
    """
    occ, speed = _raw[:, :, 1], _raw[:, :, 2]
    n_time = _raw.shape[0]
    bad = sensor_malfunction_mask(_raw)

    def rolling_prev_mean(x, w=3):
        return pd.DataFrame(x).shift(1).rolling(w, min_periods=w).mean().values

    anchor_speed = rolling_prev_mean(speed)
    anchor_occ = rolling_prev_mean(occ)
    d_speed = (speed - anchor_speed) / np.where(anchor_speed == 0, np.nan, anchor_speed) * 100
    d_occ = (occ - anchor_occ) / np.where(anchor_occ == 0, np.nan, anchor_occ) * 100

    valid = ~bad & np.isfinite(d_occ) & np.isfinite(d_speed)
    onset = (np.abs(d_occ) <= occ_thresh) & (d_speed <= -speed_thresh) & valid

    MAX_H = 12  # 1 hour cap
    records = []
    for s in range(N_SENSORS):
        sp = speed[:, s]
        b = bad[:, s]
        for t in np.where(onset[:, s])[0]:
            thresh = anchor_speed[t, s] * (1 - speed_thresh / 100)
            h = 0
            while h < MAX_H and t + h < n_time and sp[t + h] < thresh and not b[t + h]:
                h += 1
            records.append((s, t, h * 5, h == MAX_H))

    ev = pd.DataFrame(records, columns=["sensor", "start_t", "duration_min", "capped"])
    if ev.empty:
        return ev, np.zeros(N_SENSORS, dtype=int)
    ev["category"] = np.select(
        [ev["duration_min"] <= 15, ~ev["capped"]],
        ["transient", "sustained"],
        default="long_running",
    )
    flagged = ev[ev["category"] != "transient"]
    counts = flagged.groupby("sensor").size().reindex(range(N_SENSORS), fill_value=0).values
    return ev, counts


def alarm_level_bounds(level: int):
    """level 1 (lenient) -> level 5 (strict). Controls ONLY the color
    sensitivity of the screening map, not the underlying counts."""
    green_pct = 40 + (level - 1) * 5
    yellow_pct = 80 + (level - 1) * 3.75
    return green_pct, yellow_pct


GYR_SCALE = [[0, "#1E8449"], [0.5, "#E8C31E"], [1, "#C0392B"]]   # green -> yellow -> red
RGY_SCALE = [[0, "#C0392B"], [0.5, "#E8C31E"], [1, "#1E8449"]]   # red -> yellow -> green


def draw_network_plotly(pos, G, values, colorscale, cmin, cmax, title, cbar_title,
                         hover_fmt="{v:.1f}", highlight_top=0):
    """An interactive (clickable) version of the node-link topology chart.
    Points are real Plotly markers, so a click can be read back via
    st.plotly_chart(..., on_select='rerun') and used to jump to that
    sensor's Time Series."""
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        edge_x += [x1, x2, None]
        edge_y += [y1, y2, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(color="#D5D8DC", width=0.7),
                             hoverinfo="skip", showlegend=False)

    nodes = list(pos.keys())
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    c = [values[n] for n in nodes]
    labels = [f"D4-{n:03d}" for n in nodes]
    hover = [f"{lab}<br>{hover_fmt.format(v=val)}" for lab, val in zip(labels, c)]

    node_trace = go.Scatter(
        x=xs, y=ys, mode="markers", customdata=labels, text=hover,
        hovertemplate="%{text}<extra></extra>",
        marker=dict(size=9, color=c, colorscale=colorscale, cmin=cmin, cmax=cmax,
                    colorbar=dict(title=cbar_title), line=dict(width=0.6, color="#333333")),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    if highlight_top:
        top_idx = np.argsort(-np.array(c))[:highlight_top]
        for i in top_idx:
            fig.add_annotation(x=xs[i], y=ys[i], text=labels[i], showarrow=False,
                                font=dict(size=10, color="#000000"), yshift=10)
    fig.update_layout(
        title=title, height=620, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        clickmode="event+select",
    )
    return fig


def point_picker(event, key_suffix: str):
    """Show 'click a point to jump to its Time Series' UI under a network
    chart, given the event returned by st.plotly_chart(..., on_select='rerun')."""
    points = event.selection.points if event and event.selection else []
    if not points:
        st.caption("Click a point on the map to look it up in Time Series.")
        return
    clicked = points[0]["customdata"][0] if isinstance(points[0]["customdata"], (list, tuple)) \
        else points[0]["customdata"]
    if st.button(f"🔎 View {clicked} in Time Series", key=f"jump_{key_suffix}"):
        # Widgets with keys "view_radio"/"sensor_select" are already instantiated
        # in this run, so their session_state can't be written directly here.
        # Stash the target and apply it at the top of the NEXT run, before
        # those widgets are created.
        st.session_state["_pending_jump"] = clicked
        st.rerun()


def blind_spot(title: str, bullets: list[str]):
    with st.container(border=True):
        st.markdown(f"**⛔ What this page cannot tell you — {title}**")
        for b in bullets:
            st.markdown(f"- {b}")


# ============================================================== data
raw, time_index = load_raw()
G, pos = load_topology()

# Apply any pending "jump to this sensor's Time Series" request BEFORE the
# view_radio / sensor_select widgets below are instantiated for this run.
if st.session_state.get("_pending_jump"):
    st.session_state["view_radio"] = "📈 Time Series"
    st.session_state["sensor_select"] = st.session_state.pop("_pending_jump")

# ============================================================== sidebar
VIEWS = ["📈 Time Series", "🗺️ Topology Map", "⚠️ Anomaly Screening"]
with st.sidebar:
    st.header("Controls")
    view = st.radio("View", VIEWS, key="view_radio")
    st.divider()

    # Only the controls that belong to the chart on screen are shown here —
    # switching View swaps the whole control block, not just the chart.
    if view == "📈 Time Series":
        st.caption("Time Series")
        sensor = st.selectbox("Sensor", [f"D4-{i:03d}" for i in range(N_SENSORS)], key="sensor_select")
        days = st.slider("Days of history", 1, 59, 7)

    elif view == "🗺️ Topology Map":
        st.caption("Topology Map")
        metric = st.selectbox("Metric", ["speed", "occupancy"])
        segment = st.selectbox(
            "Time window",
            ["overall", "morning_peak", "evening_peak", "daytime", "nighttime"],
            format_func=lambda s: {
                "overall": "Overall median",
                "morning_peak": "Morning peak (07-09)",
                "evening_peak": "Evening peak (17-19)",
                "daytime": "Daytime (06-18)",
                "nighttime": "Nighttime (18-06)",
            }[s],
        )

    else:  # Anomaly Screening
        st.caption("Anomaly Screening")
        speed_thresh = st.slider("Speed change threshold (%)", 0.5, 10.0, 1.0, 0.5)
        occ_thresh = st.slider("Occupancy change threshold (%)", 0.5, 10.0, 2.0, 0.5)
        alarm_level = st.slider("Alarm level (1 = lenient, 5 = strict)", 1, 5, 3)

st.title("PeMS04 Traffic Dashboard")
st.caption("Caltrans PeMS, District 4 (San Francisco Bay Area) · 307 loop detectors · "
           "5-minute intervals · 2018-01-01 to 2018-02-28 · CIVL 6962 Homework 1")

VIEW_DESCRIPTIONS = {
    "📈 Time Series": "Pick a sensor and a date range to see its speed, occupancy, and flow over time.",
    "🗺️ Topology Map": "Speed or occupancy across the whole network's virtual layout, by time window — click a point to jump to its Time Series.",
    "⚠️ Anomaly Screening": "Locations where speed drops without occupancy rising — a possible early sign of trouble — click a point to inspect it.",
}
st.info(VIEW_DESCRIPTIONS[view], icon="ℹ️")

# ============================================================== views
if view == "📈 Time Series":
    sensor_idx = int(sensor.split("-")[1])
    one_speed = raw[:, sensor_idx, 2]
    one_occ = raw[:, sensor_idx, 1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean speed (this sensor)", f"{one_speed.mean():.1f} mph")
    c2.metric("Mean occupancy (this sensor)", f"{one_occ.mean():.3f}")
    c3.metric("Days shown", f"{days}")

    one = pd.DataFrame({"time": time_index, "speed": raw[:, sensor_idx, 2],
                         "occupancy": raw[:, sensor_idx, 1], "flow": raw[:, sensor_idx, 0]})
    one = one[one["time"] < one["time"].min() + pd.Timedelta(days=days)]
    st.line_chart(one, x="time", y="speed", y_label="speed (mph)")
    col_a, col_b = st.columns(2)
    with col_a:
        st.scatter_chart(one, x="occupancy", y="flow", x_label="occupancy (fraction)",
                          y_label="flow (veh/5min)")
    with col_b:
        st.line_chart(one, x="time", y="occupancy", y_label="occupancy (fraction)")
    st.download_button("⬇ Download these rows", one.to_csv(index=False).encode(),
                        file_name=f"{sensor}.csv", mime="text/csv")

elif view == "🗺️ Topology Map":
    c1, c2 = st.columns(2)
    c1.metric("Sensors in network", f"{N_SENSORS}")
    c2.metric("Days of data", "59")

    values = metric_by_segment(raw, metric, segment)
    unit = "mph" if metric == "speed" else "fraction"
    vmin, vmax = (50, 90) if metric == "speed" else (float(np.nanmin(values)), float(np.nanmax(values)))
    scale = RGY_SCALE if metric == "speed" else GYR_SCALE  # speed: red=slow,green=fast
    values_by_node = {n: values[n] for n in pos.keys()}
    fig = draw_network_plotly(
        pos, G, values_by_node, scale, vmin, vmax,
        title=f"{metric} — {segment.replace('_',' ')}",
        cbar_title=unit, hover_fmt="{v:.1f} " + unit,
    )
    event = st.plotly_chart(fig, width="stretch", on_select="rerun",
                             selection_mode="points", key="topo_select")
    point_picker(event, "topo")
    st.caption(
        "Node positions are a virtual layout recovered from the published road-adjacency "
        "graph (kamada-kawai), NOT real GPS coordinates — see the blind-spot panel below."
    )

else:  # Anomaly Screening
    ev, counts = onset_events(raw, speed_thresh, occ_thresh)
    g_pct, y_pct = alarm_level_bounds(alarm_level)
    g_val, y_val = np.percentile(counts, g_pct), np.percentile(counts, y_pct)
    cat = np.select([counts <= g_val, counts <= y_val], [0, 1], default=2)
    cat_by_node = {n: cat[n] for n in pos.keys()}

    c1, c2 = st.columns(2)
    c1.metric("Total onsets (59 days)", f"{len(ev):,}")
    c2.metric("Sustained/long-running", f"{(ev['category']!='transient').sum():,}" if len(ev) else "0")

    fig3 = draw_network_plotly(
        pos, G, cat_by_node, GYR_SCALE, 0, 2,
        title=f"Anomaly screening — alarm level {alarm_level} "
              f"(red = top {100-y_pct:.1f}% of flagged locations)",
        cbar_title="0=green 1=yellow 2=red", hover_fmt="{v:.0f} flagged-count tier",
        highlight_top=5,
    )
    event3 = st.plotly_chart(fig3, width="stretch", on_select="rerun",
                              selection_mode="points", key="anomaly_select")
    point_picker(event3, "anomaly")
    st.caption(
        f"Speed threshold {speed_thresh}% / occupancy threshold {occ_thresh}% / "
        f"alarm level {alarm_level}."
    )
    st.download_button("⬇ Download flagged events", ev.to_csv(index=False).encode(),
                        file_name="anomaly_events.csv", mime="text/csv")

# ============================================================== provenance
with st.expander("Data provenance"):
    st.markdown(
        "- **Who**: California Department of Transportation (Caltrans), Performance "
        "Measurement System (PeMS).\n"
        "- **Where**: District 4 (San Francisco Bay Area) freeway loop detectors.\n"
        "- **When**: 2018-01-01 through 2018-02-28, 5-minute aggregation.\n"
        "- **Instrument**: inductive loop detectors embedded in the pavement, reporting "
        "flow (vehicle count), occupancy (fraction of time occupied), and average speed "
        "per 5-minute interval.\n"
        "- **Adjacency graph**: `distance.csv`, published alongside the PeMS04 benchmark "
        "by the ASTGCN paper's authors (Davidham3/ASTGCN on GitHub) — 307 nodes, 340 "
        "directed edges, validated against this dataset by comparing occupancy correlation "
        "on connected sensor pairs (0.69) vs. random pairs (0.57)."
    )

# ============================================================== blind spot
blind_spot("Loop detector data & this dashboard's own assumptions", [
    "Node positions on the Topology Map are a virtual graph layout, not real GPS "
    "coordinates — no public source maps PeMS04's sensor indices to real locations, "
    "so shapes and distances on the map carry no geographic meaning.",
    "The Anomaly Screening panel flags statistical patterns (speed drop with flat "
    "occupancy, not resolving quickly) — it has not been checked against any confirmed "
    "incident log, so 'flagged' means 'worth a human look', not 'confirmed accident'.",
    "Roads without a sensor are invisible here entirely — coverage follows freeway "
    "mainlines only, with no arterial streets.",
    "About 19% of sensors (59/307) have a detected fault window (stuck readings or "
    "zeroed occupancy) at some point in the 59 days; their anomaly counts may be less "
    "reliable than the rest.",
])

st.caption(
    "A dashboard is a claim about what the data can tell you. The blind-spot panel is "
    "where that claim stops being written down — for everyone else, and for you in six months."
)
