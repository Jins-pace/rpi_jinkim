# CIVL 6962 — Homework 1: PeMS04 Traffic Dashboard

## What this is

A single-dataset Streamlit dashboard built on **PeMS04** (Caltrans PeMS, District 4 —
San Francisco Bay Area), 307 loop detectors, 5-minute intervals, 2018-01-01 to 2018-02-28.

Three charts, each with real controls:

| Chart | Type | Controls |
|---|---|---|
| Topology Map | node-link network diagram | metric (speed/occupancy), time window (overall/morning peak/evening peak/daytime/nighttime) |
| Time Series | line + scatter | sensor, days of history |
| Anomaly Screening | node-link network diagram | speed change threshold, occupancy change threshold, alarm level (1-5) |

## Files

- `app.py` — the dashboard (run this)
- `analysis.ipynb` — the exploratory analysis behind the dashboard: topology recovery,
  sensor sequencing, and the design/validation of the anomaly-screening logic
- `requirements.txt`
- `data/pems04.npz` — raw data (flow, occupancy, speed × 307 sensors × 16,992 timesteps)
- `data/pems04_distance.csv` — published road-adjacency graph (from, to, cost), used to lay
  out the virtual topology map
- `data/sensor_sequence.csv` — precomputed per-sensor position along its road (see notebook §4)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Push this folder to a GitHub repo (`app.py`, `requirements.txt`, and `data/` — everything
   here is well under the ~50MB limit).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub → **Create app**
   → **Deploy a public app from GitHub**.
3. Point it at the repo, branch, and `app.py`'s path. Under **Advanced**, set the Python
   version you developed with.
4. Watch the build log (2-4 minutes the first time).
5. Test the live URL in a private/incognito window before submitting.

Common failure modes (from the course material): `FileNotFoundError` almost always means a
data file didn't get committed (check `.gitignore` and `git status`); `ModuleNotFoundError`
means something is missing from `requirements.txt`; a blank page with no error usually means
`st.set_page_config()` wasn't the first Streamlit call, or an exception was swallowed inside a
cached function.

## Key limitations (see the in-app blind-spot panel for the full list)

- The Topology Map's node positions are a **virtual graph layout**, not real GPS coordinates —
  PeMS04 does not publish sensor locations anywhere, only the adjacency graph.
- The Anomaly Screening panel flags a statistical pattern (speed drop with flat occupancy that
  doesn't resolve quickly) — it is a **screening tool for human follow-up**, not a validated
  incident detector; there is no ground-truth accident log to check it against.
- Roads without a sensor are invisible; coverage is freeway mainlines only.
- ~19% of sensors (59/307) have a detected fault window (stuck readings or zeroed occupancy)
  at some point in the 59-day period.
