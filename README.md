# cwpAI Replit Viewer

This folder is the lightweight Replit version of the dashboard. It only views prebuilt dashboard files.

## Setup

Copy these into this folder before uploading it to Replit:

- `dashboard.py` from `unrealai/unrealai/dashboard.py`
- the whole `dashboard_data/` folder from `unrealai/unrealai/dashboard_data/`

The final Replit file tree should look like:

```text
app.py
dashboard.py
requirements.txt
.replit
dashboard_data/
  summary.json
  morning_report.csv
  symbol_metrics.csv
  symbol_timeseries.csv
  trade_log.csv
  aggregate_timeseries.csv
```

Parquet files are optional. CSV files are enough.

## Run

Replit should install from this folder's `requirements.txt`, then run:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 3000 --server.headless true
```

## Updating Data

Run `live_report.py` locally, then replace the files inside `dashboard_data/` on Replit.

From the main repo, you can publish this folder to GitHub with:

```bash
./scripts/publish_replit_viewer_branch.sh
```

That updates the `replit-viewer` branch. Import that branch into Replit so the Replit app only sees the lightweight viewer files.

The Replit launcher sets `DASHBOARD_DATA_BASE_URL` to the raw GitHub URL for the `replit-viewer` branch. That lets an already-published app load fresh CSV data from GitHub after each publish script run, without requiring a full Replit redeploy for data-only updates.
