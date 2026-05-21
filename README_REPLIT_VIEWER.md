# Replit Dashboard Viewer

This setup runs only the Streamlit dashboard viewer. It does not train models, update market data, or generate live reports.

## Files Replit Needs

Upload or commit this folder:

- `unrealai/dashboard.py`
- `requirements.viewer.txt`
- `.replit`
- `unrealai/dashboard_data/summary.json`
- `unrealai/dashboard_data/morning_report.csv`
- `unrealai/dashboard_data/symbol_metrics.csv`
- `unrealai/dashboard_data/symbol_timeseries.csv`
- `unrealai/dashboard_data/trade_log.csv`
- `unrealai/dashboard_data/aggregate_timeseries.csv`

Parquet files are optional. The viewer falls back to CSV when parquet support is unavailable.

## Replit Install Command

Use the lightweight viewer requirements, not the full training requirements:

```bash
pip install -r requirements.viewer.txt
```

## Run

The included `.replit` runs:

```bash
streamlit run unrealai/dashboard.py --server.address 0.0.0.0 --server.port 3000 --server.headless true
```

## Updating The Dashboard

Generate fresh dashboard files locally with `live_report.py`, then replace the files in `unrealai/dashboard_data/` on Replit.

GRIP/EDIP allocation spreadsheets are optional. If they are not uploaded, the viewer hides those filters and shows `ALL SYMBOLS`.

## GitHub Branch Automation

The easiest automated path is to publish the viewer to a dedicated Git branch:

```bash
./scripts/publish_replit_viewer_branch.sh
```

That script:

- refreshes `replit_viewer/dashboard.py`
- refreshes `replit_viewer/dashboard_data/`
- keeps `dist/replit_viewer.zip`
- publishes a `replit-viewer` branch whose repository root is just the viewer app

In Replit, import the `replit-viewer` branch from GitHub. After each morning update, rerun the script and Replit can pull/redeploy from that branch.
