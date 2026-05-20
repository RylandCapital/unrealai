from pathlib import Path
import os
import runpy


APP_DIR = Path(__file__).resolve().parent
DASHBOARD_PATH = APP_DIR / "dashboard.py"

os.environ.setdefault(
    "DASHBOARD_DATA_BASE_URL",
    "https://raw.githubusercontent.com/RylandCapital/unrealai/replit-viewer/dashboard_data",
)

runpy.run_path(str(DASHBOARD_PATH), run_name="__main__")
