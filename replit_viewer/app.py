from pathlib import Path
import runpy


APP_DIR = Path(__file__).resolve().parent
DASHBOARD_PATH = APP_DIR / "dashboard.py"

runpy.run_path(str(DASHBOARD_PATH), run_name="__main__")
