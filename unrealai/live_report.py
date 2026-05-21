from __future__ import annotations

import base64
import importlib.util
import io
import json
import multiprocessing
import os
import random
import re
import shutil
import time
import webbrowser
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "0")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from plotly.subplots import make_subplots
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False
try:
    from utils.feature_registry import FEATURE_COLS as REGISTRY_FEATURE_COLS
except ModuleNotFoundError:
    from unrealai.utils.feature_registry import FEATURE_COLS as REGISTRY_FEATURE_COLS
try:
    from utils.feature_compat import BACKFILLABLE_FEATURE_COLS, ensure_backfilled_feature_columns
except ModuleNotFoundError:
    from unrealai.utils.feature_compat import BACKFILLABLE_FEATURE_COLS, ensure_backfilled_feature_columns

load_dotenv()
APP_DIR = Path(__file__).resolve().parent


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return int(default)
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


class CFG:

    # Report window
    # Prefer REPORT_DAYS_BACK so the report stays inside the test CSV window.
    # Example: REPORT_DAYS_BACK=180 uses the last 180 rows of each symbol's test file.
    # REPORT_START_DATE remains available as an explicit override when set.
    REPORT_DAYS_BACK = env_int("REPORT_DAYS_BACK", 180)
    REPORT_START_DATE = (os.getenv("REPORT_START_DATE", "").strip() or None)
    REPORT_USE_COMMON_START_DATE = env_bool("REPORT_USE_COMMON_START_DATE", True)

    # Core
    SEED = 42
    WINDOW_SIZE = 21
    INITIAL_CASH = 100_000_000
    AGGREGATE_INITIAL_CASH = env_float("AGGREGATE_INITIAL_CASH", 1_000_000.0)

    # Keep these aligned with the training script
    COOLDOWN_DAYS = 3
    MIN_HOLD_DAYS = 10
    COOLDOWN_DAYS_AFTER_PROFITABLE_CLOSE = 1
    COOLDOWN_DAYS_AFTER_LOSING_CLOSE = 3
    COOLDOWN_DAYS_AFTER_STOP_EXIT = 1
    COOLDOWN_DAYS_AFTER_TAKE_PROFIT_EXIT = 0
    SLIPPAGE_RATE = 0.0000
    STOP_LOSS_TEST = .15
    TAKE_PROFIT_TEST = 1000.0

    # Optional LLM overwatch, mirroring the test-set review flow in main_w_llm.py
    OVERWATCH_ENABLED = env_bool("OVERWATCH_ENABLED_LIVE", False)
    OVERWATCH_EVERY_N_STEPS = env_int("OVERWATCH_EVERY_N_STEPS", 5)
    OVERWATCH_REASONING_EFFORT = "high"
    MODEL = os.getenv("OVERWATCH_MODEL", "gpt-5-mini")
    OVERWATCH_ALLOW_CONSTRAINT_OVERRIDE = True
    SAVE_OVERWATCH_CHARTS = True
    OVERWATCH_CHARTS_DIR = str(APP_DIR / "overwatch_charts" / "live_report")
    NUM_WORKERS = env_int("LIVE_REPORT_NUM_WORKERS", 1)
    CONSOLE_PROGRESS = True
    SIGNAL_PROGRESS_EVERY_N_STEPS = env_int("SIGNAL_PROGRESS_EVERY_N_STEPS", 25)
    OVERWATCH_VERBOSE_PROGRESS = True

    # Paths
    MODEL_DIR = os.getenv("MODEL_DIR", str(APP_DIR / "models"))
    DATA_DIR = os.getenv("DATA_DIR", str(APP_DIR / "testdata"))
    OUTPUT_DIR = os.getenv("LIVE_REPORT_OUTPUT_DIR", str(APP_DIR / "live_reports"))
    DASHBOARD_DIR = os.getenv("DASHBOARD_DIR", str(APP_DIR / "dashboard_data"))
    REPORT_VWAP_MODULE_PATH = os.getenv("REPORT_VWAP_MODULE_PATH", str(APP_DIR / "utils" / "overwatch_chart.py"))
    AGGREGATE_BENCHMARK_SYMBOLS = [
        s.strip().upper()
        for s in os.getenv("AGGREGATE_BENCHMARK_SYMBOLS", "EQAL,RSP,QQQE").split(",")
        if s.strip()
    ]
    DATA_ONLY_SYMBOLS = [
        s.strip().upper()
        for s in os.getenv(
            "DATA_ONLY_SYMBOLS",
            "EQAL,RSP,QQQE,QQQ,SPY,DIA,IWF,XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,SMH",
        ).split(",")
        if s.strip()
    ]


    # None = auto-discover from DATA_DIR
    SYMBOLS = None

    # If True, script expects open/high/low or adjusted_open/adjusted_high/adjusted_low
    REQUIRE_EVAL_PRICES = True

    # Fill modes identical to your test script
    FILL_MODE_CLOSE_T = "close_t"
    FILL_MODE_OPEN_T1 = "open_t_plus_1"
    FILL_MODE_HL2_T1 = "hl2_t_plus_1"
    ALL_FILL_MODES = [
        FILL_MODE_CLOSE_T,
        FILL_MODE_OPEN_T1,
        FILL_MODE_HL2_T1,
    ]

    # The fill mode used for the morning stance / long-vs-flat report and dashboard positioning
    PRIMARY_REPORT_FILL_MODE = FILL_MODE_OPEN_T1

    # Output controls
    WRITE_LEGACY_SYMBOL_CSVS = True
    WRITE_DASHBOARD_PARQUET = True
    WRITE_DASHBOARD_CSV_FALLBACK = True
    WRITE_PLOTLY_HTML = True
    OPEN_PLOTLY_HTML = False


ACTION_HOLD = 0
ACTION_LONG = 1
ACTION_CLOSE = 2

FEATURE_COLS = list(REGISTRY_FEATURE_COLS)

FILL_LABELS = {
    CFG.FILL_MODE_CLOSE_T: "Close T0",
    CFG.FILL_MODE_OPEN_T1: "Open T+1",
    CFG.FILL_MODE_HL2_T1: "HL2 T+1",
}

OVERWATCH_LOG_COLUMNS = [
    "symbol",
    "step",
    "date",
    "position",
    "cooldown_remaining",
    "valid_actions",
    "raw_action",
    "final_action",
    "overrode",
    "model_action",
    "model_reply",
    "model_action_invalid",
    "constraint_override",
    "constraint_override_reason",
    "chart_path",
    "error",
]


@dataclass
class DashboardSummary:
    as_of: str
    report_start_date: str
    primary_fill_mode: str
    total_symbols: int
    open_positions: int
    closed_positions: int
    total_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    max_drawdown_pct: float
    win_rate_pct: float
    avg_trade_return_pct: float
    exposure_pct: float
    total_trades: int


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _try_write_parquet(df: pd.DataFrame, path: Path) -> bool:
    if not CFG.WRITE_DASHBOARD_PARQUET:
        return False
    try:
        df.to_parquet(path, index=False)
        return True
    except Exception as exc:
        print(f"[warn] parquet write failed for {path.name}: {exc}")
        return False


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("SECRET")
OVERWATCH_MODEL = os.getenv("OVERWATCH_MODEL", CFG.MODEL)


def _get_openai_client():
    if not hasattr(_get_openai_client, "_client"):
        if not OPENAI_API_KEY:
            raise RuntimeError("Missing API key. Set OPENAI_API_KEY or SECRET.")
        from openai import OpenAI
        _get_openai_client._client = OpenAI(api_key=OPENAI_API_KEY)
    return _get_openai_client._client


def overwatch_decision(raw_action: int, img_b64_str: str) -> str:
    client = _get_openai_client()

    instructions = (
        "You are lead equity analyst on the tradedesk. You are overseeing a MOMENTUM reinforcement learning trading agent "
        "that decides whether to do hold(0), go long(1), close(2), each day. "
        "This is LONG-ONLY, quantitative trading agent that you are supervising. "
        "Periodically you will get a chart with a 10Y monthly chart, 10yr weekly chart, and a 1yr daily chart. "
        "Each chart has 3 anchored VWAPS: most recent high (blue), most recent low (pink), highest volume day (yellow). "
        "You are given the current price and YTD return in the daily chart title. The chart also informs you if we are "
        "currently long/flat as well as the current trade's PnL, and you are given relative strength against SPY and Sector as well.  "
        "For your information please note the agents strategy has a 5 day minimum trade threshold, a 5 day cooldown period after certain closes,"
        " and it uses a 15% stop loss with no fixed take profit cap. "
        "Use all timeframes to analyze price action patterns, in conjunction with the vwaps, trade history data, Relative Strength data, and Technicals to assess the stock. "
        "Do not use the internet or outside information. "
        "Your decision must not include any information from the future. "
        "You may still choose LONG or CLOSE even if cooldown or minimum-hold would normally block it, this is very important "
        "The agent is trained on past information, so regimes may change that are unusual for that stock, and sometimes it is necessary to break the usual rules. "
        "YOUR JOB IS TO BEAT BUY AND HOLD OF THAT STOCK. "
        "Reply ONLY with: <number choice> - <two-sentence reason>."
    )

    image_data_url = f"data:image/png;base64,{img_b64_str}"
    payload = {"trading_action": int(raw_action)}

    resp = client.responses.create(
        model=OVERWATCH_MODEL,
        reasoning={"effort": str(CFG.OVERWATCH_REASONING_EFFORT)},
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instructions},
                    {"type": "input_text", "text": json.dumps(payload)},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
    )
    return resp.output_text.strip()


def get_chart_module():
    if not hasattr(get_chart_module, "_module"):
        file = Path(CFG.REPORT_VWAP_MODULE_PATH)
        spec = importlib.util.spec_from_file_location("overwatch_chart", file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module spec from: {file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        get_chart_module._module = module
    return get_chart_module._module


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def close_figure(fig):
    import matplotlib.pyplot as plt

    plt.close(fig)


def build_overwatch_logs_df(logs) -> pd.DataFrame:
    return pd.DataFrame(list(logs), columns=OVERWATCH_LOG_COLUMNS)


def log_progress(message: str) -> None:
    if not bool(getattr(CFG, "CONSOLE_PROGRESS", True)):
        return
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def save_overwatch_chart(fig, *, symbol: str, review_date: str, step: int, raw_action: int, source_tag: str) -> Optional[str]:
    if not bool(getattr(CFG, "SAVE_OVERWATCH_CHARTS", False)):
        return None

    root = Path(getattr(CFG, "OVERWATCH_CHARTS_DIR", "")).expanduser()
    if not str(root):
        return None

    symbol_dir = root / str(symbol)
    symbol_dir.mkdir(parents=True, exist_ok=True)

    stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"{review_date}_step{int(step):05d}_raw{int(raw_action)}_{source_tag}_{stamp}.png"
    output_path = symbol_dir / filename
    fig.savefig(output_path, format="png", bbox_inches="tight")
    return str(output_path)


def action_name(action: int) -> str:
    return {ACTION_HOLD: "HOLD", ACTION_LONG: "LONG", ACTION_CLOSE: "CLOSE"}.get(int(action), str(action))


def _format_pct_or_na(value: Optional[float], decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100.0:+.{decimals}f}%"


def _format_bool_or_na(value: Optional[bool]) -> str:
    if value is None:
        return "N/A"
    return "Yes" if bool(value) else "No"


def _get_env_raw_feature(env, feature_name: str) -> Optional[float]:
    raw_features = getattr(env, "raw_features", None)
    feature_cols = list(getattr(env, "feature_cols", []))
    if raw_features is None or feature_name not in feature_cols:
        return None
    idx = min(max(int(getattr(env, "current_step", 0)), 0), raw_features.shape[0] - 1)
    col_idx = feature_cols.index(feature_name)
    return float(raw_features[idx, col_idx])


def build_overwatch_decision_card(env, *, raw_action: int, valid_actions: list[int]) -> str:
    pos_map = {0: "Flat", 1: "Long"}
    pos_str = pos_map.get(int(env.position), "Unknown")

    if env.position != 0 and env.entry_equity:
        pnl_pct = (env.equity - env.entry_equity) / env.entry_equity * 100.0
        days = env.current_step - env.position_open_step + 1
        cooldown_display = "N/A"
        min_hold_display = _format_bool_or_na(bool(env.can_discretionary_close()))
    else:
        pnl_pct = 0.0
        days = 0
        cooldown_display = str(int(getattr(env, "cooldown_remaining", 0)))
        min_hold_display = "N/A"

    idx = min(max(int(getattr(env, "current_step", 0)), 0), int(getattr(env, "n_steps", 1)) - 1)
    prices = np.asarray(getattr(env, "prices", []), dtype=np.float64)
    current_price = float(prices[idx]) if len(prices) else np.nan

    sma200 = np.nan
    if len(prices):
        start_200 = max(0, idx - 199)
        sma200_window = prices[start_200:idx + 1]
        if len(sma200_window) >= 200:
            sma200 = float(np.mean(sma200_window))

    above_200 = None if pd.isna(sma200) or sma200 == 0 else (current_price >= sma200)

    dist_52w_high = np.nan
    if len(prices):
        start_52w = max(0, idx - 251)
        high_52w = float(np.max(prices[start_52w:idx + 1]))
        if high_52w > 0:
            dist_52w_high = (current_price / high_52w) - 1.0

    rs_spy_21 = _get_env_raw_feature(env, "rel_ret_21_vs_spy")
    rs_sector_21 = _get_env_raw_feature(env, "rel_ret_21_vs_sector")
    range_compression = _get_env_raw_feature(env, "range_compression_20")
    range_compression_display = _format_bool_or_na(None if range_compression is None else (range_compression >= 0.5))

    valid_str = "/".join(action_name(a) for a in valid_actions)

    return (
        f"Position: {pos_str}\n"
        f"PnL: {pnl_pct:.2f}%\n"
        f"Days: {days}\n"
        f"Cooldown: {cooldown_display}\n"
        f"Raw RL: {action_name(raw_action)} | Valid: {valid_str}\n"
        f"Min Hold OK: {min_hold_display}\n"
        f"RS21 SPY/Sector: {_format_pct_or_na(rs_spy_21)} / {_format_pct_or_na(rs_sector_21)}\n"
        f"Above 200d: {_format_bool_or_na(above_200)} | 52wH: {_format_pct_or_na(dist_52w_high)}\n"
        f"Range Compression: {range_compression_display}"
    )


def resolve_constraint_override(env, model_action: int) -> tuple[bool, str]:
    if not bool(getattr(CFG, "OVERWATCH_ALLOW_CONSTRAINT_OVERRIDE", False)):
        return False, ""
    if env is None:
        return False, ""

    pos = int(getattr(env, "position", 0))
    cooldown_remaining = int(getattr(env, "cooldown_remaining", 0))

    if pos == 0 and model_action == ACTION_LONG and cooldown_remaining > 0:
        return True, "cooldown"

    can_close = bool(getattr(env, "can_discretionary_close", lambda: False)())
    if pos != 0 and model_action == ACTION_CLOSE and not can_close:
        return True, "min_hold"

    return False, ""


def reset_overwatch_chart_storage() -> None:
    if not bool(getattr(CFG, "OVERWATCH_ENABLED", False)):
        return
    if not bool(getattr(CFG, "SAVE_OVERWATCH_CHARTS", False)):
        return

    root = Path(getattr(CFG, "OVERWATCH_CHARTS_DIR", "")).expanduser()
    if not str(root):
        return

    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    log_progress(f"Reset overwatch chart storage: {root}")


def set_global_determinism(seed: int):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


@tf.keras.utils.register_keras_serializable(package="Custom")
def mean_center_advantage(x):
    return x - tf.reduce_mean(x, axis=1, keepdims=True)


class ZScoreScaler:
    def __init__(self, feature_cols):
        self.feature_cols = list(feature_cols)
        self.mean_ = None
        self.std_ = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not loaded.")
        out = df.copy()
        vals = out[self.feature_cols].to_numpy(dtype=np.float32)
        vals = (vals - self.mean_) / self.std_
        out[self.feature_cols] = vals
        return out


def load_scaler(path: str):
    z = np.load(path, allow_pickle=True)
    scaler = ZScoreScaler(list(z["feature_cols"]))
    scaler.mean_ = z["mean"].astype(np.float32)
    scaler.std_ = z["std"].astype(np.float32)
    return scaler


def load_keras_model(path: str):
    try:
        return tf.keras.models.load_model(path, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(path)


def get_checkpoint_paths(base_dir: str, symbol: str) -> dict:
    symbol_dir = os.path.join(str(base_dir), str(symbol))
    return {
        "dir": symbol_dir,
        "policy": os.path.join(symbol_dir, "policy.keras"),
        "target": os.path.join(symbol_dir, "target.keras"),
        "scaler": os.path.join(symbol_dir, "scaler.npz"),
        "state": os.path.join(symbol_dir, "agent_state.json"),
    }


def load_agent_state_json(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def first_existing_col(cols, candidates):
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def resolve_eval_price_columns(df_cols):
    open_col = first_existing_col(df_cols, ["adjusted_open", "adj_open", "open"])
    high_col = first_existing_col(df_cols, ["adjusted_high", "adj_high", "high"])
    low_col = first_existing_col(df_cols, ["adjusted_low", "adj_low", "low"])
    return open_col, high_col, low_col


def load_csv(csv_path, *, require_eval_prices=False):
    header = pd.read_csv(csv_path, nrows=0)
    existing_cols = header.columns.tolist()
    existing_set = set(existing_cols)

    missing_features = [c for c in FEATURE_COLS if c not in existing_set]
    missing_non_backfillable = [c for c in missing_features if c not in BACKFILLABLE_FEATURE_COLS]
    required_core = [c for c in ["Date", "adjusted_close"] if c not in existing_set]
    if required_core or missing_non_backfillable:
        raise ValueError(
            f"CSV missing required columns: {required_core + missing_non_backfillable}\nFile: {csv_path}"
        )

    open_col, high_col, low_col = resolve_eval_price_columns(existing_cols)
    if require_eval_prices and (open_col is None or high_col is None or low_col is None):
        raise ValueError(
            "CSV must contain open/high/low or adjusted_open/adjusted_high/adjusted_low.\n"
            f"File: {csv_path}"
        )

    usecols = ["Date"] + [c for c in FEATURE_COLS if c in existing_set] + ["adjusted_close"]
    for c in [open_col, high_col, low_col]:
        if c is not None and c not in usecols:
            usecols.append(c)

    dtypes = {c: "float32" for c in usecols if c != "Date"}
    df = pd.read_csv(csv_path, usecols=usecols, parse_dates=["Date"], dtype=dtypes)
    df = df.sort_values("Date").reset_index(drop=True)

    if open_col is not None:
        df["eval_open"] = df[open_col].astype(np.float32)
    if high_col is not None:
        df["eval_high"] = df[high_col].astype(np.float32)
    if low_col is not None:
        df["eval_low"] = df[low_col].astype(np.float32)

    if not require_eval_prices:
        if "eval_open" not in df.columns:
            df["eval_open"] = df["adjusted_close"].astype(np.float32)
        if "eval_high" not in df.columns:
            df["eval_high"] = df["adjusted_close"].astype(np.float32)
        if "eval_low" not in df.columns:
            df["eval_low"] = df["adjusted_close"].astype(np.float32)

    df = ensure_backfilled_feature_columns(df)

    missing_after_backfill = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_after_backfill:
        raise ValueError(
            f"CSV is still missing required feature columns after backfill: {missing_after_backfill}\n"
            f"File: {csv_path}"
        )

    needed_check_cols = FEATURE_COLS + ["adjusted_close"]
    if require_eval_prices:
        needed_check_cols += ["eval_open", "eval_high", "eval_low"]

    if df[needed_check_cols].isnull().any().any():
        bad_cols = df[needed_check_cols].columns[df[needed_check_cols].isnull().any()].tolist()
        raise ValueError(f"NaNs found in required columns: {bad_cols}\nFile: {csv_path}")

    return df


class TradingEnv:
    def __init__(
        self,
        df,
        *,
        raw_feature_df=None,
        window_size=20,
        initial_cash=1_000_000.0,
        slippage_rate=0.0002,
        stop_loss=1_000,
        take_profit=1_000,
        cooldown_days=0,
        min_hold_days=0,
        profitable_close_cooldown_days=None,
        losing_close_cooldown_days=None,
        stop_exit_cooldown_days=None,
        take_profit_exit_cooldown_days=None,
    ):
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        if len(df) <= window_size + 2:
            raise ValueError("Dataframe too short for selected window_size.")

        self.window_size = int(window_size)
        self.initial_cash = float(initial_cash)
        self.slippage_rate = float(slippage_rate)
        self.stop_loss = float(stop_loss)
        self.take_profit = float(take_profit)
        self.cooldown_days = int(cooldown_days)
        self.min_hold_days = int(max(0, min_hold_days))
        self.profitable_close_cooldown_days = int(
            self.cooldown_days if profitable_close_cooldown_days is None else profitable_close_cooldown_days
        )
        self.losing_close_cooldown_days = int(
            self.cooldown_days if losing_close_cooldown_days is None else losing_close_cooldown_days
        )
        self.stop_exit_cooldown_days = int(
            self.cooldown_days if stop_exit_cooldown_days is None else stop_exit_cooldown_days
        )
        self.take_profit_exit_cooldown_days = int(
            self.cooldown_days if take_profit_exit_cooldown_days is None else take_profit_exit_cooldown_days
        )

        self.feature_cols = FEATURE_COLS
        self.raw_feature_df = None
        if raw_feature_df is not None:
            raw_feature_df = raw_feature_df.copy()
            if "Date" in raw_feature_df.columns:
                raw_feature_df["Date"] = pd.to_datetime(raw_feature_df["Date"])
                raw_feature_df = raw_feature_df.sort_values("Date").reset_index(drop=True)
            else:
                raw_feature_df = raw_feature_df.reset_index(drop=True)
            self.raw_feature_df = raw_feature_df
        self.portfolio_cols = [
            "pos",
            "in_trade",
            "entry_over_price",
            "unrealized_pnl",
            "days_in_trade",
            "cooldown_frac",
        ]
        self.obs_dim = len(self.feature_cols) + len(self.portfolio_cols)

        self.features = df[self.feature_cols].to_numpy(dtype=np.float32)
        self.raw_features = (
            self.raw_feature_df[self.feature_cols].to_numpy(dtype=np.float32)
            if self.raw_feature_df is not None
            else None
        )
        self.prices = df["adjusted_close"].to_numpy(dtype=np.float32)
        self.dates = df["Date"].to_numpy(dtype="datetime64[ns]")
        self.n_steps = int(len(df))

        self.reset(start_offset=0)

    def reset(self, start_offset: int = 0):
        start_offset = int(max(0, start_offset))
        max_offset = max(0, self.n_steps - (self.window_size + 2))
        start_offset = min(start_offset, max_offset)

        self.current_step = self.window_size + start_offset
        self.last_price_idx = None
        self.position = 0
        self.shares = 0.0
        self.entry_price = 0.0
        self.entry_equity = None
        self.position_open_step = None
        self.overwatch_force_open_once = False
        self.overwatch_force_close_once = False
        self.cash = float(self.initial_cash)
        self.equity = float(self.initial_cash)
        self.prev_equity = float(self.initial_cash)
        self.cooldown_remaining = 0
        self.trades = []
        self.close_hits = 0
        self.done = False

        self.episode_start_step = int(self.current_step)
        self.episode_start_price = float(self.prices[self.episode_start_step])
        self.bh_shares = (self.initial_cash / self.episode_start_price) if self.episode_start_price > 0 else 0.0
        self.bh_equity = float(self.initial_cash)
        self.prev_bh_equity = float(self.initial_cash)

        return self._get_obs()

    @property
    def current_datetime(self):
        idx = min(max(self.current_step, 0), self.n_steps - 1)
        return pd.Timestamp(self.dates[idx]).to_pydatetime()

    def bars_in_trade(self) -> int:
        if self.position == 0 or self.position_open_step is None:
            return 0
        return max(0, int(self.current_step - self.position_open_step))

    def can_discretionary_close(self) -> bool:
        if self.position == 0:
            return False
        if self.min_hold_days <= 0:
            return True
        return self.bars_in_trade() >= self.min_hold_days

    def _portfolio_state(self, obs_price: float) -> np.ndarray:
        pos = float(self.position)
        in_trade = 1.0 if self.position != 0 else 0.0

        if self.position != 0 and self.entry_price > 0 and obs_price > 0:
            entry_over_price = float(self.entry_price / obs_price)
            unreal_pnl = float((obs_price - self.entry_price) / (self.entry_price + 1e-12))
            days = float(self.bars_in_trade())
        else:
            entry_over_price = 0.0
            unreal_pnl = 0.0
            days = 0.0

        days_in_trade = days / 252.0
        cooldown_frac = (float(self.cooldown_remaining) / float(self.cooldown_days)) if self.cooldown_days > 0 else 0.0
        return np.array([pos, in_trade, entry_over_price, unreal_pnl, days_in_trade, cooldown_frac], dtype=np.float32)

    def _get_obs(self):
        start = self.current_step - self.window_size
        market_window = self.features[start:self.current_step]
        obs_idx = max(0, min(self.current_step - 1, self.n_steps - 1))
        obs_price = float(self.prices[obs_idx])
        p = self._portfolio_state(obs_price)
        p_mat = np.repeat(p[np.newaxis, :], self.window_size, axis=0)
        return np.concatenate([market_window, p_mat], axis=1).astype(np.float32)

    def _mark_to_market(self, price: float):
        self.equity = self.cash + self.shares * price if self.position == 1 else self.cash

    def _update_benchmark(self, price: float):
        self.bh_equity = float(self.bh_shares * price)

    def _cooldown_days_after_exit(self, pnl_pct: float, exit_reason: str) -> int:
        reason = str(exit_reason or "").strip().lower()
        if reason == "stop_loss":
            return int(self.stop_exit_cooldown_days)
        if reason == "take_profit":
            return int(self.take_profit_exit_cooldown_days)
        if float(pnl_pct) > 0.0:
            return int(self.profitable_close_cooldown_days)
        return int(self.losing_close_cooldown_days)

    def _do_close(self, price: float):
        if self.position == 0:
            return 0.0
        gross = self.shares * price
        slip_cost = gross * self.slippage_rate
        proceeds = gross - slip_cost
        pnl_dollars = proceeds - (self.shares * self.entry_price)
        self.cash += proceeds
        denom = self.shares * self.entry_price
        pct = (pnl_dollars / denom) if denom else 0.0
        ttm = self.bars_in_trade() or 1
        self.trades.append(("close_long", self.current_step, float(price), float(pct), int(ttm)))
        self.position = 0
        self.shares = 0.0
        self.entry_price = 0.0
        self.position_open_step = None
        self.entry_equity = None
        self.equity = self.cash
        return float(pct)

    def _log_excess_return_reward(self, prev_equity: float, prev_bh_equity: float):
        eps = 1e-12
        if prev_equity <= eps or self.equity <= eps:
            return -10.0, True
        if prev_bh_equity <= eps or self.bh_equity <= eps:
            return float(np.log(self.equity / prev_equity)), False
        r_agent = float(np.log(self.equity / prev_equity))
        r_bh = float(np.log(self.bh_equity / prev_bh_equity))
        return float(r_agent - r_bh), False

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode already done")

        force_open = bool(getattr(self, "overwatch_force_open_once", False))
        force_close = bool(getattr(self, "overwatch_force_close_once", False))
        self.overwatch_force_open_once = False
        self.overwatch_force_close_once = False
        pre_position = int(self.position)
        close_reason = ""
        opened_trade = False
        closed_trade = False

        prev_equity = float(self.equity)
        prev_bh_equity = float(self.bh_equity)

        if self.current_step >= self.n_steps - 1:
            price = float(self.prices[self.current_step])
            self.last_price_idx = int(self.current_step)
            if self.position != 0:
                self._do_close(price)
                closed_trade = True
                close_reason = "terminal"
            self._mark_to_market(price)
            self._update_benchmark(price)
            reward, blew_up = self._log_excess_return_reward(prev_equity, prev_bh_equity)
            self.done = True
            info = {
                "bankrupt": bool(blew_up),
                "opened_trade": False,
                "closed_trade": bool(closed_trade),
                "close_reason": close_reason,
                "force_open": bool(force_open),
                "force_close": bool(force_close),
                "pre_position": int(pre_position),
            }
            return self._get_obs(), float(reward), True, info

        price = float(self.prices[self.current_step])
        self.last_price_idx = int(self.current_step)

        if self.position != 0 and action == ACTION_CLOSE and (self.can_discretionary_close() or force_close):
            self.close_hits += 1
            close_pct = self._do_close(price)
            close_reason = "discretionary"
            closed_trade = True
            self.cooldown_remaining = self._cooldown_days_after_exit(close_pct, close_reason)

        if self.position != 0:
            pct_move = (price - self.entry_price) / (self.entry_price + 1e-12)
            if pct_move <= -self.stop_loss or pct_move >= self.take_profit:
                close_reason = "stop_loss" if pct_move <= -self.stop_loss else "take_profit"
                close_pct = self._do_close(price)
                closed_trade = True
                self.cooldown_remaining = self._cooldown_days_after_exit(close_pct, close_reason)

        if self.position == 0 and (self.cooldown_remaining == 0 or force_open):
            if action == ACTION_LONG:
                opened_trade = True
                trade_value = float(self.cash)
                if trade_value > 0 and price > 0:
                    slip_cost = trade_value * self.slippage_rate
                    shares = (trade_value - slip_cost) / price
                    self.cash -= trade_value
                    self.position = 1
                    self.shares = float(shares)
                    self.entry_price = float(price)
                    self.position_open_step = int(self.current_step)
                    self.trades.append(("open_long", self.current_step, price, None, None))
                    self._mark_to_market(price)
                    self.entry_equity = float(self.equity)
        elif self.position == 0 and self.cooldown_remaining > 0 and not force_open:
            self.cooldown_remaining -= 1

        self._mark_to_market(price)
        self._update_benchmark(price)
        reward, blew_up = self._log_excess_return_reward(prev_equity, prev_bh_equity)

        self.current_step += 1
        if self.current_step >= self.n_steps:
            self.done = True
        info = {
            "bankrupt": bool(blew_up),
            "opened_trade": bool(opened_trade),
            "closed_trade": bool(closed_trade),
            "close_reason": close_reason,
            "force_open": bool(force_open),
            "force_close": bool(force_close),
            "pre_position": int(pre_position),
        }
        if blew_up:
            self.done = True
        return self._get_obs(), float(reward), self.done, info


class InferenceAgent:
    def __init__(self, model):
        self.model = model
        self.overwatch_counter = 0
        self.override_count = 0
        self.overwatch_logs = []
        self.overwatch_enabled = True

    def _valid_actions(self, env) -> list:
        pos = int(getattr(env, "position", 0))
        cd = int(getattr(env, "cooldown_remaining", 0))
        if pos == 0:
            if cd > 0:
                return [ACTION_HOLD]
            return [ACTION_HOLD, ACTION_LONG]
        if hasattr(env, "can_discretionary_close") and not env.can_discretionary_close():
            return [ACTION_HOLD]
        return [ACTION_HOLD, ACTION_CLOSE]

    def _mask_q_values(self, q_vals: np.ndarray, valid_actions: list) -> np.ndarray:
        masked = np.array(q_vals, copy=True)
        invalid = set(range(3)) - set(valid_actions)
        for a in invalid:
            masked[a] = -1e9
        return masked

    def _apply_constraint_override(self, env, model_action: int) -> tuple[bool, str]:
        allowed, reason = resolve_constraint_override(env, int(model_action))
        if not allowed:
            return False, ""

        if reason == "cooldown":
            env.overwatch_force_open_once = True
            env.overwatch_force_close_once = False
        elif reason == "min_hold":
            env.overwatch_force_close_once = True
            env.overwatch_force_open_once = False

        return True, reason

    def act_greedy(self, state, env=None):
        valid_actions = self._valid_actions(env)
        q_vals = self.model(state[np.newaxis, :], training=False).numpy()[0]
        q_masked = self._mask_q_values(q_vals, valid_actions)
        action = int(np.argmax(q_masked))
        if action not in valid_actions:
            action = int(valid_actions[0])
        return action, q_vals, q_masked

    def act(self, state, symbol=None, env=None, use_overwatch=True):
        self.overwatch_counter += 1

        valid_actions = self._valid_actions(env)
        q_vals = self.model(state[np.newaxis, :], training=False).numpy()[0]
        q_masked = self._mask_q_values(q_vals, valid_actions)
        raw_action = int(np.argmax(q_masked))
        if raw_action not in valid_actions:
            raw_action = int(valid_actions[0])

        allow_overwatch = (
            bool(use_overwatch)
            and bool(self.overwatch_enabled)
            and symbol is not None
            and env is not None
        )
        every_n = max(1, int(CFG.OVERWATCH_EVERY_N_STEPS))
        if not allow_overwatch or (self.overwatch_counter % every_n != 0):
            return raw_action, q_vals, q_masked

        review_start = time.perf_counter()
        review_date = env.current_datetime.strftime("%Y-%m-%d")
        if bool(getattr(CFG, "OVERWATCH_VERBOSE_PROGRESS", True)):
            log_progress(
                f"{symbol}: overwatch review starting on {review_date} "
                f"(step={int(env.current_step)}, raw_action={int(raw_action)}, position={int(env.position)})"
            )

        try:
            chart_module = get_chart_module()
            fig = chart_module.build_fig(symbol, end=env.current_datetime.strftime("%Y-%m-%d"))

            fig.text(
                0.01,
                0.99,
                build_overwatch_decision_card(env, raw_action=int(raw_action), valid_actions=valid_actions),
                va="top",
                ha="left",
                color="white",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.6),
            )

            chart_path = save_overwatch_chart(
                fig,
                symbol=symbol,
                review_date=review_date,
                step=int(env.current_step),
                raw_action=int(raw_action),
                source_tag="overwatch",
            )
            img_bytes = fig_to_bytes(fig)
            close_figure(fig)

            img_b64_str = base64.b64encode(img_bytes).decode("utf-8")
            resp_txt = overwatch_decision(raw_action, img_b64_str)

            m = re.match(r"^\s*([0-2])\s*-\s*", resp_txt)
            model_action = int(m.group(1)) if m else None

            model_action_invalid = False
            constraint_override = False
            constraint_override_reason = ""
            if model_action is None:
                final_action = raw_action
            else:
                if model_action not in valid_actions:
                    constraint_override, constraint_override_reason = self._apply_constraint_override(env, model_action)
                    if constraint_override:
                        final_action = int(model_action)
                    else:
                        model_action_invalid = True
                        final_action = ACTION_HOLD if ACTION_HOLD in valid_actions else int(valid_actions[0])
                else:
                    final_action = int(model_action)

            if final_action != raw_action:
                self.override_count += 1

            self.overwatch_logs.append(
                {
                    "symbol": symbol,
                    "step": int(env.current_step),
                    "date": env.current_datetime.strftime("%Y-%m-%d"),
                    "position": int(env.position),
                    "cooldown_remaining": int(getattr(env, "cooldown_remaining", 0)),
                    "valid_actions": ",".join(map(str, valid_actions)),
                    "raw_action": int(raw_action),
                    "final_action": int(final_action),
                    "overrode": bool(final_action != raw_action),
                    "model_action": model_action,
                    "model_reply": resp_txt,
                    "model_action_invalid": bool(model_action_invalid),
                    "constraint_override": bool(constraint_override),
                    "constraint_override_reason": constraint_override_reason,
                    "chart_path": chart_path or "",
                    "error": "",
                }
            )
            if bool(getattr(CFG, "OVERWATCH_VERBOSE_PROGRESS", True)):
                log_progress(
                    f"{symbol}: overwatch review finished on {review_date} "
                    f"in {time.perf_counter() - review_start:.1f}s "
                    f"(raw={int(raw_action)}, final={int(final_action)}, overrode={bool(final_action != raw_action)})"
                )
            return final_action, q_vals, q_masked
        except Exception as e:
            try:
                close_figure(fig)
            except Exception:
                pass
            self.overwatch_logs.append(
                {
                    "symbol": symbol,
                    "step": int(getattr(env, "current_step", -1)),
                    "date": env.current_datetime.strftime("%Y-%m-%d") if env is not None else "",
                    "position": int(getattr(env, "position", 0)),
                    "cooldown_remaining": int(getattr(env, "cooldown_remaining", 0)),
                    "valid_actions": ",".join(map(str, valid_actions)),
                    "raw_action": int(raw_action),
                    "final_action": int(raw_action),
                    "overrode": False,
                    "model_action": None,
                    "model_reply": "",
                    "model_action_invalid": False,
                    "constraint_override": False,
                    "constraint_override_reason": "",
                    "chart_path": chart_path if 'chart_path' in locals() and chart_path else "",
                    "error": str(e),
                }
            )
            log_progress(
                f"{symbol}: overwatch review failed on {review_date} "
                f"after {time.perf_counter() - review_start:.1f}s with error: {e}"
            )
            return raw_action, q_vals, q_masked


def build_env(df, *, window_size: int, initial_cash: float, raw_feature_df=None) -> TradingEnv:
    return TradingEnv(
        df,
        raw_feature_df=raw_feature_df,
        window_size=int(window_size),
        initial_cash=float(initial_cash),
        slippage_rate=float(CFG.SLIPPAGE_RATE),
        stop_loss=float(CFG.STOP_LOSS_TEST),
        take_profit=float(CFG.TAKE_PROFIT_TEST),
        cooldown_days=int(CFG.COOLDOWN_DAYS),
        min_hold_days=int(CFG.MIN_HOLD_DAYS),
        profitable_close_cooldown_days=int(CFG.COOLDOWN_DAYS_AFTER_PROFITABLE_CLOSE),
        losing_close_cooldown_days=int(CFG.COOLDOWN_DAYS_AFTER_LOSING_CLOSE),
        stop_exit_cooldown_days=int(CFG.COOLDOWN_DAYS_AFTER_STOP_EXIT),
        take_profit_exit_cooldown_days=int(CFG.COOLDOWN_DAYS_AFTER_TAKE_PROFIT_EXIT),
    )


def get_report_start_idx(
    df: pd.DataFrame,
    report_start_date: str | None = None,
    report_days_back: int | None = None,
) -> int:
    min_idx = int(getattr(CFG, "WINDOW_SIZE", 0) or 0)
    if len(df) <= min_idx:
        raise ValueError(f"Dataframe too short for WINDOW_SIZE={min_idx}")

    if report_start_date:
        dt = pd.Timestamp(report_start_date)
        matches = np.where(pd.to_datetime(df["Date"]).to_numpy() >= np.datetime64(dt))[0]
        if len(matches) == 0:
            raise ValueError(f"No rows on/after REPORT_START_DATE={report_start_date}")
        return max(int(matches[0]), min_idx)

    days_back = int(report_days_back if report_days_back is not None else getattr(CFG, "REPORT_DAYS_BACK", 0) or 0)
    if days_back <= 0:
        return min_idx
    start_idx = max(0, int(len(df)) - days_back)
    return max(start_idx, min_idx)


def generate_signal_log(agent, df, *, window_size=20, initial_cash=100_000, start_idx=0, symbol="", raw_feature_df=None):
    if int(start_idx) < int(window_size):
        raise ValueError(
            f"REPORT_START_DATE lands too early in the file. Need at least WINDOW_SIZE={window_size} rows before it; got start_idx={start_idx}."
        )

    env = build_env(df, window_size=window_size, initial_cash=initial_cash, raw_feature_df=raw_feature_df)
    agent.overwatch_enabled = bool(CFG.OVERWATCH_ENABLED)
    agent.overwatch_counter = 0
    agent.override_count = 0
    agent.overwatch_logs = []

    start_offset = int(start_idx - window_size)
    state = env.reset(start_offset=start_offset)
    total_bars = int(len(df) - start_idx)
    start_date = pd.Timestamp(df["Date"].iloc[start_idx]).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(df["Date"].iloc[-1]).strftime("%Y-%m-%d")
    log_progress(
        f"{symbol}: generating signals from {start_date} to {end_date} "
        f"({total_bars} bars, overwatch={'ON' if CFG.OVERWATCH_ENABLED else 'OFF'})"
    )

    signal_rows = []
    last_q_vals = None
    last_q_masked = None
    progress_every = max(1, int(getattr(CFG, "SIGNAL_PROGRESS_EVERY_N_STEPS", 25)))

    while True:
        position_before = int(env.position)
        cooldown_before = int(getattr(env, "cooldown_remaining", 0))

        if CFG.OVERWATCH_ENABLED:
            action, q_vals, q_masked = agent.act(state, env=env, symbol=symbol, use_overwatch=True)
        else:
            action, q_vals, q_masked = agent.act_greedy(state, env=env)
        last_q_vals = q_vals
        last_q_masked = q_masked

        force_open = bool(getattr(env, "overwatch_force_open_once", False))
        force_close = bool(getattr(env, "overwatch_force_close_once", False))
        next_state, _, done, info = env.step(action)
        idx = int(env.last_price_idx) if env.last_price_idx is not None else int(max(0, env.current_step - 1))
        dt = pd.Timestamp(env.dates[idx])
        closed_trade = bool(info.get("closed_trade", False))
        close_reason = str(info.get("close_reason", "") or "")
        replay_action = ACTION_CLOSE if closed_trade else int(action)

        signal_rows.append(
            {
                "step": idx,
                "date": dt,
                "action": int(action),
                "replay_action": int(replay_action),
                "action_name": action_to_name(int(action)),
                "position_before": int(position_before),
                "position_after": int(env.position),
                "cooldown_before": int(cooldown_before),
                "cooldown_after": int(getattr(env, "cooldown_remaining", 0)),
                "opened_trade": bool(info.get("opened_trade", False)),
                "closed_trade": bool(closed_trade),
                "close_reason": close_reason,
                "force_open": bool(force_open or info.get("force_open", False)),
                "force_close": bool(force_close or info.get("force_close", False)),
                "q_hold": float(q_vals[ACTION_HOLD]),
                "q_long": float(q_vals[ACTION_LONG]),
                "q_close": float(q_vals[ACTION_CLOSE]),
                "q_hold_masked": float(q_masked[ACTION_HOLD]),
                "q_long_masked": float(q_masked[ACTION_LONG]),
                "q_close_masked": float(q_masked[ACTION_CLOSE]),
            }
        )

        processed = len(signal_rows)
        if processed == 1 or (processed % progress_every == 0) or done:
            log_progress(
                f"{symbol}: processed {processed}/{total_bars} bars through {dt.strftime('%Y-%m-%d')} "
                f"(last_action={action_to_name(int(action))}, overrides={int(agent.override_count)})"
            )

        state = next_state
        if done:
            break

    log_progress(
        f"{symbol}: signal generation complete "
        f"({len(signal_rows)} bars, overwatch_overrides={int(agent.override_count)})"
    )
    return {
        "signals": signal_rows,
        "last_q_vals": last_q_vals,
        "last_q_masked": last_q_masked,
        "override_count": int(agent.override_count),
        "override_pct": (float(agent.override_count) / len(signal_rows) * 100.0) if signal_rows else 0.0,
        "overwatch_logs": list(agent.overwatch_logs),
    }


def get_fill_price(df, idx: int, fill_mode: str) -> float:
    if fill_mode == CFG.FILL_MODE_CLOSE_T:
        return float(df["adjusted_close"].iloc[idx])
    if fill_mode == CFG.FILL_MODE_OPEN_T1:
        return float(df["eval_open"].iloc[idx])
    if fill_mode == CFG.FILL_MODE_HL2_T1:
        return float((df["eval_high"].iloc[idx] + df["eval_low"].iloc[idx]) / 2.0)
    raise ValueError(f"Unknown fill mode: {fill_mode}")


def buy_hold_curve_from_signals(df, first_step: int, last_step: int, initial_cash: float):
    px = df["adjusted_close"].to_numpy(dtype=np.float64)
    start_px = float(px[first_step])
    shares = (float(initial_cash) / start_px) if start_px > 0 else 0.0
    dates = []
    bh = []
    for idx in range(first_step, last_step + 1):
        dates.append(pd.Timestamp(df["Date"].iloc[idx]))
        bh.append(float(shares * px[idx]))
    return dates, bh


def _read_benchmark_price_series(symbol: str) -> Optional[pd.DataFrame]:
    csv_path = Path(CFG.DATA_DIR) / f"{symbol}.csv"
    if not csv_path.exists():
        print(f"[warn] aggregate benchmark file not found: {csv_path}")
        return None

    header = pd.read_csv(csv_path, nrows=0)
    price_col = next(
        (c for c in ["adjusted_close", "Adj Close", "adj_close", "Close", "close"] if c in header.columns),
        None,
    )
    if price_col is None:
        print(f"[warn] aggregate benchmark {symbol} has no usable close column: {csv_path}")
        return None

    df = pd.read_csv(csv_path, usecols=["Date", price_col], parse_dates=["Date"])
    df = df.rename(columns={price_col: symbol})
    df[symbol] = pd.to_numeric(df[symbol], errors="coerce")
    df = df.dropna(subset=["Date", symbol]).sort_values("Date").reset_index(drop=True)
    if df.empty:
        print(f"[warn] aggregate benchmark {symbol} has no usable prices: {csv_path}")
        return None
    return df


def add_aggregate_benchmarks(agg_curve: pd.DataFrame) -> pd.DataFrame:
    out = agg_curve.copy()
    if out.empty:
        return out

    benchmark_symbols = list(getattr(CFG, "AGGREGATE_BENCHMARK_SYMBOLS", []))
    if not benchmark_symbols:
        return out

    bench = pd.DataFrame({"Date": pd.to_datetime(out["Date"])})
    loaded_symbols = []
    for symbol in benchmark_symbols:
        price_df = _read_benchmark_price_series(symbol)
        if price_df is None:
            continue
        bench = bench.merge(price_df, on="Date", how="left")
        loaded_symbols.append(symbol)

    if not loaded_symbols:
        raise RuntimeError(
            "No aggregate benchmark series loaded. "
            "Set AGGREGATE_BENCHMARK_SYMBOLS to CSVs available in DATA_DIR."
        )

    loaded_benchmark_cols = []
    for symbol in loaded_symbols:
        prices = pd.to_numeric(bench[symbol], errors="coerce").ffill()
        first_valid = prices.dropna()
        if first_valid.empty or float(first_valid.iloc[0]) <= 0:
            print(f"[warn] aggregate benchmark {symbol} has no valid start price in the report window")
            continue
        norm_col = f"benchmark_{symbol}_index"
        bench[norm_col] = prices / float(first_valid.iloc[0])
        loaded_benchmark_cols.append((symbol, norm_col))

    if not loaded_benchmark_cols:
        raise RuntimeError("Aggregate benchmark series could not be normalized.")

    capital_base = float(getattr(CFG, "AGGREGATE_INITIAL_CASH", CFG.INITIAL_CASH))

    for symbol, norm_col in loaded_benchmark_cols:
        out[f"benchmark_{symbol}"] = capital_base * bench[norm_col]
    return out


def replay_signals_with_fill_mode(
    df,
    signal_rows,
    fill_mode: str,
    *,
    initial_cash: float,
    slippage_rate: float,
    cooldown_days: int,
    min_hold_days: int,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    profitable_close_cooldown_days: int | None = None,
    losing_close_cooldown_days: int | None = None,
    force_close_end: bool,
):
    if not signal_rows:
        return {
            "dates": [],
            "steps": [],
            "portfolio_values": [],
            "invested_history": [],
            "position_history": [],
            "cash_history": [],
            "shares_history": [],
            "entry_price_history": [],
            "trades": [],
            "dropped_pending_orders": 0,
            "skipped_entry_filter_count": 0,
            "final_position": 0,
            "entry_price": np.nan,
            "entry_step": None,
            "current_equity": float(initial_cash),
            "cash": float(initial_cash),
            "shares": 0.0,
            "unrealized_pct": 0.0,
            "days_in_trade": 0,
        }

    first_step = int(signal_rows[0]["step"])
    last_step = int(signal_rows[-1]["step"])
    signal_by_step = {int(r["step"]): dict(r) for r in signal_rows}

    closes = df["adjusted_close"].to_numpy(dtype=np.float64)
    dates_all = pd.to_datetime(df["Date"]).tolist()

    cash = float(initial_cash)
    shares = 0.0
    position = 0
    entry_price = 0.0
    entry_step = None
    cooldown_remaining = 0

    trades = []
    pending_order = None
    dropped_pending_orders = 0
    skipped_entry_filter_count = 0

    out_dates = []
    out_steps = []
    out_equity = []
    out_invested = []
    out_position = []
    out_cash = []
    out_shares = []
    out_entry_price = []

    def _can_open(force: bool = False):
        return position == 0 and (cooldown_remaining == 0 or bool(force))

    def _bars_held(asof_idx: int) -> int:
        if position != 1 or entry_step is None:
            return 0
        return max(0, int(asof_idx) - int(entry_step))

    def _can_close(asof_idx: int, force: bool = False):
        if position != 1:
            return False
        if bool(force):
            return True
        if int(min_hold_days) <= 0:
            return True
        return _bars_held(asof_idx) >= int(min_hold_days)

    def _open_long(fill_idx: int, fill_px: float):
        nonlocal cash, shares, position, entry_price, entry_step
        trade_value = float(cash)
        if trade_value <= 0 or fill_px <= 0 or position != 0:
            return
        slip_cost = trade_value * float(slippage_rate)
        shares_local = (trade_value - slip_cost) / fill_px
        cash -= trade_value
        shares = float(shares_local)
        position = 1
        entry_price = float(fill_px)
        entry_step = int(fill_idx)
        trades.append(("open_long", int(fill_idx), float(fill_px), None, None))

    def _close_long(fill_idx: int, fill_px: float):
        nonlocal cash, shares, position, entry_price, entry_step, cooldown_remaining
        if position != 1:
            return
        gross = shares * fill_px
        slip_cost = gross * float(slippage_rate)
        proceeds = gross - slip_cost
        pnl_dollars = proceeds - (shares * entry_price)
        cash += proceeds
        denom = shares * entry_price
        pct = (pnl_dollars / denom) if denom else 0.0
        ttm = (int(fill_idx) - int(entry_step)) if entry_step is not None else 1
        ttm = max(1, ttm)
        trades.append(("close_long", int(fill_idx), float(fill_px), float(pct), int(ttm)))
        shares = 0.0
        position = 0
        entry_price = 0.0
        entry_step = None
        win_cd = int(cooldown_days if profitable_close_cooldown_days is None else profitable_close_cooldown_days)
        loss_cd = int(cooldown_days if losing_close_cooldown_days is None else losing_close_cooldown_days)
        cooldown_remaining = win_cd if float(pct) > 0.0 else loss_cd

    def _entry_filter_passes(signal_idx: int, fill_idx: int, fill_px: float) -> bool:
        return True

    def _stop_or_take_reason(idx: int) -> str:
        if position != 1 or entry_price <= 0:
            return ""
        pct_move = (float(closes[idx]) - float(entry_price)) / (float(entry_price) + 1e-12)
        if stop_loss is not None and pct_move <= -float(stop_loss):
            return "stop_loss"
        if take_profit is not None and pct_move >= float(take_profit):
            return "take_profit"
        return ""

    for idx in range(first_step, last_step + 1):
        if pending_order is not None and int(pending_order["fill_idx"]) == idx:
            fill_px = get_fill_price(df, idx, fill_mode)
            if pending_order["type"] == "open_long":
                signal_idx = int(pending_order.get("signal_idx", idx - 1))
                if _can_open(force=bool(pending_order.get("force", False))):
                    if _entry_filter_passes(signal_idx, idx, fill_px):
                        _open_long(idx, fill_px)
                    else:
                        skipped_entry_filter_count += 1
            elif pending_order["type"] == "close_long":
                if _can_close(idx, force=bool(pending_order.get("force", False))):
                    _close_long(idx, fill_px)
            pending_order = None

        if position == 0 and cooldown_remaining > 0:
            cooldown_remaining -= 1

        close_px = float(closes[idx])
        equity = cash + shares * close_px if position == 1 else cash
        out_dates.append(pd.Timestamp(dates_all[idx]))
        out_steps.append(int(idx))
        out_equity.append(float(equity))
        out_invested.append(float(initial_cash if position == 1 else 0.0))
        out_position.append(int(position))
        out_cash.append(float(cash))
        out_shares.append(float(shares))
        out_entry_price.append(float(entry_price) if position == 1 else np.nan)

        signal = signal_by_step.get(idx, {})
        action = int(signal.get("replay_action", signal.get("action", ACTION_HOLD)))
        close_reason = str(signal.get("close_reason", "") or "")
        force_open = bool(signal.get("force_open", False))
        force_close = bool(signal.get("force_close", False)) or close_reason in {"stop_loss", "take_profit", "terminal"}

        auto_reason = _stop_or_take_reason(idx)
        if auto_reason:
            action = ACTION_CLOSE
            force_close = True

        if fill_mode == CFG.FILL_MODE_CLOSE_T:
            fill_px = get_fill_price(df, idx, fill_mode)
            if action == ACTION_LONG and _can_open(force=force_open):
                _open_long(idx, fill_px)
            elif action == ACTION_CLOSE and _can_close(idx, force=force_close):
                _close_long(idx, fill_px)

            close_px = float(closes[idx])
            equity_after = cash + shares * close_px if position == 1 else cash
            out_equity[-1] = float(equity_after)
            out_invested[-1] = float(initial_cash if position == 1 else 0.0)
            out_position[-1] = int(position)
            out_cash[-1] = float(cash)
            out_shares[-1] = float(shares)
            out_entry_price[-1] = float(entry_price) if position == 1 else np.nan
        else:
            if idx < last_step:
                if action == ACTION_LONG and _can_open(force=force_open):
                    pending_order = {
                        "type": "open_long",
                        "fill_idx": idx + 1,
                        "signal_idx": idx,
                        "force": bool(force_open),
                    }
                elif action == ACTION_CLOSE and _can_close(idx + 1, force=force_close):
                    pending_order = {
                        "type": "close_long",
                        "fill_idx": idx + 1,
                        "signal_idx": idx,
                        "force": bool(force_close),
                    }
            else:
                if action == ACTION_LONG and _can_open(force=force_open):
                    dropped_pending_orders += 1
                elif action == ACTION_CLOSE and _can_close(idx + 1, force=force_close) and not force_close_end:
                    dropped_pending_orders += 1

    if force_close_end and position == 1:
        final_idx = int(last_step)
        final_px = float(closes[final_idx])
        _close_long(final_idx, final_px)
        out_equity[-1] = float(cash)
        out_invested[-1] = 0.0
        out_position[-1] = int(position)
        out_cash[-1] = float(cash)
        out_shares[-1] = float(shares)
        out_entry_price[-1] = np.nan

    current_px = float(closes[last_step])
    current_equity = float(cash + shares * current_px if position == 1 else cash)
    unrealized_pct = float((current_px - entry_price) / (entry_price + 1e-12)) if position == 1 else 0.0
    days_in_trade = int(last_step - entry_step) if (position == 1 and entry_step is not None) else 0

    return {
        "dates": out_dates,
        "steps": out_steps,
        "portfolio_values": out_equity,
        "invested_history": out_invested,
        "position_history": out_position,
        "cash_history": out_cash,
        "shares_history": out_shares,
        "entry_price_history": out_entry_price,
        "trades": trades,
        "dropped_pending_orders": int(dropped_pending_orders),
        "skipped_entry_filter_count": int(skipped_entry_filter_count),
        "final_position": int(position),
        "entry_price": float(entry_price) if position == 1 else np.nan,
        "entry_step": int(entry_step) if entry_step is not None and position == 1 else None,
        "entry_date": pd.Timestamp(dates_all[entry_step]) if entry_step is not None and position == 1 else pd.NaT,
        "current_equity": current_equity,
        "cash": float(cash),
        "shares": float(shares),
        "unrealized_pct": unrealized_pct,
        "days_in_trade": days_in_trade,
        "current_price": current_px,
        "final_date": pd.Timestamp(dates_all[last_step]),
    }


def compute_portfolio_stats(equity_curve, invested_curve, trades, initial_value, buy_hold_curve=None):
    if not equity_curve:
        return {}

    eq = np.array(equity_curve, dtype=np.float64)
    inv = np.array(invested_curve, dtype=np.float64) if invested_curve is not None else np.zeros_like(eq)
    final_v = float(eq[-1])
    tot_ret = (final_v - initial_value) / initial_value
    ann_ret = ((final_v / initial_value) ** (252 / len(eq)) - 1) if len(eq) > 1 else np.nan
    run_max = np.maximum.accumulate(eq)
    dd = (run_max - eq) / (run_max + 1e-12)
    max_dd = float(np.max(dd))

    if len(eq) > 1:
        dr = np.diff(eq) / (eq[:-1] + 1e-12)
        sr = float(np.mean(dr) / (np.std(dr) + 1e-9) * np.sqrt(252))
    else:
        sr = np.nan

    close_trades = [t for t in trades if t[0] == "close_long"]
    trade_profits = [t[3] for t in close_trades if t[3] is not None]
    wins = sum(p > 0 for p in trade_profits)
    total = len(trade_profits)
    win_pct = wins / total * 100 if total else 0.0
    compounded_realized = (np.prod([1.0 + p for p in trade_profits]) - 1.0) if trade_profits else np.nan
    exposure_pct = float(np.mean(inv / float(initial_value)) * 100.0) if len(inv) else np.nan

    stats = {
        "final_value": final_v,
        "total_return_pct": tot_ret * 100,
        "annualized_return_pct": ann_ret * 100 if not pd.isna(ann_ret) else np.nan,
        "max_drawdown_pct": max_dd * 100,
        "sharpe_ratio": sr,
        "time_in_market_pct": exposure_pct,
        "total_closed_trades": total,
        "win_pct_long": win_pct,
        "compounded_realized_trade_return_pct": compounded_realized * 100 if not np.isnan(compounded_realized) else np.nan,
    }

    if buy_hold_curve is not None and len(buy_hold_curve) > 1:
        bh_arr = np.array(buy_hold_curve, dtype=np.float64)
        fbv = float(bh_arr[-1])
        bhr = (fbv - initial_value) / initial_value
        bh_dr = np.diff(bh_arr) / (bh_arr[:-1] + 1e-12)
        bh_sr = float(np.mean(bh_dr) / (np.std(bh_dr) + 1e-9) * np.sqrt(252))
        bh_run_max = np.maximum.accumulate(bh_arr)
        bh_dd = (bh_run_max - bh_arr) / (bh_run_max + 1e-12)
        bh_max_dd = float(np.max(bh_dd))
        stats.update(
            {
                "buy_hold_total_return_pct": bhr * 100,
                "alpha_over_buy_hold": (tot_ret - bhr) * 100,
                "buy_hold_sharpe_ratio": bh_sr,
                "buy_hold_max_drawdown_pct": bh_max_dd * 100,
            }
        )

    return stats


def compute_drawdown_pct_series(equity: pd.Series) -> pd.Series:
    s = pd.to_numeric(equity, errors="coerce")
    roll_max = s.cummax().replace(0, np.nan)
    return (s / roll_max - 1.0) * 100.0


def compute_max_drawdown_pct(equity: pd.Series) -> float:
    s = pd.to_numeric(equity, errors="coerce").dropna()
    if s.empty:
        return 0.0
    return float(compute_drawdown_pct_series(s).min())


def get_fill_mode_display_name(fill_mode: str) -> str:
    return FILL_LABELS.get(fill_mode, str(fill_mode))


def action_to_name(action: int) -> str:
    return {0: "HOLD", 1: "LONG", 2: "CLOSE"}.get(int(action), str(action))


def build_trade_log_df(symbol: str, trades, df_source: pd.DataFrame, initial_cash: float, live_pack: dict) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "symbol",
                "status",
                "entry_step",
                "entry_date",
                "entry_price",
                "exit_step",
                "exit_date",
                "exit_price",
                "holding_days",
                "trade_return_pct",
                "realized_pnl",
                "capital_at_entry",
                "capital_after_exit",
                "unrealized_pct",
                "unrealized_pnl",
                "current_price",
                "current_date",
            ]
        )

    rows = []
    capital = float(initial_cash)
    open_trade = None

    for side, step, price, pnl_pct, bars in trades:
        dt = pd.Timestamp(df_source["Date"].iloc[int(step)])
        if side == "open_long":
            open_trade = {
                "symbol": symbol,
                "status": "OPEN",
                "entry_step": int(step),
                "entry_date": dt,
                "entry_price": float(price),
                "capital_at_entry": float(capital),
            }
        elif side == "close_long" and open_trade is not None:
            realized_pct = float(pnl_pct) if pnl_pct is not None else 0.0
            realized_pnl = float(open_trade["capital_at_entry"] * realized_pct)
            capital_after_exit = float(open_trade["capital_at_entry"] * (1.0 + realized_pct))
            row = {
                **open_trade,
                "status": "CLOSED",
                "exit_step": int(step),
                "exit_date": dt,
                "exit_price": float(price),
                "holding_days": int(bars) if bars is not None else np.nan,
                "trade_return_pct": realized_pct * 100.0,
                "realized_pnl": realized_pnl,
                "capital_after_exit": capital_after_exit,
                "unrealized_pct": 0.0,
                "unrealized_pnl": 0.0,
                "current_price": float(price),
                "current_date": dt,
            }
            rows.append(row)
            capital = capital_after_exit
            open_trade = None

    if open_trade is not None:
        unrealized_pct = float(live_pack.get("unrealized_pct", 0.0))
        current_price = float(live_pack.get("current_price", np.nan))
        current_date = pd.Timestamp(live_pack.get("final_date", pd.NaT))
        unrealized_pnl = float(open_trade["capital_at_entry"] * unrealized_pct)
        rows.append(
            {
                **open_trade,
                "exit_step": np.nan,
                "exit_date": pd.NaT,
                "exit_price": np.nan,
                "holding_days": int(live_pack.get("days_in_trade", 0)),
                "trade_return_pct": np.nan,
                "realized_pnl": 0.0,
                "capital_after_exit": np.nan,
                "unrealized_pct": unrealized_pct * 100.0,
                "unrealized_pnl": unrealized_pnl,
                "current_price": current_price,
                "current_date": current_date,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["entry_date", "entry_step"]).reset_index(drop=True)
    return out


def build_symbol_timeseries_df(symbol: str, raw_df: pd.DataFrame, signal_rows, curve_df: pd.DataFrame, primary_live: dict) -> pd.DataFrame:
    first_step = int(signal_rows[0]["step"])
    last_step = int(signal_rows[-1]["step"])
    idx_slice = slice(first_step, last_step + 1)
    base_dates = pd.to_datetime(raw_df["Date"].iloc[idx_slice]).reset_index(drop=True)

    out = pd.DataFrame(
        {
            "date": base_dates,
            "symbol": symbol,
            "close": raw_df["adjusted_close"].iloc[idx_slice].to_numpy(dtype=np.float64),
            "eval_open": raw_df["eval_open"].iloc[idx_slice].to_numpy(dtype=np.float64),
            "eval_high": raw_df["eval_high"].iloc[idx_slice].to_numpy(dtype=np.float64),
            "eval_low": raw_df["eval_low"].iloc[idx_slice].to_numpy(dtype=np.float64),
            "equity": np.array(primary_live["portfolio_values"], dtype=np.float64),
            "benchmark_equity": np.array(curve_df["buy_hold"], dtype=np.float64),
            "position": np.array(primary_live["position_history"], dtype=np.int64),
            "cash": np.array(primary_live["cash_history"], dtype=np.float64),
            "shares": np.array(primary_live["shares_history"], dtype=np.float64),
            "entry_price": np.array(primary_live["entry_price_history"], dtype=np.float64),
            "primary_fill_mode": CFG.PRIMARY_REPORT_FILL_MODE,
        }
    )

    out["market_value"] = np.where(out["position"] == 1, out["shares"] * out["close"], 0.0)
    out["unrealized_pnl"] = np.where(
        out["position"] == 1,
        (out["close"] - out["entry_price"]).fillna(0.0) * out["shares"],
        0.0,
    )
    out["return_pct"] = (out["equity"] / float(CFG.INITIAL_CASH) - 1.0) * 100.0
    out["benchmark_return_pct"] = (out["benchmark_equity"] / float(CFG.INITIAL_CASH) - 1.0) * 100.0
    out["alpha_pct"] = out["return_pct"] - out["benchmark_return_pct"]
    out["drawdown_pct"] = compute_drawdown_pct_series(out["equity"])

    action_map = {int(r["step"]): r for r in signal_rows}
    actions = []
    q_hold = []
    q_long = []
    q_close = []
    q_hold_masked = []
    q_long_masked = []
    q_close_masked = []
    action_steps = []

    for step in range(first_step, last_step + 1):
        row = action_map.get(step)
        if row is None:
            actions.append("HOLD")
            q_hold.append(np.nan)
            q_long.append(np.nan)
            q_close.append(np.nan)
            q_hold_masked.append(np.nan)
            q_long_masked.append(np.nan)
            q_close_masked.append(np.nan)
            action_steps.append(step)
        else:
            actions.append(row["action_name"])
            q_hold.append(row["q_hold"])
            q_long.append(row["q_long"])
            q_close.append(row["q_close"])
            q_hold_masked.append(row["q_hold_masked"])
            q_long_masked.append(row["q_long_masked"])
            q_close_masked.append(row["q_close_masked"])
            action_steps.append(int(row["step"]))

    out["signal_action"] = actions
    out["signal_step"] = action_steps
    out["q_hold"] = q_hold
    out["q_long"] = q_long
    out["q_close"] = q_close
    out["q_hold_masked"] = q_hold_masked
    out["q_long_masked"] = q_long_masked
    out["q_close_masked"] = q_close_masked

    for mode in CFG.ALL_FILL_MODES:
        out[f"equity_{mode}"] = curve_df[mode].to_numpy(dtype=np.float64)
    out["equity_primary_live"] = np.array(primary_live["portfolio_values"], dtype=np.float64)

    return out


def build_symbol_summary_row(
    symbol: str,
    raw_df: pd.DataFrame,
    signals: list[dict],
    primary_live: dict,
    primary_backtest_stats: dict,
    primary_live_stats: dict,
    trade_log_df: pd.DataFrame,
    agent_state: dict,
    signal_pack: dict,
    report_start_date: pd.Timestamp,
) -> dict:
    last_signal = signals[-1]
    last_step = int(signals[-1]["step"])
    latest_price = float(raw_df["adjusted_close"].iloc[last_step])
    entry_step = primary_live["entry_step"]
    entry_date = pd.Timestamp(raw_df["Date"].iloc[entry_step]) if entry_step is not None else pd.NaT

    closed_trades = trade_log_df.loc[trade_log_df["status"] == "CLOSED"].copy() if not trade_log_df.empty else pd.DataFrame()
    realized_pnl = float(pd.to_numeric(closed_trades.get("realized_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    unrealized_pnl = float(primary_live.get("shares", 0.0) * (primary_live.get("current_price", 0.0) - _safe_float(primary_live.get("entry_price", np.nan), 0.0))) if int(primary_live["final_position"]) == 1 else 0.0

    return {
        "symbol": symbol,
        "last_bar_date": pd.Timestamp(raw_df["Date"].iloc[last_step]),
        "report_start_date": pd.Timestamp(report_start_date),
        "primary_fill_mode": CFG.PRIMARY_REPORT_FILL_MODE,
        "stance": "LONG" if int(primary_live["final_position"]) == 1 else "FLAT",
        "position": int(primary_live["final_position"]),
        "last_signal_action": action_to_name(int(last_signal["action"])),
        "last_signal_date": pd.Timestamp(last_signal["date"]),
        "current_price": latest_price,
        "entry_date": entry_date,
        "entry_price": float(primary_live["entry_price"]) if pd.notna(primary_live["entry_price"]) else np.nan,
        "days_in_trade": int(primary_live["days_in_trade"]),
        "unrealized_pct": float(primary_live["unrealized_pct"] * 100.0),
        "unrealized_pnl": unrealized_pnl,
        "realized_pnl": realized_pnl,
        "current_equity_live": float(primary_live["current_equity"]),
        "cash_live": float(primary_live["cash"]),
        "shares_live": float(primary_live["shares"]),
        "total_return_pct_live": float(primary_live_stats.get("total_return_pct", np.nan)),
        "annualized_return_pct_live": float(primary_live_stats.get("annualized_return_pct", np.nan)),
        "alpha_over_buy_hold_pct_live": float(primary_live_stats.get("alpha_over_buy_hold", np.nan)),
        "max_drawdown_pct_live": float(primary_live_stats.get("max_drawdown_pct", np.nan)),
        "time_in_market_pct_live": float(primary_live_stats.get("time_in_market_pct", np.nan)),
        "total_closed_trades_live": float(primary_live_stats.get("total_closed_trades", np.nan)),
        "win_pct_long_live": float(primary_live_stats.get("win_pct_long", np.nan)),
        "total_return_pct_backtest_style": float(primary_backtest_stats.get("total_return_pct", np.nan)),
        "annualized_return_pct_backtest_style": float(primary_backtest_stats.get("annualized_return_pct", np.nan)),
        "alpha_over_buy_hold_pct_backtest_style": float(primary_backtest_stats.get("alpha_over_buy_hold", np.nan)),
        "max_drawdown_pct_backtest_style": float(primary_backtest_stats.get("max_drawdown_pct", np.nan)),
        "time_in_market_pct_backtest_style": float(primary_backtest_stats.get("time_in_market_pct", np.nan)),
        "total_closed_trades_backtest_style": float(primary_backtest_stats.get("total_closed_trades", np.nan)),
        "win_pct_long_backtest_style": float(primary_backtest_stats.get("win_pct_long", np.nan)),
        "last_q_hold": float(last_signal["q_hold"]),
        "last_q_long": float(last_signal["q_long"]),
        "last_q_close": float(last_signal["q_close"]),
        "last_q_hold_masked": float(last_signal["q_hold_masked"]),
        "last_q_long_masked": float(last_signal["q_long_masked"]),
        "last_q_close_masked": float(last_signal["q_close_masked"]),
        "overwatch_enabled": bool(CFG.OVERWATCH_ENABLED),
        "overwatch_override_count": int(signal_pack.get("override_count", 0)),
        "overwatch_override_pct": float(signal_pack.get("override_pct", 0.0)),
        "overwatch_review_count": int(len(signal_pack.get("overwatch_logs", []))),
        "saved_model_last_trained_date": agent_state.get("last_trained_date"),
        "saved_model_timestamp_utc": agent_state.get("saved_at_utc"),
    }


def compute_symbol_metrics(symbol_timeseries: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    if symbol_timeseries.empty:
        return pd.DataFrame()

    rows = []
    summary_map = summary_df.set_index("symbol").to_dict(orient="index") if not summary_df.empty else {}

    for symbol, g in symbol_timeseries.groupby("symbol", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        first_eq = _safe_float(g["equity"].iloc[0], CFG.INITIAL_CASH)
        last_eq = _safe_float(g["equity"].iloc[-1], CFG.INITIAL_CASH)
        first_bh = _safe_float(g["benchmark_equity"].iloc[0], CFG.INITIAL_CASH)
        last_bh = _safe_float(g["benchmark_equity"].iloc[-1], CFG.INITIAL_CASH)
        total_return_pct = ((last_eq / first_eq) - 1.0) * 100.0 if first_eq else 0.0
        benchmark_return_pct = ((last_bh / first_bh) - 1.0) * 100.0 if first_bh else 0.0
        alpha_pct = total_return_pct - benchmark_return_pct
        snap = summary_map.get(symbol, {})

        rows.append(
            {
                "symbol": symbol,
                "as_of": pd.Timestamp(g["date"].iloc[-1]),
                "stance": snap.get("stance", "FLAT"),
                "position": int(_safe_float(g["position"].iloc[-1], 0)),
                "current_price": _safe_float(g["close"].iloc[-1], np.nan),
                "entry_price": _safe_float(g["entry_price"].iloc[-1], np.nan),
                "days_in_trade": int(snap.get("days_in_trade", 0)),
                "unrealized_pct": _safe_float(snap.get("unrealized_pct", 0.0)),
                "unrealized_pnl": _safe_float(snap.get("unrealized_pnl", 0.0)),
                "realized_pnl": _safe_float(snap.get("realized_pnl", 0.0)),
                "current_equity": last_eq,
                "benchmark_equity": last_bh,
                "total_return_pct": total_return_pct,
                "benchmark_return_pct": benchmark_return_pct,
                "alpha_pct": alpha_pct,
                "max_drawdown_pct": compute_max_drawdown_pct(g["equity"]),
                "time_in_market_pct": float((pd.to_numeric(g["position"], errors="coerce").fillna(0) == 1).mean() * 100.0),
                "last_signal_action": snap.get("last_signal_action", "HOLD"),
                "saved_model_last_trained_date": snap.get("saved_model_last_trained_date"),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["position", "alpha_pct", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
    return out


def aggregate_curve_tables(symbol_curve_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    aggregate_cols = ["Date", "buy_hold", "primary_live", "primary_position"] + list(CFG.ALL_FILL_MODES)
    for symbol, df_curve in symbol_curve_tables.items():
        keep_cols = [c for c in aggregate_cols if c in df_curve.columns]
        temp = df_curve.loc[:, keep_cols].copy()
        rename_map = {
            "buy_hold": f"buy_hold__{symbol}",
            "primary_live": f"primary_live__{symbol}",
            "primary_position": f"primary_position__{symbol}",
        }
        rename_map.update({mode: f"{mode}__{symbol}" for mode in CFG.ALL_FILL_MODES})
        temp = temp.rename(columns=rename_map)

        if merged is None:
            merged = temp
        else:
            merged = merged.merge(temp, on="Date", how="outer")

    merged = merged.sort_values("Date").reset_index(drop=True)

    primary_live_cols = [c for c in merged.columns if c.startswith("primary_live__")]
    pos_cols = [c for c in merged.columns if c.startswith("primary_position__")]

    complete_mask = merged[primary_live_cols].notna().all(axis=1)
    dropped_rows = int((~complete_mask).sum())
    if dropped_rows:
        dropped_dates = pd.to_datetime(merged.loc[~complete_mask, "Date"], errors="coerce")
        dropped_preview = ", ".join(dropped_dates.dt.strftime("%Y-%m-%d").dropna().head(10).tolist())
        more = "..." if dropped_rows > 10 else ""
        print(
            f"[warn] aggregate curve dropped {dropped_rows} incomplete date rows "
            f"where one or more symbol sleeves were missing: {dropped_preview}{more}"
        )
    merged = merged.loc[complete_mask].reset_index(drop=True)

    out = pd.DataFrame({"Date": merged["Date"]})

    out["primary_live"] = merged[primary_live_cols].sum(axis=1, skipna=True)
    out["open_positions"] = merged[pos_cols].sum(axis=1, skipna=True)
    out["symbols_reporting"] = merged[primary_live_cols].notna().sum(axis=1)

    for mode in CFG.ALL_FILL_MODES:
        cols = [c for c in merged.columns if c.startswith(f"{mode}__")]
        out[mode] = merged[cols].sum(axis=1, skipna=True)

    reporting = pd.to_numeric(out["symbols_reporting"], errors="coerce").replace(0, np.nan)
    aggregate_capital_base = float(getattr(CFG, "AGGREGATE_INITIAL_CASH", CFG.INITIAL_CASH))
    source_capital_base = float(CFG.INITIAL_CASH) * reporting
    scale = aggregate_capital_base / source_capital_base

    equity_cols = ["primary_live"] + list(CFG.ALL_FILL_MODES)
    for col in equity_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * scale

    out["aggregate_initial_cash"] = aggregate_capital_base
    out["model_symbol_allocation"] = aggregate_capital_base / reporting
    out["portfolio_return_pct"] = (out["primary_live"] / aggregate_capital_base - 1.0) * 100.0
    out["drawdown_pct"] = compute_drawdown_pct_series(out["primary_live"])
    out = add_aggregate_benchmarks(out)

    return out


def compute_trade_stats(trade_log: pd.DataFrame) -> dict[str, float]:
    if trade_log.empty:
        return {
            "realized_pnl": 0.0,
            "win_rate_pct": 0.0,
            "avg_trade_return_pct": 0.0,
            "total_trades": 0,
        }

    closed = trade_log.loc[trade_log["status"] == "CLOSED"].copy()
    if closed.empty:
        return {
            "realized_pnl": 0.0,
            "win_rate_pct": 0.0,
            "avg_trade_return_pct": 0.0,
            "total_trades": 0,
        }

    realized = pd.to_numeric(closed["realized_pnl"], errors="coerce").fillna(0.0)
    trade_ret = pd.to_numeric(closed["trade_return_pct"], errors="coerce").fillna(0.0)
    wins = int((realized > 0).sum())
    total = int(len(closed))
    return {
        "realized_pnl": float(realized.sum()),
        "win_rate_pct": float((wins / total) * 100.0) if total else 0.0,
        "avg_trade_return_pct": float(trade_ret.mean()) if total else 0.0,
        "total_trades": total,
    }


def build_dashboard_summary(
    summary_df: pd.DataFrame,
    symbol_metrics: pd.DataFrame,
    aggregate_timeseries: pd.DataFrame,
    trade_log: pd.DataFrame,
) -> DashboardSummary:
    trade_stats = compute_trade_stats(trade_log)

    if aggregate_timeseries.empty:
        as_of = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        total_return_pct = 0.0
        benchmark_return_pct = 0.0
        alpha_pct = 0.0
        max_drawdown_pct = 0.0
    else:
        latest = aggregate_timeseries.iloc[-1]
        as_of = pd.Timestamp(latest["Date"]).strftime("%Y-%m-%d")
        total_return_pct = _safe_float(latest.get("portfolio_return_pct", 0.0))
        benchmark_return_pct = _safe_float(latest.get("benchmark_return_pct", 0.0))
        alpha_pct = _safe_float(latest.get("alpha_pct", 0.0))
        max_drawdown_pct = compute_max_drawdown_pct(aggregate_timeseries["primary_live"])

    open_positions = int((pd.to_numeric(symbol_metrics.get("position", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int) == 1).sum())
    total_symbols = int(len(symbol_metrics))
    closed_positions = max(total_symbols - open_positions, 0)
    exposure_pct = (open_positions / total_symbols * 100.0) if total_symbols else 0.0
    unrealized_pnl = float(pd.to_numeric(symbol_metrics.get("unrealized_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    realized_pnl = float(trade_stats["realized_pnl"])
    if "report_start_date" in summary_df.columns and not summary_df.empty:
        starts = pd.to_datetime(summary_df["report_start_date"], errors="coerce").dropna()
        report_start_date = starts.min().strftime("%Y-%m-%d") if not starts.empty else ""
    else:
        report_start_date = str(CFG.REPORT_START_DATE or f"last_{int(CFG.REPORT_DAYS_BACK)}_rows")

    return DashboardSummary(
        as_of=as_of,
        report_start_date=report_start_date,
        primary_fill_mode=CFG.PRIMARY_REPORT_FILL_MODE,
        total_symbols=total_symbols,
        open_positions=open_positions,
        closed_positions=closed_positions,
        total_return_pct=total_return_pct,
        benchmark_return_pct=benchmark_return_pct,
        alpha_pct=alpha_pct,
        unrealized_pnl=unrealized_pnl,
        realized_pnl=realized_pnl,
        total_pnl=unrealized_pnl + realized_pnl,
        max_drawdown_pct=max_drawdown_pct,
        win_rate_pct=float(trade_stats["win_rate_pct"]),
        avg_trade_return_pct=float(trade_stats["avg_trade_return_pct"]),
        exposure_pct=exposure_pct,
        total_trades=int(trade_stats["total_trades"]),
    )


def save_dashboard_outputs(
    summary_df: pd.DataFrame,
    symbol_metrics_df: pd.DataFrame,
    symbol_timeseries_df: pd.DataFrame,
    trade_log_df: pd.DataFrame,
    aggregate_timeseries_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    _ensure_dir(output_dir)

    dashboard_summary = build_dashboard_summary(summary_df, symbol_metrics_df, aggregate_timeseries_df, trade_log_df)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(dashboard_summary), f, indent=2, default=str)

    if CFG.WRITE_DASHBOARD_CSV_FALLBACK:
        summary_df.to_csv(output_dir / "morning_report.csv", index=False)
        symbol_metrics_df.to_csv(output_dir / "symbol_metrics.csv", index=False)
        symbol_timeseries_df.to_csv(output_dir / "symbol_timeseries.csv", index=False)
        trade_log_df.to_csv(output_dir / "trade_log.csv", index=False)
        aggregate_timeseries_df.to_csv(output_dir / "aggregate_timeseries.csv", index=False)

    _try_write_parquet(summary_df, output_dir / "morning_report.parquet")
    _try_write_parquet(symbol_metrics_df, output_dir / "symbol_metrics.parquet")
    _try_write_parquet(symbol_timeseries_df, output_dir / "symbol_timeseries.parquet")
    _try_write_parquet(trade_log_df, output_dir / "trade_log.parquet")
    _try_write_parquet(aggregate_timeseries_df, output_dir / "aggregate_timeseries.parquet")

    print(f"[dashboard] wrote dashboard-ready data to: {output_dir}")


def make_symbol_figure(symbol: str, curve_df: pd.DataFrame, raw_df: pd.DataFrame, trade_log_df: pd.DataFrame, summary_row: dict) -> go.Figure:
    price_df = pd.DataFrame({
        "Date": pd.to_datetime(curve_df["Date"]),
        "Close": raw_df.loc[raw_df["Date"].isin(pd.to_datetime(curve_df["Date"])), "adjusted_close"].to_numpy(dtype=np.float64),
    })

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.34, 0.46, 0.20],
        subplot_titles=(
            f"{symbol} Price / Trade Markers",
            f"{symbol} Equity Curves ({get_fill_mode_display_name(CFG.PRIMARY_REPORT_FILL_MODE)} live highlighted)",
            "Q-Values",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=price_df["Date"],
            y=price_df["Close"],
            mode="lines",
            name="Adjusted Close",
            hovertemplate="%{x|%Y-%m-%d}<br>Close=%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if not trade_log_df.empty:
        opens = trade_log_df[["entry_date", "entry_price"]].dropna().rename(columns={"entry_date": "date", "entry_price": "price"})
        if not opens.empty:
            fig.add_trace(
                go.Scatter(
                    x=opens["date"],
                    y=opens["price"],
                    mode="markers",
                    name="Entries",
                    marker=dict(symbol="triangle-up", size=10),
                    hovertemplate="%{x|%Y-%m-%d}<br>Entry=%{y:,.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        closes = trade_log_df[["exit_date", "exit_price"]].dropna().rename(columns={"exit_date": "date", "exit_price": "price"})
        if not closes.empty:
            fig.add_trace(
                go.Scatter(
                    x=closes["date"],
                    y=closes["price"],
                    mode="markers",
                    name="Exits",
                    marker=dict(symbol="triangle-down", size=10),
                    hovertemplate="%{x|%Y-%m-%d}<br>Exit=%{y:,.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=curve_df["Date"],
            y=curve_df["buy_hold"],
            mode="lines",
            name="Buy & Hold",
            hovertemplate="%{x|%Y-%m-%d}<br>BuyHold=%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    for mode in CFG.ALL_FILL_MODES:
        fig.add_trace(
            go.Scatter(
                x=curve_df["Date"],
                y=curve_df[mode],
                mode="lines",
                name=get_fill_mode_display_name(mode),
                visible=True if mode == CFG.PRIMARY_REPORT_FILL_MODE else "legendonly",
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{get_fill_mode_display_name(mode)}=%{{y:,.0f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=curve_df["Date"],
            y=curve_df["primary_live"],
            mode="lines",
            name=f"{get_fill_mode_display_name(CFG.PRIMARY_REPORT_FILL_MODE)} Live",
            line=dict(width=4),
            hovertemplate="%{x|%Y-%m-%d}<br>Primary Live=%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    q_df = curve_df[["Date", "q_hold", "q_long", "q_close"]].copy()
    fig.add_trace(go.Scatter(x=q_df["Date"], y=q_df["q_hold"], mode="lines", name="Q Hold", visible="legendonly"), row=3, col=1)
    fig.add_trace(go.Scatter(x=q_df["Date"], y=q_df["q_long"], mode="lines", name="Q Long"), row=3, col=1)
    fig.add_trace(go.Scatter(x=q_df["Date"], y=q_df["q_close"], mode="lines", name="Q Close", visible="legendonly"), row=3, col=1)

    fig.update_layout(
        title=(
            f"{symbol} | Stance={summary_row['stance']} | "
            f"Unrlzd={summary_row['unrealized_pct']:.2f}% | "
            f"Ret={summary_row['total_return_pct_live']:.2f}% | "
            f"Alpha={summary_row['alpha_over_buy_hold_pct_live']:.2f}%"
        ),
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=1000,
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def make_aggregate_figure(agg_curve: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.72, 0.28])
    default_benchmark = str(CFG.AGGREGATE_BENCHMARK_SYMBOLS[0]) if CFG.AGGREGATE_BENCHMARK_SYMBOLS else ""
    for col in [c for c in agg_curve.columns if c.startswith("benchmark_")]:
        symbol = col.removeprefix("benchmark_")
        fig.add_trace(
            go.Scatter(
                x=agg_curve["Date"],
                y=agg_curve[col],
                mode="lines",
                name=f"Benchmark {symbol}",
                visible=True if symbol == default_benchmark else "legendonly",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(go.Scatter(x=agg_curve["Date"], y=agg_curve["primary_live"], mode="lines", name=f"Aggregate {get_fill_mode_display_name(CFG.PRIMARY_REPORT_FILL_MODE)} Live", line=dict(width=4)), row=1, col=1)
    for mode in CFG.ALL_FILL_MODES:
        fig.add_trace(
            go.Scatter(
                x=agg_curve["Date"],
                y=agg_curve[mode],
                mode="lines",
                name=f"Aggregate {get_fill_mode_display_name(mode)}",
                visible=True if mode == CFG.PRIMARY_REPORT_FILL_MODE else "legendonly",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(go.Bar(x=agg_curve["Date"], y=agg_curve["open_positions"], name="Open Positions"), row=2, col=1)
    fig.update_layout(
        title=f"Aggregate Live Report | Primary Fill={get_fill_mode_display_name(CFG.PRIMARY_REPORT_FILL_MODE)}",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=850,
    )
    return fig


def save_plotly_figure(fig: go.Figure, output_path: Path) -> None:
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)
    if CFG.OPEN_PLOTLY_HTML:
        try:
            webbrowser.open(output_path.resolve().as_uri())
        except Exception:
            pass


def discover_symbols(data_dir: str):
    symbols = sorted([p.stem.upper() for p in Path(data_dir).glob("*.csv")])
    benchmark_symbols = set(getattr(CFG, "AGGREGATE_BENCHMARK_SYMBOLS", []))
    data_only_symbols = set(getattr(CFG, "DATA_ONLY_SYMBOLS", []))
    runnable = []
    skipped_no_checkpoint = []
    for symbol in symbols:
        ckpt = get_checkpoint_paths(CFG.MODEL_DIR, symbol)
        if os.path.exists(ckpt["policy"]) and os.path.exists(ckpt["scaler"]):
            runnable.append(symbol)
        elif symbol in benchmark_symbols or symbol in data_only_symbols:
            continue
        else:
            skipped_no_checkpoint.append(symbol)

    if skipped_no_checkpoint:
        print(
            "[warn] skipping symbols without policy/scaler checkpoints: "
            + ", ".join(skipped_no_checkpoint)
        )
    return runnable


def compute_common_report_start_date(symbols: list[str]) -> Optional[pd.Timestamp]:
    if CFG.REPORT_START_DATE:
        return pd.Timestamp(CFG.REPORT_START_DATE)
    if not bool(getattr(CFG, "REPORT_USE_COMMON_START_DATE", True)):
        return None

    starts = []
    for symbol in symbols:
        csv_path = Path(CFG.DATA_DIR) / f"{symbol}.csv"
        if not csv_path.exists():
            continue
        date_df = pd.read_csv(csv_path, usecols=["Date"], parse_dates=["Date"])
        date_df = date_df.sort_values("Date").reset_index(drop=True)
        start_idx = get_report_start_idx(
            date_df,
            report_start_date=None,
            report_days_back=CFG.REPORT_DAYS_BACK,
        )
        starts.append(pd.Timestamp(date_df["Date"].iloc[start_idx]))

    if not starts:
        return None
    return max(starts)


def validate_scaler_feature_order(scaler: ZScoreScaler):
    if list(scaler.feature_cols) != FEATURE_COLS:
        raise ValueError(
            "Saved scaler feature_cols do not match FEATURE_COLS in this live script. "
            "Keep them identical or update this script to the same feature order used in training."
        )


def run_for_symbol(symbol: str):
    symbol_start = time.perf_counter()
    csv_path = os.path.join(CFG.DATA_DIR, f"{symbol}.csv")
    ckpt = get_checkpoint_paths(CFG.MODEL_DIR, symbol)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing data file for {symbol}: {csv_path}")
    if not os.path.exists(ckpt["policy"]):
        raise FileNotFoundError(f"Missing saved policy model for {symbol}: {ckpt['policy']}")
    if not os.path.exists(ckpt["scaler"]):
        raise FileNotFoundError(f"Missing saved scaler for {symbol}: {ckpt['scaler']}")

    raw_df = load_csv(csv_path, require_eval_prices=CFG.REQUIRE_EVAL_PRICES)
    log_progress(f"{symbol}: loaded data with {len(raw_df)} rows from {csv_path}")
    scaler = load_scaler(ckpt["scaler"])
    validate_scaler_feature_order(scaler)
    df = scaler.transform(raw_df)

    report_start_idx = get_report_start_idx(
        df,
        report_start_date=CFG.REPORT_START_DATE,
        report_days_back=CFG.REPORT_DAYS_BACK,
    )
    report_start_date = pd.Timestamp(df["Date"].iloc[report_start_idx])
    log_progress(
        f"{symbol}: using report start {report_start_date.strftime('%Y-%m-%d')} "
        f"(index={report_start_idx}, days_back={int(CFG.REPORT_DAYS_BACK) if CFG.REPORT_START_DATE is None else 'date override'})"
    )
    model = load_keras_model(ckpt["policy"])
    log_progress(f"{symbol}: loaded policy model from {ckpt['policy']}")
    agent = InferenceAgent(model)
    agent_state = load_agent_state_json(ckpt["state"])

    signal_pack = generate_signal_log(
        agent,
        df,
        window_size=CFG.WINDOW_SIZE,
        initial_cash=CFG.INITIAL_CASH,
        start_idx=report_start_idx,
        symbol=symbol,
        raw_feature_df=raw_df,
    )
    signals = signal_pack["signals"]
    if not signals:
        raise RuntimeError(f"No signals generated for {symbol}")

    first_step = int(signals[0]["step"])
    last_step = int(signals[-1]["step"])
    bh_dates, bh_curve = buy_hold_curve_from_signals(df, first_step, last_step, CFG.INITIAL_CASH)

    fills_backtest = {}
    fills_live = {}
    stats_rows = []
    live_stats_rows = []
    for mode in CFG.ALL_FILL_MODES:
        fills_backtest[mode] = replay_signals_with_fill_mode(
            df=df,
            signal_rows=signals,
            fill_mode=mode,
            initial_cash=CFG.INITIAL_CASH,
            slippage_rate=float(CFG.SLIPPAGE_RATE),
            cooldown_days=int(CFG.COOLDOWN_DAYS),
            min_hold_days=int(CFG.MIN_HOLD_DAYS),
            stop_loss=float(CFG.STOP_LOSS_TEST),
            take_profit=float(CFG.TAKE_PROFIT_TEST),
            profitable_close_cooldown_days=int(CFG.COOLDOWN_DAYS_AFTER_PROFITABLE_CLOSE),
            losing_close_cooldown_days=int(CFG.COOLDOWN_DAYS_AFTER_LOSING_CLOSE),
            force_close_end=True,
        )
        fills_live[mode] = replay_signals_with_fill_mode(
            df=df,
            signal_rows=signals,
            fill_mode=mode,
            initial_cash=CFG.INITIAL_CASH,
            slippage_rate=float(CFG.SLIPPAGE_RATE),
            cooldown_days=int(CFG.COOLDOWN_DAYS),
            min_hold_days=int(CFG.MIN_HOLD_DAYS),
            stop_loss=float(CFG.STOP_LOSS_TEST),
            take_profit=float(CFG.TAKE_PROFIT_TEST),
            profitable_close_cooldown_days=int(CFG.COOLDOWN_DAYS_AFTER_PROFITABLE_CLOSE),
            losing_close_cooldown_days=int(CFG.COOLDOWN_DAYS_AFTER_LOSING_CLOSE),
            force_close_end=False,
        )

        stats = compute_portfolio_stats(
            equity_curve=fills_backtest[mode]["portfolio_values"],
            invested_curve=fills_backtest[mode]["invested_history"],
            trades=fills_backtest[mode]["trades"],
            initial_value=CFG.INITIAL_CASH,
            buy_hold_curve=bh_curve,
        )
        stats["fill_mode"] = mode
        stats_rows.append(stats)

        live_stats = compute_portfolio_stats(
            equity_curve=fills_live[mode]["portfolio_values"],
            invested_curve=fills_live[mode]["invested_history"],
            trades=fills_live[mode]["trades"],
            initial_value=CFG.INITIAL_CASH,
            buy_hold_curve=bh_curve,
        )
        live_stats["fill_mode"] = mode
        live_stats_rows.append(live_stats)

    primary_live = fills_live[CFG.PRIMARY_REPORT_FILL_MODE]
    primary_backtest = fills_backtest[CFG.PRIMARY_REPORT_FILL_MODE]
    stats_df = pd.DataFrame(stats_rows)
    stats_live_df = pd.DataFrame(live_stats_rows)
    primary_stats = stats_df.loc[stats_df["fill_mode"] == CFG.PRIMARY_REPORT_FILL_MODE].iloc[0].to_dict()
    primary_live_stats = stats_live_df.loc[stats_live_df["fill_mode"] == CFG.PRIMARY_REPORT_FILL_MODE].iloc[0].to_dict()

    curve_df = pd.DataFrame({"Date": bh_dates, "buy_hold": bh_curve})
    for mode in CFG.ALL_FILL_MODES:
        curve_df[mode] = fills_backtest[mode]["portfolio_values"]
    curve_df["primary_live"] = primary_live["portfolio_values"]
    curve_df["primary_position"] = primary_live["position_history"]
    curve_df["q_hold"] = [row["q_hold"] for row in signals]
    curve_df["q_long"] = [row["q_long"] for row in signals]
    curve_df["q_close"] = [row["q_close"] for row in signals]
    curve_df["action"] = [row["action_name"] for row in signals]

    trade_log_df = build_trade_log_df(symbol, primary_live["trades"], raw_df, CFG.INITIAL_CASH, primary_live)
    symbol_timeseries_df = build_symbol_timeseries_df(symbol, raw_df, signals, curve_df, primary_live)
    summary_row = build_symbol_summary_row(
        symbol=symbol,
        raw_df=raw_df,
        signals=signals,
        primary_live=primary_live,
        primary_backtest_stats=primary_stats,
        primary_live_stats=primary_live_stats,
        trade_log_df=trade_log_df,
        agent_state=agent_state,
        signal_pack=signal_pack,
        report_start_date=report_start_date,
    )

    log_progress(
        f"{symbol}: finished symbol in {time.perf_counter() - symbol_start:.1f}s "
        f"(stance={summary_row['stance']}, overrides={int(signal_pack.get('override_count', 0))})"
    )

    return {
        "summary_row": summary_row,
        "signals_df": pd.DataFrame(signals),
        "overwatch_logs_df": build_overwatch_logs_df(signal_pack.get("overwatch_logs", [])),
        "curve_df": curve_df,
        "stats_df": stats_df,
        "stats_live_df": stats_live_df,
        "trade_log_df": trade_log_df,
        "symbol_timeseries_df": symbol_timeseries_df,
        "primary_live": primary_live,
        "primary_backtest": primary_backtest,
        "raw_df": raw_df,
    }


def run_for_symbol_safe(symbol: str) -> dict:
    started_at = time.perf_counter()
    try:
        return {
            "ok": True,
            "symbol": symbol,
            "result": run_for_symbol(symbol),
            "error": "",
            "elapsed_sec": time.perf_counter() - started_at,
        }
    except Exception as exc:
        return {
            "ok": False,
            "symbol": symbol,
            "result": None,
            "error": str(exc),
            "elapsed_sec": time.perf_counter() - started_at,
        }


def persist_symbol_result(
    *,
    symbol: str,
    res: dict,
    output_dir: Path,
    html_dir: Path,
    summary_rows: list,
    curve_tables: dict,
    symbol_timeseries_frames: list,
    trade_log_frames: list,
    overwatch_log_frames: list,
) -> None:
    summary_rows.append(res["summary_row"])
    curve_tables[symbol] = res["curve_df"]
    symbol_timeseries_frames.append(res["symbol_timeseries_df"])
    trade_log_frames.append(res["trade_log_df"])
    if not res["overwatch_logs_df"].empty:
        overwatch_log_frames.append(res["overwatch_logs_df"])

    symbol_dir = output_dir / symbol
    _ensure_dir(symbol_dir)

    if CFG.WRITE_LEGACY_SYMBOL_CSVS:
        res["signals_df"].to_csv(symbol_dir / f"{symbol}_signals.csv", index=False)
        if CFG.OVERWATCH_ENABLED or not res["overwatch_logs_df"].empty:
            res["overwatch_logs_df"].to_csv(symbol_dir / f"{symbol}_overwatch.csv", index=False)
        res["curve_df"].to_csv(symbol_dir / f"{symbol}_equity_curves.csv", index=False)
        res["stats_df"].to_csv(symbol_dir / f"{symbol}_stats_by_fill_mode.csv", index=False)
        res["stats_live_df"].to_csv(symbol_dir / f"{symbol}_stats_by_fill_mode_live.csv", index=False)
        res["trade_log_df"].to_csv(symbol_dir / f"{symbol}_trade_log_{CFG.PRIMARY_REPORT_FILL_MODE}.csv", index=False)
        res["symbol_timeseries_df"].to_csv(symbol_dir / f"{symbol}_dashboard_timeseries.csv", index=False)

    if CFG.WRITE_PLOTLY_HTML:
        fig = make_symbol_figure(symbol, res["curve_df"], res["raw_df"], res["trade_log_df"], res["summary_row"])
        save_plotly_figure(fig, html_dir / f"{symbol}_interactive_report.html")


def main():
    overall_start = time.perf_counter()
    set_global_determinism(CFG.SEED)
    reset_overwatch_chart_storage()

    output_dir = Path(CFG.OUTPUT_DIR)
    dashboard_dir = Path(CFG.DASHBOARD_DIR)
    html_dir = output_dir / "html"
    _ensure_dir(output_dir)
    _ensure_dir(dashboard_dir)
    if CFG.WRITE_PLOTLY_HTML:
        _ensure_dir(html_dir)

    symbols = CFG.SYMBOLS if CFG.SYMBOLS else discover_symbols(CFG.DATA_DIR)
    if not symbols:
        raise RuntimeError(f"No symbols found in DATA_DIR: {CFG.DATA_DIR}")

    common_report_start_date = compute_common_report_start_date(symbols)
    if common_report_start_date is not None:
        CFG.REPORT_START_DATE = common_report_start_date.strftime("%Y-%m-%d")
        os.environ["REPORT_START_DATE"] = CFG.REPORT_START_DATE
        log_progress(
            f"Using common report start date {CFG.REPORT_START_DATE} "
            f"from REPORT_DAYS_BACK={int(CFG.REPORT_DAYS_BACK)} across {len(symbols)} symbols"
        )

    worker_count = max(1, min(int(getattr(CFG, "NUM_WORKERS", 12)), len(symbols)))
    log_progress(
        f"Starting live report for {len(symbols)} symbols "
        f"(overwatch={'ON' if CFG.OVERWATCH_ENABLED else 'OFF'}, "
        f"report_window={'start_date=' + str(CFG.REPORT_START_DATE) if CFG.REPORT_START_DATE else 'days_back=' + str(CFG.REPORT_DAYS_BACK)}, "
        f"workers={worker_count})"
    )

    summary_rows = []
    curve_tables = {}
    symbol_timeseries_frames = []
    trade_log_frames = []
    overwatch_log_frames = []

    if worker_count == 1:
        for idx, symbol in enumerate(symbols, start=1):
            symbol_start = time.perf_counter()
            log_progress(f"[{idx}/{len(symbols)}] Starting {symbol}")
            safe_res = run_for_symbol_safe(symbol)
            if not safe_res["ok"]:
                log_progress(f"[{idx}/{len(symbols)}] Skipping {symbol} after {safe_res['elapsed_sec']:.1f}s: {safe_res['error']}")
                print(f"[SKIP] {symbol}: {safe_res['error']}")
                continue

            persist_symbol_result(
                symbol=symbol,
                res=safe_res["result"],
                output_dir=output_dir,
                html_dir=html_dir,
                summary_rows=summary_rows,
                curve_tables=curve_tables,
                symbol_timeseries_frames=symbol_timeseries_frames,
                trade_log_frames=trade_log_frames,
                overwatch_log_frames=overwatch_log_frames,
            )
            log_progress(f"[{idx}/{len(symbols)}] Completed {symbol} in {time.perf_counter() - symbol_start:.1f}s")
    else:
        log_progress(f"Launching multiprocessing pool with {worker_count} workers")
        ctx = multiprocessing.get_context("spawn")
        completed = 0
        with ctx.Pool(processes=worker_count) as pool:
            for safe_res in pool.imap_unordered(run_for_symbol_safe, symbols):
                completed += 1
                symbol = str(safe_res["symbol"])
                if not safe_res["ok"]:
                    log_progress(f"[{completed}/{len(symbols)}] Skipping {symbol} after {safe_res['elapsed_sec']:.1f}s: {safe_res['error']}")
                    print(f"[SKIP] {symbol}: {safe_res['error']}")
                    continue

                persist_symbol_result(
                    symbol=symbol,
                    res=safe_res["result"],
                    output_dir=output_dir,
                    html_dir=html_dir,
                    summary_rows=summary_rows,
                    curve_tables=curve_tables,
                    symbol_timeseries_frames=symbol_timeseries_frames,
                    trade_log_frames=trade_log_frames,
                    overwatch_log_frames=overwatch_log_frames,
                )
                log_progress(f"[{completed}/{len(symbols)}] Completed {symbol} in {safe_res['elapsed_sec']:.1f}s")

    if not summary_rows:
        raise RuntimeError("No symbols ran successfully.")

    morning_report = pd.DataFrame(summary_rows).sort_values("symbol").reset_index(drop=True)
    morning_report.to_csv(output_dir / "morning_report.csv", index=False)
    overwatch_logs_df = (
        pd.concat(overwatch_log_frames, ignore_index=True)
        if overwatch_log_frames
        else build_overwatch_logs_df([])
    )
    if CFG.WRITE_LEGACY_SYMBOL_CSVS and (CFG.OVERWATCH_ENABLED or not overwatch_logs_df.empty):
        overwatch_logs_df.to_csv(output_dir / "aggregated_overwatch.csv", index=False)

    symbol_timeseries_df = pd.concat(symbol_timeseries_frames, ignore_index=True) if symbol_timeseries_frames else pd.DataFrame()
    trade_log_df = pd.concat(trade_log_frames, ignore_index=True) if trade_log_frames else pd.DataFrame()
    symbol_metrics_df = compute_symbol_metrics(symbol_timeseries_df, morning_report)

    agg_curve = aggregate_curve_tables(curve_tables)
    agg_curve.to_csv(output_dir / "aggregate_equity_curves.csv", index=False)

    if CFG.WRITE_PLOTLY_HTML:
        agg_fig = make_aggregate_figure(agg_curve)
        save_plotly_figure(agg_fig, html_dir / "aggregate_interactive_report.html")

    save_dashboard_outputs(
        summary_df=morning_report,
        symbol_metrics_df=symbol_metrics_df,
        symbol_timeseries_df=symbol_timeseries_df,
        trade_log_df=trade_log_df,
        aggregate_timeseries_df=agg_curve,
        output_dir=dashboard_dir,
    )

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 240)

    print("\n=== MORNING REPORT ===")
    print(morning_report)
    print("\nSaved live report outputs to:")
    print(output_dir)
    print("\nSaved dashboard-ready outputs to:")
    print(dashboard_dir)
    if CFG.WRITE_PLOTLY_HTML:
        print("\nSaved interactive HTML reports to:")
        print(html_dir)
    log_progress(f"Live report finished in {time.perf_counter() - overall_start:.1f}s")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    multiprocessing.freeze_support()
    main()
