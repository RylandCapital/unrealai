import os

# ─────────────────────────────────────────────────────────────────────────────
# Determinism / runtime env
# Must be set before importing tensorflow
# ─────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("PYTHONHASHSEED", "42")

import io
import random
import time
import gc
import re
import csv
import json
import base64
import multiprocessing
import shutil
import threading
from collections import deque
from pathlib import Path
import importlib.util
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gym
from gym import spaces

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input, Lambda, Add

import psutil

from dotenv import load_dotenv
try:
    from utils.feature_registry import FEATURE_COLS as REGISTRY_FEATURE_COLS
except ModuleNotFoundError:
    from unrealai.utils.feature_registry import FEATURE_COLS as REGISTRY_FEATURE_COLS
try:
    from utils.feature_compat import BACKFILLABLE_FEATURE_COLS, ensure_backfilled_feature_columns
except ModuleNotFoundError:
    from unrealai.utils.feature_compat import BACKFILLABLE_FEATURE_COLS, ensure_backfilled_feature_columns

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# ✅ USER-TUNABLE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
class CFG:

    NUMBER_OF_POOLS = 12
    TASKS_PER_CHILD = 1
    SEED = 42

    COOLDOWN_DAYS = 3
    MIN_HOLD_DAYS = 3
    COOLDOWN_DAYS_AFTER_PROFITABLE_CLOSE = 0
    COOLDOWN_DAYS_AFTER_LOSING_CLOSE = 3
    COOLDOWN_DAYS_AFTER_STOP_EXIT = 1
    COOLDOWN_DAYS_AFTER_TAKE_PROFIT_EXIT = 0

    MOMENTUM_HOLD_BONUS = 0.00025
    MOMENTUM_FLAT_PENALTY = 0.00025
    MOMENTUM_PREMATURE_CLOSE_PENALTY = 0.00025

    OVERWATCH_ENABLED_TRAIN = False
    OVERWATCH_ENABLED_TEST = False
    OVERWATCH_EVERY_N_STEPS = 5
    OVERWATCH_ALLOW_CONSTRAINT_OVERRIDE = True
    SAVE_OVERWATCH_CHARTS = True
    OVERWATCH_CHARTS_DIR = r"P:\10_CWP Trade Department\Ryland\unrealai\unrealai\overwatch_charts\main_w_llm"

    TRADE_OPEN_PENALTY_TRAIN = 0.0000
    TRADE_OPEN_PENALTY_TEST = 0.0000

    SLIPPAGE_RATE = 0.0000

    STOP_LOSS_TRAIN = .15
    TAKE_PROFIT_TRAIN = 1000.0
    STOP_LOSS_TEST = .15
    TAKE_PROFIT_TEST = 1000.0

    TRAIN_ROLLING_YEARS = 2
    TRAIN_ROLLING_STEP_YEARS = 1
    TRAIN_RANDOM_START_MAX = 55

    # Replay buffer reset policy
    # "interval" -> reset every TRAIN_MEMORY_RESET_EVERY episodes
    # "once"     -> reset once at TRAIN_MEMORY_RESET_ONCE_AT_EPISODE
    # "never"    -> never reset
    TRAIN_MEMORY_RESET_MODE = "interval"   # "interval", "once", "never"
    TRAIN_MEMORY_RESET_EVERY = 20
    TRAIN_MEMORY_RESET_ONCE_AT_EPISODE = 20

    # main toggles for speed.
    TRAIN_REPLAY_EVERY_N_STEPS = 21
    BATCHSIZE = 64
    WINDOW_SIZE = 21
    EPISODES = 150
    GREEDY_EVAL_EVERY = 5

    REWARD_ROLLING_MEDIAN_WINDOW = 10
    GREEDY_EVAL_MAX_WINDOWS = None

    PER_ENABLED = True
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_END = 1.0
    PER_BETA_FRAMES = 10_000
    PER_EPS = 1e-6
    PER_CLIP_ABS_TD = None

    OVERWATCH_REASONING_EFFORT = "high"
    MODEL = "gpt-5-mini"

    OUTPUT_DIR = r"P:\10_CWP Trade Department\Ryland\unrealai\unrealai\reports"

    # Model checkpointing
    MODEL_DIR = r"P:\10_CWP Trade Department\Ryland\unrealai\unrealai\models"
    SAVE_MODEL_CHECKPOINTS = True
    RESUME_FROM_CHECKPOINT = False   # first run False; tomorrow after close set True
    REUSE_SAVED_SCALER = True        # when resuming, keep old scaler for continuity

    TRAIN_DIR = Path(r"P:\10_CWP Trade Department\Ryland\unrealai\unrealai\traindata")
    TEST_DIR  = Path(r"P:\10_CWP Trade Department\Ryland\unrealai\unrealai\testdata")

    train_map = {p.stem.upper(): p for p in TRAIN_DIR.glob("*.csv")}
    test_map  = {p.stem.upper(): p for p in TEST_DIR.glob("*.csv")}

    common_symbols = sorted(set(train_map) & set(test_map))

    if not common_symbols:
        raise FileNotFoundError("No matching CSV symbols found in both traindata and testdata.")

    TRAIN_FILES = []
    TEST_FILES = []

    for sym in common_symbols:
        TRAIN_FILES.append(str(train_map[sym]))
        TEST_FILES.append(str(test_map[sym]))

    
    INITIAL_CASH = 100_000_000

    REPORT_VWAP_MODULE_PATH = r"P:\10_CWP Trade Department\Ryland\unrealai\unrealai\utils\overwatch_chart.py"

    # Test execution replay modes
    FILL_MODE_CLOSE_T = "close_t"
    FILL_MODE_OPEN_T1 = "open_t_plus_1"
    FILL_MODE_HL2_T1 = "hl2_t_plus_1"
    FILL_MODE_OPEN_T1_DISCOUNT_ONLY = "open_t_plus_1_discount_only"
    ALL_FILL_MODES = [
        FILL_MODE_CLOSE_T,
        FILL_MODE_OPEN_T1,
        FILL_MODE_HL2_T1,
        FILL_MODE_OPEN_T1_DISCOUNT_ONLY,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility helpers
# ─────────────────────────────────────────────────────────────────────────────
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


def aggressive_worker_cleanup(log_queue=None, symbol=""):
    cleanup_notes = []

    try:
        plt.close("all")
        cleanup_notes.append("matplotlib_closed")
    except Exception:
        pass

    try:
        tf.keras.backend.clear_session()
        cleanup_notes.append("keras_session_cleared")
    except Exception:
        pass

    try:
        tf.compat.v1.reset_default_graph()
        cleanup_notes.append("tf_graph_reset")
    except Exception:
        pass

    gc_total = 0
    for _ in range(3):
        try:
            gc_total += int(gc.collect())
        except Exception:
            pass
    cleanup_notes.append(f"gc_objects={gc_total}")

    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32")
            psapi = ctypes.WinDLL("psapi")
            handle = kernel32.GetCurrentProcess()
            trimmed = bool(psapi.EmptyWorkingSet(handle))
            cleanup_notes.append("working_set_trimmed" if trimmed else "working_set_trim_failed")
        except Exception as exc:
            cleanup_notes.append(f"working_set_trim_error={exc.__class__.__name__}")

    child_log(log_queue, f"[{symbol}] Worker cleanup: {', '.join(cleanup_notes)}")


@tf.keras.utils.register_keras_serializable(package="Custom")
def mean_center_advantage(x):
    return x - tf.reduce_mean(x, axis=1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# Replay reset policy helpers
# ─────────────────────────────────────────────────────────────────────────────
def should_reset_memory_after_episode(episode_idx_0_based: int) -> bool:
    e = int(episode_idx_0_based)
    if e <= 0:
        return False

    mode = str(getattr(CFG, "TRAIN_MEMORY_RESET_MODE", "interval")).strip().lower()

    if mode == "never":
        return False

    if mode == "interval":
        every = int(getattr(CFG, "TRAIN_MEMORY_RESET_EVERY", 0) or 0)
        return every > 0 and (e % every == 0)

    if mode == "once":
        once_at = int(getattr(CFG, "TRAIN_MEMORY_RESET_ONCE_AT_EPISODE", 0) or 0)
        return once_at > 0 and e == once_at

    raise ValueError(
        f"Invalid CFG.TRAIN_MEMORY_RESET_MODE={CFG.TRAIN_MEMORY_RESET_MODE!r}. "
        f"Use 'interval', 'once', or 'never'."
    )


def describe_memory_reset_policy() -> str:
    mode = str(getattr(CFG, "TRAIN_MEMORY_RESET_MODE", "interval")).strip().lower()

    if mode == "never":
        return "never"

    if mode == "interval":
        every = int(getattr(CFG, "TRAIN_MEMORY_RESET_EVERY", 0) or 0)
        return f"every {every} episodes" if every > 0 else "disabled (invalid interval)"

    if mode == "once":
        once_at = int(getattr(CFG, "TRAIN_MEMORY_RESET_ONCE_AT_EPISODE", 0) or 0)
        return f"once at episode {once_at}" if once_at > 0 else "disabled (invalid once episode)"

    return f"unknown mode={mode}"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_symbol_model_dir(base_dir: str, symbol: str) -> str:
    return os.path.join(str(base_dir), str(symbol))


def get_checkpoint_paths(base_dir: str, symbol: str) -> dict:
    symbol_dir = get_symbol_model_dir(base_dir, symbol)
    return {
        "dir": symbol_dir,
        "policy": os.path.join(symbol_dir, "policy.keras"),
        "target": os.path.join(symbol_dir, "target.keras"),
        "scaler": os.path.join(symbol_dir, "scaler.npz"),
        "state": os.path.join(symbol_dir, "agent_state.json"),
    }


def checkpoint_exists(base_dir: str, symbol: str) -> bool:
    paths = get_checkpoint_paths(base_dir, symbol)
    required = [paths["policy"], paths["target"], paths["scaler"], paths["state"]]
    return all(os.path.exists(p) for p in required)


def save_scaler(scaler, path: str):
    np.savez(
        path,
        mean=scaler.mean_,
        std=scaler.std_,
        feature_cols=np.array(scaler.feature_cols, dtype=object),
    )


def load_scaler(path: str):
    z = np.load(path, allow_pickle=True)
    scaler = ZScoreScaler(list(z["feature_cols"]))
    scaler.mean_ = z["mean"].astype(np.float32)
    scaler.std_ = z["std"].astype(np.float32)
    return scaler


def scaler_feature_layout_matches(scaler, feature_cols) -> bool:
    return list(getattr(scaler, "feature_cols", [])) == list(feature_cols)


def load_keras_model(path: str):
    try:
        return tf.keras.models.load_model(path, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(path)


def save_agent_checkpoint(agent, scaler, symbol: str, base_dir: str, last_trained_date=None):
    paths = get_checkpoint_paths(base_dir, symbol)
    os.makedirs(paths["dir"], exist_ok=True)

    agent.model.save(paths["policy"])
    agent.target_model.save(paths["target"])
    save_scaler(scaler, paths["scaler"])

    state = {
        "symbol": str(symbol),
        "epsilon": float(agent.epsilon),
        "train_step_counter": int(agent.train_step_counter),
        "target_update_freq": int(agent.target_update_freq),
        "last_trained_date": str(last_trained_date) if last_trained_date is not None else None,
        "saved_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(paths["state"], "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_agent_checkpoint(agent, symbol: str, base_dir: str):
    paths = get_checkpoint_paths(base_dir, symbol)

    agent.model = load_keras_model(paths["policy"])
    agent.target_model = load_keras_model(paths["target"])
    scaler = load_scaler(paths["scaler"])

    with open(paths["state"], "r", encoding="utf-8") as f:
        state = json.load(f)

    agent.epsilon = float(state.get("epsilon", agent.epsilon))
    agent.train_step_counter = int(state.get("train_step_counter", 0))
    agent.target_update_freq = int(state.get("target_update_freq", agent.target_update_freq))

    return agent, scaler, state


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────────────────────────────────────
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("SECRET")
OVERWATCH_MODEL = os.getenv("OVERWATCH_MODEL", CFG.MODEL)


def _get_openai_client():
    if not hasattr(_get_openai_client, "_client"):
        if not OPENAI_API_KEY:
            raise RuntimeError("Missing API key. Set OPENAI_API_KEY or SECRET.")
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


overwatch_decision_gpt5mini = overwatch_decision


# ─────────────────────────────────────────────────────────────────────────────
# Lazy chart module loader
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Action constants
# ─────────────────────────────────────────────────────────────────────────────
ACTION_HOLD = 0
ACTION_LONG = 1
ACTION_CLOSE = 2


# ─────────────────────────────────────────────────────────────────────────────
# Feature columns
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = list(REGISTRY_FEATURE_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def logging_listener(q):
    while True:
        msg = q.get()
        if msg == "END":
            break
        print(msg, flush=True)


def child_log(q, msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    q.put(f"[{ts}] {msg}")


def save_overwatch_chart(fig, *, symbol: str, review_date: str, step: int, raw_action: int, source_tag: str) -> str | None:
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


def _format_pct_or_na(value, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100.0:+.{decimals}f}%"


def _format_bool_or_na(value) -> str:
    if value is None:
        return "N/A"
    return "Yes" if bool(value) else "No"


def _get_env_raw_feature(env, feature_name: str):
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


def reset_overwatch_chart_storage(log_queue=None) -> None:
    overwatch_active = bool(getattr(CFG, "OVERWATCH_ENABLED_TRAIN", False)) or bool(getattr(CFG, "OVERWATCH_ENABLED_TEST", False))
    if not overwatch_active:
        return
    if not bool(getattr(CFG, "SAVE_OVERWATCH_CHARTS", False)):
        return

    root = Path(getattr(CFG, "OVERWATCH_CHARTS_DIR", "")).expanduser()
    if not str(root):
        return

    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    msg = f"Reset overwatch chart storage: {root}"
    if log_queue is not None:
        child_log(log_queue, msg)
    else:
        print(msg, flush=True)


def get_ram_usage():
    return psutil.virtual_memory().percent


def make_rolling_windows(df: pd.DataFrame,
                         *,
                         years: int = 2,
                         step_years: int = 1,
                         trading_days_per_year: int = 252):
    if df is None or df.empty:
        return []

    window_len = int(years * trading_days_per_year)
    step_len = int(step_years * trading_days_per_year)

    n = len(df)
    if n <= window_len:
        return [(0, n)]

    windows = []
    start = 0
    while start + window_len <= n:
        windows.append((start, start + window_len))
        start += step_len

    if windows and windows[-1][1] < n:
        end = n
        start = max(0, end - window_len)
        if not windows or windows[-1] != (start, end):
            windows.append((start, end))

    uniq = []
    seen = set()
    for s, e in windows:
        key = (int(s), int(e))
        if key not in seen and e - s > 10:
            uniq.append(key)
            seen.add(key)

    return uniq


def select_fixed_eval_windows(windows, max_windows=None):
    windows = list(windows or [])
    if not windows:
        return []
    if max_windows is None or len(windows) <= int(max_windows):
        return windows
    idxs = np.linspace(0, len(windows) - 1, num=int(max_windows), dtype=int)
    idxs = sorted(set(int(i) for i in idxs))
    return [windows[i] for i in idxs]


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
    """
    Loads:
      - Date
      - FEATURE_COLS
      - adjusted_close
      - optional eval OHLC columns mapped into:
            eval_open, eval_high, eval_low
    """
    header = pd.read_csv(csv_path, nrows=0)
    existing_cols = header.columns.tolist()
    existing_set = set(existing_cols)

    missing_features = [c for c in FEATURE_COLS if c not in existing_set]
    missing_non_backfillable = [c for c in missing_features if c not in BACKFILLABLE_FEATURE_COLS]
    required_core = [c for c in ["Date", "adjusted_close"] if c not in existing_set]

    if required_core or missing_non_backfillable:
        raise ValueError(
            f"CSV missing required columns: {required_core + missing_non_backfillable}\n"
            f"File: {csv_path}"
        )

    open_col, high_col, low_col = resolve_eval_price_columns(existing_cols)

    if require_eval_prices and (open_col is None or high_col is None or low_col is None):
        raise ValueError(
            "Test CSV must contain open/high/low or adjusted_open/adjusted_high/adjusted_low.\n"
            f"File: {csv_path}"
        )

    usecols = ["Date"] + [c for c in FEATURE_COLS if c in existing_set] + ["adjusted_close"]
    for c in [open_col, high_col, low_col]:
        if c is not None and c not in usecols:
            usecols.append(c)

    dtypes = {c: "float32" for c in usecols if c != "Date"}

    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        parse_dates=["Date"],
        dtype=dtypes,
    )
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
        raise ValueError(
            f"NaNs found in required columns: {bad_cols}\n"
            f"File: {csv_path}"
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Z-score scaler
# ─────────────────────────────────────────────────────────────────────────────
class ZScoreScaler:
    def __init__(self, feature_cols):
        self.feature_cols = list(feature_cols)
        self.mean_ = None
        self.std_ = None

    def fit(self, df: pd.DataFrame):
        x = df[self.feature_cols].astype(np.float64)
        mean = x.mean(axis=0).to_numpy()
        std = x.std(axis=0, ddof=0).to_numpy()
        std = np.where(std == 0, 1.0, std)
        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not fit() yet.")
        out = df.copy()
        vals = out[self.feature_cols].to_numpy(dtype=np.float32)
        vals = (vals - self.mean_) / self.std_
        out[self.feature_cols] = vals
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Training environment
# ─────────────────────────────────────────────────────────────────────────────
class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 df,
                 raw_feature_df=None,
                 window_size=20,
                 initial_cash=1_000_000.0,
                 slippage_rate=0.0002,
                 stop_loss=1_000,
                 take_profit=1_000,
                 cooldown_days=0,
                 min_hold_days=0,
                 trade_open_penalty=0.0025,
                 profitable_close_cooldown_days=None,
                 losing_close_cooldown_days=None,
                 stop_exit_cooldown_days=None,
                 take_profit_exit_cooldown_days=None,
                 momentum_hold_bonus=0.0,
                 momentum_flat_penalty=0.0,
                 momentum_premature_close_penalty=0.0):
        super().__init__()

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
        self.trade_open_penalty = float(trade_open_penalty)
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
        self.momentum_hold_bonus = float(momentum_hold_bonus)
        self.momentum_flat_penalty = float(momentum_flat_penalty)
        self.momentum_premature_close_penalty = float(momentum_premature_close_penalty)

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
        self.portfolio_dim = len(self.portfolio_cols)
        self.obs_dim = len(self.feature_cols) + self.portfolio_dim

        self.features = df[self.feature_cols].to_numpy(dtype=np.float32)
        self.raw_features = (
            self.raw_feature_df[self.feature_cols].to_numpy(dtype=np.float32)
            if self.raw_feature_df is not None
            else None
        )
        self.prices = df["adjusted_close"].to_numpy(dtype=np.float32)
        self.dates = df["Date"].to_numpy(dtype="datetime64[ns]")
        self.n_steps = int(len(df))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.obs_dim),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

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

        if self.cooldown_days > 0:
            cooldown_frac = float(self.cooldown_remaining) / float(self.cooldown_days)
        else:
            cooldown_frac = 0.0

        return np.array(
            [pos, in_trade, entry_over_price, unreal_pnl, days_in_trade, cooldown_frac],
            dtype=np.float32
        )

    def _get_obs(self):
        start = self.current_step - self.window_size
        market_window = self.features[start:self.current_step]

        obs_idx = max(0, min(self.current_step - 1, self.n_steps - 1))
        obs_price = float(self.prices[obs_idx])

        p = self._portfolio_state(obs_price)
        p_mat = np.repeat(p[np.newaxis, :], self.window_size, axis=0)

        obs = np.concatenate([market_window, p_mat], axis=1).astype(np.float32)
        return obs

    def _mark_to_market(self, price: float):
        if self.position == 1:
            self.equity = self.cash + self.shares * price
        else:
            self.equity = self.cash

    def _update_benchmark(self, price: float):
        self.bh_equity = float(self.bh_shares * price)

    def _feature_value(self, name: str, idx: int | None = None, default: float = np.nan) -> float:
        if self.raw_feature_df is None or name not in self.raw_feature_df.columns:
            return float(default)
        use_idx = self.current_step if idx is None else int(idx)
        use_idx = min(max(use_idx, 0), self.n_steps - 1)
        try:
            value = self.raw_feature_df.at[use_idx, name]
            return float(value) if pd.notna(value) else float(default)
        except Exception:
            return float(default)

    def _momentum_regime_strength(self, idx: int | None = None) -> float:
        use_idx = self.current_step if idx is None else int(idx)
        use_idx = min(max(use_idx, 0), self.n_steps - 1)

        checks = []
        breakout_flags = [
            self._feature_value("new_high_20", use_idx, 0.0) >= 0.5,
            self._feature_value("new_high_60", use_idx, 0.0) >= 0.5,
            self._feature_value("new_high_252", use_idx, 0.0) >= 0.5,
        ]
        breakout_now = any(breakout_flags)

        checks.extend([
            self._feature_value("dist_sma_21", use_idx, 0.0) > 0.0,
            self._feature_value("dist_sma_55", use_idx, 0.0) > 0.0,
            self._feature_value("sma_21_slope_10", use_idx, 0.0) > 0.0,
            self._feature_value("ret_21", use_idx, 0.0) > 0.0,
            self._feature_value("ret_63", use_idx, 0.0) > 0.0,
            self._feature_value("ret_126", use_idx, 0.0) > 0.0,
            self._feature_value("mom_score_63", use_idx, 0.0) > 0.0,
            self._feature_value("mom_score_126", use_idx, 0.0) > 0.0,
            self._feature_value("trend_strength_21_200", use_idx, 0.0) > 0.0,
            self._feature_value("rel_ret_21_vs_spy", use_idx, 0.0) > 0.0,
            self._feature_value("rel_ret_63_vs_spy", use_idx, 0.0) > 0.0,
            self._feature_value("rel_ret_21_vs_sector", use_idx, 0.0) > 0.0,
            self._feature_value("rel_ret_63_vs_sector", use_idx, 0.0) > 0.0,
            (
                self._feature_value("dist_prev_high_20", use_idx, -1.0) > -0.02
                or self._feature_value("dist_prev_high_60", use_idx, -1.0) > -0.03
                or self._feature_value("dist_prev_high_252", use_idx, -1.0) > -0.05
                or breakout_now
            ),
            (
                self._feature_value("volume_rel_20", use_idx, 0.0) > 1.0
                or self._feature_value("volume_rel_63", use_idx, 0.0) > 1.0
                or self._feature_value("dollar_volume_rel_20", use_idx, 0.0) > 1.0
            ),
        ])

        strength = float(np.mean(checks)) if checks else 0.0
        if breakout_now:
            strength = min(1.0, strength + 0.10)
        return strength

    def _cooldown_days_after_exit(self, pnl_pct: float, exit_reason: str) -> int:
        reason = str(exit_reason or "").strip().lower()
        if reason == "stop_loss":
            return int(self.stop_exit_cooldown_days)
        if reason == "take_profit":
            return int(self.take_profit_exit_cooldown_days)
        if float(pnl_pct) > 0.0:
            return int(self.profitable_close_cooldown_days)
        return int(self.losing_close_cooldown_days)

    def _apply_momentum_reward_shaping(self, reward: float, *, action: int, pre_position: int, close_reason: str) -> float:
        strength = self._momentum_regime_strength(self.last_price_idx)
        if strength < 0.55:
            return float(reward)

        if pre_position == 1 and self.position == 1 and action == ACTION_HOLD:
            reward += float(self.momentum_hold_bonus) * strength
        elif pre_position == 0 and self.position == 0 and action == ACTION_HOLD and self.cooldown_remaining == 0:
            reward -= float(self.momentum_flat_penalty) * strength
        elif pre_position == 1 and self.position == 0 and close_reason == "discretionary":
            reward -= float(self.momentum_premature_close_penalty) * strength

        return float(reward)

    def _do_close(self, price: float):
        if self.position == 0:
            return 0.0

        gross = self.shares * price
        slip_cost = gross * self.slippage_rate
        proceeds = gross - slip_cost
        pnl_dollars = proceeds - (self.shares * self.entry_price)
        self.cash += proceeds

        denom = (self.shares * self.entry_price)
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

        prev_equity = float(self.equity)
        prev_bh_equity = float(self.bh_equity)

        if self.current_step >= self.n_steps - 1:
            price = float(self.prices[self.current_step])
            self.last_price_idx = int(self.current_step)

            if self.position != 0:
                self._do_close(price)

            self._mark_to_market(price)
            self._update_benchmark(price)

            reward, blew_up = self._log_excess_return_reward(prev_equity, prev_bh_equity)

            self.done = True
            self.prev_equity = float(self.equity)
            self.prev_bh_equity = float(self.bh_equity)

            info = {"bankrupt": True} if blew_up else {}
            return self._get_obs(), float(reward), True, info

        price = float(self.prices[self.current_step])
        self.last_price_idx = int(self.current_step)

        if self.position != 0 and action == ACTION_CLOSE and (self.can_discretionary_close() or force_close):
            self.close_hits += 1
            close_pct = self._do_close(price)
            close_reason = "discretionary"
            self.cooldown_remaining = self._cooldown_days_after_exit(close_pct, close_reason)

        if self.position != 0:
            pct_move = (price - self.entry_price) / (self.entry_price + 1e-12)
            if pct_move <= -self.stop_loss or pct_move >= self.take_profit:
                close_reason = "stop_loss" if pct_move <= -self.stop_loss else "take_profit"
                close_pct = self._do_close(price)
                self.cooldown_remaining = self._cooldown_days_after_exit(close_pct, close_reason)

        opened_trade = False
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

        if opened_trade and self.trade_open_penalty:
            reward -= float(self.trade_open_penalty)

        reward = self._apply_momentum_reward_shaping(
            reward,
            action=int(action),
            pre_position=pre_position,
            close_reason=close_reason,
        )

        self.current_step += 1
        if self.current_step >= self.n_steps:
            self.done = True

        self.prev_equity = float(self.equity)
        self.prev_bh_equity = float(self.bh_equity)

        info = {"bankrupt": True} if blew_up else {}
        if blew_up:
            self.done = True

        return self._get_obs(), float(reward), self.done, info

    def render(self, mode="human"):
        print(
            f"Step={self.current_step}, Pos={self.position}, Shares={self.shares:.2f}, "
            f"Cash={self.cash:.2f}, Eq={self.equity:.2f}, BH={self.bh_equity:.2f}"
        )


def build_env(df, *, mode: str, window_size: int, initial_cash: float, raw_feature_df=None) -> TradingEnv:
    mode = (mode or "").lower().strip()
    if mode not in {"train", "test"}:
        raise ValueError("mode must be 'train' or 'test'")

    if mode == "train":
        stop_loss = float(CFG.STOP_LOSS_TRAIN)
        take_profit = float(CFG.TAKE_PROFIT_TRAIN)
        trade_open_penalty = float(CFG.TRADE_OPEN_PENALTY_TRAIN)
    else:
        stop_loss = float(CFG.STOP_LOSS_TEST)
        take_profit = float(CFG.TAKE_PROFIT_TEST)
        trade_open_penalty = float(CFG.TRADE_OPEN_PENALTY_TEST)

    return TradingEnv(
        df,
        raw_feature_df=raw_feature_df,
        window_size=int(window_size),
        initial_cash=float(initial_cash),
        slippage_rate=float(CFG.SLIPPAGE_RATE),
        stop_loss=stop_loss,
        take_profit=take_profit,
        cooldown_days=int(CFG.COOLDOWN_DAYS),
        min_hold_days=int(CFG.MIN_HOLD_DAYS),
        trade_open_penalty=trade_open_penalty,
        profitable_close_cooldown_days=int(CFG.COOLDOWN_DAYS_AFTER_PROFITABLE_CLOSE),
        losing_close_cooldown_days=int(CFG.COOLDOWN_DAYS_AFTER_LOSING_CLOSE),
        stop_exit_cooldown_days=int(CFG.COOLDOWN_DAYS_AFTER_STOP_EXIT),
        take_profit_exit_cooldown_days=int(CFG.COOLDOWN_DAYS_AFTER_TAKE_PROFIT_EXIT),
        momentum_hold_bonus=float(CFG.MOMENTUM_HOLD_BONUS),
        momentum_flat_penalty=float(CFG.MOMENTUM_FLAT_PENALTY),
        momentum_premature_close_penalty=float(CFG.MOMENTUM_PREMATURE_CLOSE_PENALTY),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PER
# ─────────────────────────────────────────────────────────────────────────────
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, state_shape, alpha: float, beta_start: float, beta_end: float,
                 beta_frames: int, eps: float):
        self.capacity = int(capacity)
        self.state_shape = tuple(state_shape)
        self.alpha = float(alpha)
        self.beta = float(beta_start)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_frames = int(max(1, beta_frames))
        self.beta_step = (self.beta_end - self.beta_start) / float(self.beta_frames)
        self.eps = float(eps)

        self.states = np.zeros((self.capacity, *self.state_shape), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *self.state_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.uint8)

        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.max_priority = 1.0

        self.pos = 0
        self.size = 0

    def __len__(self):
        return int(self.size)

    def add(self, s, a, r, s2, d):
        i = int(self.pos)
        self.states[i] = s
        self.next_states[i] = s2
        self.actions[i] = int(a)
        self.rewards[i] = float(r)
        self.dones[i] = 1 if bool(d) else 0

        self.priorities[i] = float(self.max_priority)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        if self.size == 0:
            raise RuntimeError("Cannot sample from empty buffer.")

        batch_size = int(batch_size)
        self.beta = min(self.beta_end, self.beta + self.beta_step)

        p = self.priorities[:self.size].astype(np.float64)
        p = np.maximum(p, self.eps)
        p_alpha = p ** self.alpha
        probs = p_alpha / np.sum(p_alpha)

        replace = self.size < batch_size
        idxs = np.random.choice(self.size, size=batch_size, replace=replace, p=probs).astype(np.int32)

        weights = (self.size * probs[idxs]) ** (-self.beta)
        weights /= (weights.max() + 1e-12)
        weights = weights.astype(np.float32)

        S = self.states[idxs]
        A = self.actions[idxs]
        R = self.rewards[idxs]
        S2 = self.next_states[idxs]
        D = self.dones[idxs].astype(np.bool_)

        return S, A, R, S2, D, idxs, weights

    def update_priorities(self, idxs, td_errors):
        td = np.asarray(td_errors, dtype=np.float32)
        td = np.abs(td)

        if CFG.PER_CLIP_ABS_TD is not None:
            td = np.minimum(td, float(CFG.PER_CLIP_ABS_TD))

        new_p = td + self.eps
        idxs = np.asarray(idxs, dtype=np.int32)
        self.priorities[idxs] = new_p
        mx = float(new_p.max()) if new_p.size else 0.0
        if mx > self.max_priority:
            self.max_priority = mx


# ─────────────────────────────────────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_size, action_size,
                 learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.999,
                 epsilon_min=0.05, batch_size=32,
                 memory_size=10_000):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory_size = int(memory_size)

        self.train_step_counter = 0
        self.target_update_freq = 50
        self.overwatch_counter = 0
        self.override_count = 0
        self.overwatch_logs = []
        self.use_episode_epsilon_schedule = True
        self.overwatch_enabled = True

        self.per_enabled = bool(CFG.PER_ENABLED)

        if self.per_enabled:
            self.memory = PrioritizedReplayBuffer(
                capacity=self.memory_size,
                state_shape=self.state_size,
                alpha=float(CFG.PER_ALPHA),
                beta_start=float(CFG.PER_BETA_START),
                beta_end=float(CFG.PER_BETA_END),
                beta_frames=int(CFG.PER_BETA_FRAMES),
                eps=float(CFG.PER_EPS),
            )
        else:
            self.memory = deque(maxlen=self.memory_size)

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self) -> Model:
        inp = Input(shape=self.state_size)
        x = Flatten()(inp)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)

        v = Dense(32, activation="relu")(x)
        v = Dense(1, activation="linear")(v)

        a = Dense(32, activation="relu")(x)
        a = Dense(self.action_size, activation="linear")(a)

        a_norm = Lambda(mean_center_advantage, name="adv_center")(a)
        q = Add()([v, a_norm])

        m = Model(inputs=inp, outputs=q)
        m.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse"
        )
        return m

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, s, a, r, s2, d):
        if self.per_enabled:
            self.memory.add(s, a, r, s2, d)
        else:
            self.memory.append((s, a, r, s2, d))

    def reset_memory(self):
        if self.per_enabled:
            self.memory = PrioritizedReplayBuffer(
                capacity=self.memory_size,
                state_shape=self.state_size,
                alpha=float(CFG.PER_ALPHA),
                beta_start=float(CFG.PER_BETA_START),
                beta_end=float(CFG.PER_BETA_END),
                beta_frames=int(CFG.PER_BETA_FRAMES),
                eps=float(CFG.PER_EPS),
            )
        else:
            self.memory = deque(maxlen=self.memory_size)

    def _valid_actions(self, env) -> list:
        if env is None:
            return list(range(self.action_size))

        pos = int(getattr(env, "position", 0))
        cd = int(getattr(env, "cooldown_remaining", 0))

        if pos == 0:
            if cd > 0:
                return [ACTION_HOLD]
            return [ACTION_HOLD, ACTION_LONG]

        if hasattr(env, "can_discretionary_close") and not env.can_discretionary_close():
            return [ACTION_HOLD]

        return [ACTION_HOLD, ACTION_CLOSE]

    def _valid_actions_from_state(self, state_window: np.ndarray) -> list:
        if state_window is None or not isinstance(state_window, np.ndarray):
            return list(range(self.action_size))

        try:
            last_row = state_window[-1]
            base = len(FEATURE_COLS)
            pos = float(last_row[base + 0])
            days_in_trade_scaled = float(last_row[base + 4])
            cooldown_frac = float(last_row[base + 5])
        except Exception:
            return list(range(self.action_size))

        pos_int = 1 if pos >= 0.5 else 0
        in_cooldown = cooldown_frac > 1e-6

        if pos_int == 0:
            if in_cooldown:
                return [ACTION_HOLD]
            return [ACTION_HOLD, ACTION_LONG]

        min_hold_days = int(getattr(CFG, "MIN_HOLD_DAYS", 0) or 0)
        days_in_trade = float(days_in_trade_scaled) * 252.0

        if min_hold_days > 0 and days_in_trade + 1e-9 < float(min_hold_days):
            return [ACTION_HOLD]

        return [ACTION_HOLD, ACTION_CLOSE]

    def _mask_q_values(self, q_vals: np.ndarray, valid_actions: list) -> np.ndarray:
        masked = np.array(q_vals, copy=True)
        invalid = set(range(self.action_size)) - set(valid_actions)
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
        valid_actions = self._valid_actions(env) if env is not None else self._valid_actions_from_state(state)
        q_vals = self.model(state[np.newaxis, :], training=False).numpy()[0]
        q_masked = self._mask_q_values(q_vals, valid_actions)
        action = int(np.argmax(q_masked))
        if action not in valid_actions:
            action = int(valid_actions[0])
        return action

    def act(self, state, symbol=None, env=None, use_overwatch=True):
        self.overwatch_counter += 1

        allow_overwatch = (
            bool(use_overwatch)
            and bool(self.overwatch_enabled)
            and symbol is not None
            and env is not None
        )

        valid_actions = self._valid_actions(env)

        if np.random.rand() <= self.epsilon:
            raw_action = int(random.choice(valid_actions))
        else:
            q_vals = self.model(state[np.newaxis, :], training=False).numpy()[0]
            q_masked = self._mask_q_values(q_vals, valid_actions)
            raw_action = int(np.argmax(q_masked))
            if raw_action not in valid_actions:
                raw_action = int(valid_actions[0])

        every_n = max(1, int(CFG.OVERWATCH_EVERY_N_STEPS))
        if allow_overwatch and (self.overwatch_counter % every_n == 0):
            chart_module = get_chart_module()
            fig = chart_module.build_fig(symbol, end=env.current_datetime.strftime("%Y-%m-%d"))

            fig.text(
                0.01, 0.99, build_overwatch_decision_card(env, raw_action=int(raw_action), valid_actions=valid_actions),
                va="top", ha="left",
                color="white", fontsize=8,
                bbox=dict(facecolor="black", alpha=0.6)
            )

            chart_path = save_overwatch_chart(
                fig,
                symbol=symbol,
                review_date=env.current_datetime.strftime("%Y-%m-%d"),
                step=int(env.current_step),
                raw_action=int(raw_action),
                source_tag="overwatch",
            )
            img_bytes = fig_to_bytes(fig)
            plt.close(fig)

            img_b64_str = base64.b64encode(img_bytes).decode("utf-8")

            try:
                resp_txt = overwatch_decision(raw_action, img_b64_str)
            except Exception as e:
                self.overwatch_logs.append({
                    "symbol": symbol,
                    "step": int(env.current_step),
                    "date": env.current_datetime.strftime("%Y-%m-%d"),
                    "position": int(env.position),
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
                })
                return raw_action

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

            self.overwatch_logs.append({
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
            })
            return final_action

        return raw_action

    def replay(self):
        if self.per_enabled:
            if len(self.memory) < self.batch_size:
                return
            S, A, R, S2, D, idxs, isw = self.memory.sample(self.batch_size)
        else:
            if len(self.memory) < self.batch_size:
                return
            minibatch = random.sample(self.memory, self.batch_size)
            S = np.array([m[0] for m in minibatch], dtype=np.float32)
            A = np.array([m[1] for m in minibatch], dtype=np.int32)
            R = np.array([m[2] for m in minibatch], dtype=np.float32)
            S2 = np.array([m[3] for m in minibatch], dtype=np.float32)
            D = np.array([m[4] for m in minibatch], dtype=np.bool_)
            idxs = None
            isw = None

        q_curr = self.model.predict(S, verbose=0)
        target = np.array(q_curr, copy=True)

        q_next_online = self.model.predict(S2, verbose=0)
        q_next_target = self.target_model.predict(S2, verbose=0)

        td_errors = np.zeros((len(A),), dtype=np.float32)

        for i in range(len(A)):
            a_taken = int(A[i])

            if bool(D[i]):
                y = float(R[i])
            else:
                valid = self._valid_actions_from_state(S2[i])
                q_masked = self._mask_q_values(q_next_online[i], valid)
                a_star = int(np.argmax(q_masked))
                y = float(R[i] + self.gamma * float(q_next_target[i][a_star]))

            td = y - float(q_curr[i][a_taken])
            td_errors[i] = float(td)
            target[i][a_taken] = y

        if self.per_enabled:
            self.model.fit(S, target, sample_weight=isw, epochs=1, verbose=0)
            self.memory.update_priorities(idxs, td_errors)
        else:
            self.model.fit(S, target, epochs=1, verbose=0)

        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.update_target_model()

        if (not self.use_episode_epsilon_schedule) and self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────
def summarize_trades(trades, title="Trades Summary", log_queue=None):
    close_trades = [t for t in trades if t[0] == "close_long"]
    pnls = np.array([pct for *_, pct, _ in close_trades if pct is not None], dtype=np.float64)

    compounded = float(np.prod(1.0 + pnls) - 1.0) if pnls.size else np.nan
    avg_pct = float(np.nanmean(pnls)) if pnls.size else np.nan
    median_pct = float(np.nanmedian(pnls)) if pnls.size else np.nan

    def _log(msg):
        child_log(log_queue, msg) if log_queue else print(msg)

    _log("\n" + "=" * 40)
    _log(title)
    _log("=" * 40)
    _log(f" Compounded Realized Return (%): {compounded * 100:.2f}%")
    _log(f" Avg        Realized PnL (%):   {avg_pct * 100:.2f}%")
    _log(f" Median     Realized PnL (%):   {median_pct * 100:.2f}%")
    _log(f" Closed Trades:                 {len(close_trades)}")
    _log("=" * 40 + "\n")

    return {
        "compounded_realized_return_pct": compounded,
        "avg_pnl_pct": avg_pct,
        "median_pnl_pct": median_pct,
        "closed_trades": len(close_trades),
    }


def compute_portfolio_stats(equity_curve, invested_curve, trades,
                            initial_value, buy_hold_curve=None):
    if not equity_curve:
        return {}

    eq = np.array(equity_curve, dtype=np.float64)
    inv = np.array(invested_curve, dtype=np.float64) if invested_curve is not None else np.zeros_like(eq)

    final_v = float(eq[-1])
    tot_ret = (final_v - initial_value) / initial_value
    ann_ret = ((final_v / initial_value) ** (252 / len(eq)) - 1
               if len(eq) > 1 else np.nan)

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

        stats.update({
            "buy_hold_total_return_pct": bhr * 100,
            "alpha_over_buy_hold": (tot_ret - bhr) * 100,
            "buy_hold_sharpe_ratio": bh_sr,
            "buy_hold_max_drawdown_pct": bh_max_dd * 100,
        })

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_reward_only(reward_hist, save_path, symbol=""):
    rewards = np.array(reward_hist, dtype=np.float64)
    roll_med = pd.Series(rewards).rolling(
        window=max(1, int(CFG.REWARD_ROLLING_MEDIAN_WINDOW)),
        min_periods=1
    ).median().to_numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(rewards, label="Episode Reward")
    plt.plot(roll_med, label=f"{CFG.REWARD_ROLLING_MEDIAN_WINDOW}-Ep Rolling Median")
    plt.title(f"{symbol} Reward (Log Excess vs BH)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_diagnostics(train_diag, save_path, symbol=""):
    rewards = np.array(train_diag["reward_hist"], dtype=np.float64)
    roll_med = np.array(train_diag["rolling_median_reward"], dtype=np.float64)
    time_in_market = np.array(train_diag["time_in_market_pct"], dtype=np.float64)
    opens = np.array(train_diag["opens_per_episode"], dtype=np.float64)
    closes = np.array(train_diag["closes_per_episode"], dtype=np.float64)
    eval_eps = np.array(train_diag["eval_episodes"], dtype=np.int32)
    eval_rewards = np.array(train_diag["eval_rewards"], dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=True)

    axes[0].plot(rewards, label="Episode Reward")
    axes[0].plot(roll_med, label=f"{CFG.REWARD_ROLLING_MEDIAN_WINDOW}-Ep Rolling Median")
    axes[0].set_title(f"{symbol} Training Reward (Log Excess vs BH)")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    axes[1].plot(eval_eps, eval_rewards, marker="o", label=f"Greedy Eval Every {CFG.GREEDY_EVAL_EVERY} Episodes")
    axes[1].set_title(f"{symbol} Greedy Eval Reward on Fixed Windows")
    axes[1].set_ylabel("Eval Reward")
    axes[1].legend()

    axes[2].plot(time_in_market, label="Time in Market %")
    axes[2].set_title(f"{symbol} Time in Market by Episode")
    axes[2].set_ylabel("% of Steps")
    axes[2].legend()

    axes[3].plot(opens, label="Open Count")
    axes[3].plot(closes, label="Close Count")
    axes[3].set_title(f"{symbol} Opens / Closes per Episode")
    axes[3].set_xlabel("Episode")
    axes[3].set_ylabel("Count")
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_fill_mode_display_name(fill_mode: str) -> str:
    if fill_mode == CFG.FILL_MODE_CLOSE_T:
        return "Close T0"
    if fill_mode == CFG.FILL_MODE_OPEN_T1:
        return "Open T+1"
    if fill_mode == CFG.FILL_MODE_HL2_T1:
        return "HL2 T+1"
    if fill_mode == CFG.FILL_MODE_OPEN_T1_DISCOUNT_ONLY:
        return "Open T+1 Discount Only"
    return str(fill_mode)


def plot_symbol_test_results(symbol, result, save_path):
    dates = result["dates"]
    fills = result["fills"]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(f"{symbol} Test Equity Curves", "Invested Capital by Fill Mode"),
        vertical_spacing=0.1
    )

    fig.add_trace(
        go.Scatter(x=dates, y=result["buy_hold_values"], mode="lines", name="Buy & Hold"),
        row=1, col=1
    )

    for mode in CFG.ALL_FILL_MODES:
        label = get_fill_mode_display_name(mode)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=fills[mode]["portfolio_values"],
                mode="lines",
                name=f"Strategy {label}"
            ),
            row=1, col=1
        )

    for mode in CFG.ALL_FILL_MODES:
        label = get_fill_mode_display_name(mode)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=fills[mode]["invested_history"],
                mode="lines",
                name=f"Invested {label}"
            ),
            row=2, col=1
        )

    fig.update_layout(
        title=f"{symbol} Test Backtest: Buy & Hold vs Execution Assumptions",
        hovermode="x unified"
    )
    fig.write_html(save_path)


def plot_aggregated_results_multi(curves_dict, invested_dict, stats_by_mode, dates, save_path=None):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Aggregated Equity Curves", "Aggregated Invested Capital"),
        vertical_spacing=0.1
    )

    fig.add_trace(
        go.Scatter(x=dates, y=curves_dict["buy_hold"], mode="lines", name="Buy & Hold"),
        row=1, col=1
    )

    for mode in CFG.ALL_FILL_MODES:
        label = get_fill_mode_display_name(mode)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=curves_dict[mode],
                mode="lines",
                name=f"Strategy {label}"
            ),
            row=1, col=1
        )

    for mode in CFG.ALL_FILL_MODES:
        label = get_fill_mode_display_name(mode)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=invested_dict[mode],
                mode="lines",
                name=f"Invested {label}"
            ),
            row=2, col=1
        )

    fig.update_layout(
        title="Aggregated Performance: Buy & Hold vs Execution Assumptions",
        hovermode="x unified"
    )

    if save_path:
        fig.write_html(save_path)

    print("\n--- Aggregated Stats by Fill Mode ---")
    for mode, stats in stats_by_mode.items():
        print(f"\n[{mode}]")
        for k, v in stats.items():
            if isinstance(v, (int, float, np.floating)) and not pd.isna(v):
                suffix = "%" if "pct" in k else ""
                print(f"{k}: {v:.2f}{suffix}")
            else:
                print(f"{k}: {v}")
    print("-------------------------------------\n")


# ─────────────────────────────────────────────────────────────────────────────
# Greedy eval helper
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_agent_on_fixed_windows(agent, df, fixed_windows, *,
                                    window_size=20, initial_cash=100_000):
    if not fixed_windows:
        return np.nan

    rewards = []
    for win_start, win_end in fixed_windows:
        df_win = df.iloc[win_start:win_end].reset_index(drop=True)
        env = build_env(df_win, mode="train", window_size=window_size, initial_cash=initial_cash)
        state = env.reset(start_offset=0)

        total_reward = 0.0
        while True:
            action = agent.act_greedy(state, env=env)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)

    return float(np.mean(rewards)) if rewards else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Train helper
# ─────────────────────────────────────────────────────────────────────────────
def train_agent_on_df(agent, df, *, episodes=10, window_size=20,
                      initial_cash=100_000,
                      report_csv_path=None, log_queue=None, symbol=""):
    reward_hist = []
    all_trades = []

    time_in_market_pct_hist = []
    opens_hist = []
    closes_hist = []
    eval_episodes = []
    eval_rewards = []

    windows = make_rolling_windows(
        df,
        years=int(CFG.TRAIN_ROLLING_YEARS),
        step_years=int(CFG.TRAIN_ROLLING_STEP_YEARS),
        trading_days_per_year=252
    )
    if not windows:
        windows = [(0, len(df))]

    fixed_eval_windows = select_fixed_eval_windows(windows, CFG.GREEDY_EVAL_MAX_WINDOWS)

    csv_file = open(report_csv_path, "w", newline="") if report_csv_path else None
    csv_writer = None
    if csv_file:
        csv_writer = csv.DictWriter(csv_file,
                                    fieldnames=["episode", "price_idx", "epsilon",
                                                "window_idx", "win_start", "win_end",
                                                "start_offset",
                                                "action", "equity", "bh_equity", "reward"])
        csv_writer.writeheader()

    def _episode_epsilon(ep_num_1_based: int) -> float:
        ep = int(ep_num_1_based)
        if ep <= 40:
            return float(np.interp(ep, [1, 40], [1.00, 0.10]))
        if ep <= 70:
            return float(np.interp(ep, [41, 70], [0.10, 0.05]))
        if ep <= 85:
            return float(np.interp(ep, [71, 85], [0.05, 0.02]))
        return 0.02

    train_every = max(1, int(CFG.TRAIN_REPLAY_EVERY_N_STEPS))
    agent.overwatch_enabled = False

    child_log(log_queue, f"[{symbol}] Replay reset policy: {describe_memory_reset_policy()}")

    for e in range(episodes):
        if should_reset_memory_after_episode(e):
            agent.reset_memory()
            child_log(log_queue, f"[{symbol}] Replay buffer reset at episode {e}")

        agent.override_count = 0
        if agent.use_episode_epsilon_schedule:
            agent.epsilon = _episode_epsilon(e + 1)

        win_idx = e % len(windows)
        win_start, win_end = windows[win_idx]
        df_win = df.iloc[win_start:win_end].reset_index(drop=True)

        max_offset_allowed = max(0, len(df_win) - (window_size + 2))
        start_offset = random.randint(0, min(int(CFG.TRAIN_RANDOM_START_MAX), int(max_offset_allowed)))

        env = build_env(df_win, mode="train", window_size=window_size, initial_cash=initial_cash)

        state = env.reset(start_offset=start_offset)
        tot_r = 0.0
        step_i = 0
        in_market_steps = 0

        while True:
            action = agent.act(state, env=env, symbol=symbol, use_overwatch=False)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            step_i += 1
            if env.position != 0:
                in_market_steps += 1

            if step_i % train_every == 0:
                agent.replay()

            if csv_writer:
                csv_writer.writerow({
                    "episode": e,
                    "price_idx": int(env.last_price_idx) if env.last_price_idx is not None else -1,
                    "epsilon": agent.epsilon,
                    "window_idx": win_idx,
                    "win_start": int(win_start),
                    "win_end": int(win_end),
                    "start_offset": int(start_offset),
                    "action": action,
                    "equity": env.equity,
                    "bh_equity": env.bh_equity,
                    "reward": reward,
                })
                

            state = next_state
            tot_r += reward
            if done:
                break

        reward_hist.append(float(tot_r))
        all_trades.extend(env.trades)

        opens_ct = sum(1 for t in env.trades if t[0] == "open_long")
        closes_ct = sum(1 for t in env.trades if t[0] == "close_long")
        tim_pct = (in_market_steps / max(1, step_i)) * 100.0

        opens_hist.append(int(opens_ct))
        closes_hist.append(int(closes_ct))
        time_in_market_pct_hist.append(float(tim_pct))

        if (e + 1) % int(CFG.GREEDY_EVAL_EVERY) == 0:
            eval_reward = evaluate_agent_on_fixed_windows(
                agent,
                df,
                fixed_eval_windows,
                window_size=window_size,
                initial_cash=initial_cash
            )
            eval_episodes.append(int(e + 1))
            eval_rewards.append(float(eval_reward))

        child_log(
            log_queue,
            f"[{symbol}] Episode {e + 1}/{episodes} "
            f"epsilon={agent.epsilon:.4f} "
            f"reward={tot_r:.4f} "
            f"time_in_mkt={tim_pct:.1f}% "
            f"opens={opens_ct} closes={closes_ct} "
            f"overrides={agent.override_count} "
            f"win={win_idx+1}/{len(windows)} offset={start_offset}"
        )

    if csv_file:
        csv_file.close()

    summarize_trades(all_trades, title=f"{symbol} Train Trades", log_queue=log_queue)
    gc.collect()

    rolling_med = pd.Series(reward_hist).rolling(
        window=max(1, int(CFG.REWARD_ROLLING_MEDIAN_WINDOW)),
        min_periods=1
    ).median().tolist()

    train_diag = {
        "reward_hist": [float(x) for x in reward_hist],
        "rolling_median_reward": [float(x) for x in rolling_med],
        "time_in_market_pct": [float(x) for x in time_in_market_pct_hist],
        "opens_per_episode": [int(x) for x in opens_hist],
        "closes_per_episode": [int(x) for x in closes_hist],
        "eval_episodes": [int(x) for x in eval_episodes],
        "eval_rewards": [float(x) for x in eval_rewards],
    }

    return reward_hist, all_trades, train_diag


# ─────────────────────────────────────────────────────────────────────────────
# Signal generation on test set
# Generates a signal path once, then we replay that path under multiple fills.
# ─────────────────────────────────────────────────────────────────────────────
def generate_test_signal_log(agent, df, *, window_size=20, initial_cash=100_000,
                             log_queue=None, symbol="", raw_feature_df=None):
    env = build_env(
        df,
        mode="test",
        window_size=window_size,
        initial_cash=initial_cash,
        raw_feature_df=raw_feature_df,
    )

    agent.overwatch_enabled = bool(CFG.OVERWATCH_ENABLED_TEST)
    agent.override_count = 0
    agent.overwatch_logs = []

    state = env.reset(start_offset=0)

    signal_rows = []

    while True:
        if CFG.OVERWATCH_ENABLED_TEST:
            action = agent.act(state, env=env, symbol=symbol, use_overwatch=True)
        else:
            action = agent.act_greedy(state, env=env)

        next_state, _, done, _ = env.step(action)

        idx = int(env.last_price_idx) if env.last_price_idx is not None else int(max(0, env.current_step - 1))
        dt = pd.Timestamp(env.dates[idx])

        signal_rows.append({
            "step": idx,
            "date": dt,
            "action": int(action),
        })

        state = next_state
        if done:
            break

    override_ct = agent.override_count
    override_pct = override_ct / len(signal_rows) * 100 if signal_rows else 0.0

    return {
        "signals": signal_rows,
        "override_count": override_ct,
        "override_pct": override_pct,
        "overwatch_logs": agent.overwatch_logs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Execution replay helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_fill_price(df, idx: int, fill_mode: str) -> float:
    if fill_mode == CFG.FILL_MODE_CLOSE_T:
        return float(df["adjusted_close"].iloc[idx])
    if fill_mode in (CFG.FILL_MODE_OPEN_T1, CFG.FILL_MODE_OPEN_T1_DISCOUNT_ONLY):
        return float(df["eval_open"].iloc[idx])
    if fill_mode == CFG.FILL_MODE_HL2_T1:
        return float((df["eval_high"].iloc[idx] + df["eval_low"].iloc[idx]) / 2.0)
    raise ValueError(f"Unknown fill mode: {fill_mode}")


def buy_hold_curve_from_test(df, first_step: int, last_step: int, initial_cash: float):
    px = df["adjusted_close"].to_numpy(dtype=np.float64)
    start_px = float(px[first_step])
    shares = (float(initial_cash) / start_px) if start_px > 0 else 0.0

    dates = []
    bh = []
    for idx in range(first_step, last_step + 1):
        dates.append(pd.Timestamp(df["Date"].iloc[idx]))
        bh.append(float(shares * px[idx]))
    return dates, bh


def replay_signals_with_fill_mode(df, signal_rows, fill_mode: str, *,
                                  initial_cash: float,
                                  slippage_rate: float,
                                  cooldown_days: int,
                                  min_hold_days: int,
                                  profitable_close_cooldown_days: int | None = None,
                                  losing_close_cooldown_days: int | None = None):
    if not signal_rows:
        return {
            "dates": [],
            "steps": [],
            "portfolio_values": [],
            "invested_history": [],
            "trades": [],
            "dropped_pending_orders": 0,
            "skipped_entry_filter_count": 0,
        }

    first_step = int(signal_rows[0]["step"])
    last_step = int(signal_rows[-1]["step"])

    action_by_step = {int(r["step"]): int(r["action"]) for r in signal_rows}

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

    def _can_open():
        return position == 0 and cooldown_remaining == 0

    def _bars_held(asof_idx: int) -> int:
        if position != 1 or entry_step is None:
            return 0
        return max(0, int(asof_idx) - int(entry_step))

    def _can_close(asof_idx: int):
        if position != 1:
            return False
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

        denom = (shares * entry_price)
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
        if fill_mode != CFG.FILL_MODE_OPEN_T1_DISCOUNT_ONLY:
            return True
        if fill_idx <= signal_idx:
            return False
        signal_close = float(closes[signal_idx])
        return float(fill_px) < signal_close

    for idx in range(first_step, last_step + 1):
        if pending_order is not None and int(pending_order["fill_idx"]) == idx:
            fill_px = get_fill_price(df, idx, fill_mode)
            if pending_order["type"] == "open_long":
                signal_idx = int(pending_order.get("signal_idx", idx - 1))
                if _can_open():
                    if _entry_filter_passes(signal_idx, idx, fill_px):
                        _open_long(idx, fill_px)
                    else:
                        skipped_entry_filter_count += 1
            elif pending_order["type"] == "close_long":
                if _can_close(idx):
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

        action = int(action_by_step.get(idx, ACTION_HOLD))

        if fill_mode == CFG.FILL_MODE_CLOSE_T:
            fill_px = get_fill_price(df, idx, fill_mode)
            if action == ACTION_LONG and _can_open():
                _open_long(idx, fill_px)
            elif action == ACTION_CLOSE and _can_close(idx):
                _close_long(idx, fill_px)
        else:
            if idx < last_step:
                if action == ACTION_LONG and _can_open():
                    pending_order = {"type": "open_long", "fill_idx": idx + 1, "signal_idx": idx}
                elif action == ACTION_CLOSE and _can_close(idx + 1):
                    pending_order = {"type": "close_long", "fill_idx": idx + 1, "signal_idx": idx}
            else:
                if action == ACTION_LONG and _can_open():
                    dropped_pending_orders += 1
                elif action == ACTION_CLOSE and _can_close(idx + 1):
                    pass

        if fill_mode == CFG.FILL_MODE_CLOSE_T:
            close_px = float(closes[idx])
            equity_after = cash + shares * close_px if position == 1 else cash
            out_equity[-1] = float(equity_after)
            out_invested[-1] = float(initial_cash if position == 1 else 0.0)

    if position == 1:
        final_idx = int(last_step)
        final_px = float(closes[final_idx])
        _close_long(final_idx, final_px)
        out_equity[-1] = float(cash)
        out_invested[-1] = 0.0

    return {
        "dates": out_dates,
        "steps": out_steps,
        "portfolio_values": out_equity,
        "invested_history": out_invested,
        "trades": trades,
        "dropped_pending_orders": int(dropped_pending_orders),
        "skipped_entry_filter_count": int(skipped_entry_filter_count),
    }


def build_test_result_with_fill_modes(agent, df, *, window_size=20, initial_cash=100_000,
                                      log_queue=None, symbol="", raw_feature_df=None):
    signal_pack = generate_test_signal_log(
        agent, df,
        window_size=window_size,
        initial_cash=initial_cash,
        log_queue=log_queue,
        symbol=symbol,
        raw_feature_df=raw_feature_df,
    )

    signals = signal_pack["signals"]
    if not signals:
        return {
            "dates": [],
            "steps": [],
            "buy_hold_values": [],
            "fills": {},
            "override_count": signal_pack["override_count"],
            "override_pct": signal_pack["override_pct"],
            "overwatch_logs": signal_pack["overwatch_logs"],
            "sleeve_initial_cash": float(initial_cash),
        }

    first_step = int(signals[0]["step"])
    last_step = int(signals[-1]["step"])
    bh_dates, bh_curve = buy_hold_curve_from_test(df, first_step, last_step, initial_cash)

    fills = {}
    for mode in CFG.ALL_FILL_MODES:
        fills[mode] = replay_signals_with_fill_mode(
            df=df,
            signal_rows=signals,
            fill_mode=mode,
            initial_cash=initial_cash,
            slippage_rate=float(CFG.SLIPPAGE_RATE),
            cooldown_days=int(CFG.COOLDOWN_DAYS),
            min_hold_days=int(CFG.MIN_HOLD_DAYS),
            profitable_close_cooldown_days=int(CFG.COOLDOWN_DAYS_AFTER_PROFITABLE_CLOSE),
            losing_close_cooldown_days=int(CFG.COOLDOWN_DAYS_AFTER_LOSING_CLOSE),
        )

    return {
        "dates": bh_dates,
        "steps": [int(r["step"]) for r in signals],
        "signals": signals,
        "buy_hold_values": bh_curve,
        "fills": fills,
        "override_count": signal_pack["override_count"],
        "override_pct": signal_pack["override_pct"],
        "overwatch_logs": signal_pack["overwatch_logs"],
        "sleeve_initial_cash": float(initial_cash),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-asset aggregation
# ─────────────────────────────────────────────────────────────────────────────
def build_real_multiasset_portfolio(results, total_initial_cash, *, fill_mode=None, use_buy_hold=False):
    if not results:
        raise ValueError("No results to aggregate.")

    n_assets = len(results)
    sleeve_capital = float(total_initial_cash) / float(n_assets)

    merged = None
    for i, r in enumerate(results):
        if use_buy_hold:
            dates = r["dates"]
            curve = r["buy_hold_values"]
            inv = [sleeve_capital] * len(curve)
        else:
            dates = r["fills"][fill_mode]["dates"]
            curve = r["fills"][fill_mode]["portfolio_values"]
            inv = r["fills"][fill_mode]["invested_history"]

        df_i = pd.DataFrame({
            "Date": pd.to_datetime(dates),
            f"eq_{i}": sleeve_capital * (np.array(curve, dtype=np.float64) / float(r["sleeve_initial_cash"])),
            f"inv_{i}": sleeve_capital * (np.array(inv, dtype=np.float64) / float(r["sleeve_initial_cash"])),
        }).groupby("Date", as_index=False).last()

        if merged is None:
            merged = df_i
        else:
            merged = merged.merge(df_i, on="Date", how="inner")

    if merged is None or merged.empty:
        raise RuntimeError("Could not build aggregated portfolio: no common test dates across assets.")

    eq_cols = [c for c in merged.columns if c.startswith("eq_")]
    inv_cols = [c for c in merged.columns if c.startswith("inv_")]

    agg_dates = merged["Date"].tolist()
    agg_equity = merged[eq_cols].sum(axis=1).astype(float).tolist()
    agg_inv = merged[inv_cols].sum(axis=1).astype(float).tolist()

    return agg_dates, agg_equity, agg_inv


# ─────────────────────────────────────────────────────────────────────────────
# Disk-based aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_trades_from_csv(path):
    if not os.path.exists(path):
        return []

    df = pd.read_csv(path)
    if df.empty:
        return []

    out = []
    for _, row in df.iterrows():
        pnl_pct = None if pd.isna(row.get("pnl_pct")) else float(row["pnl_pct"])
        time_in_trade = None if pd.isna(row.get("time")) else int(row["time"])
        out.append((
            str(row["type"]),
            int(row["step"]),
            float(row["price"]),
            pnl_pct,
            time_in_trade,
        ))
    return out


def load_symbol_result_from_disk(output_dir, symbol, sleeve_initial_cash):
    curve_path = os.path.join(output_dir, f"{symbol}_curves.csv")
    stats_path = os.path.join(output_dir, f"{symbol}_stats.csv")
    signals_path = os.path.join(output_dir, f"{symbol}_signals.csv")

    if not os.path.exists(curve_path):
        raise FileNotFoundError(f"Missing curve file for {symbol}: {curve_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing stats file for {symbol}: {stats_path}")
    if not os.path.exists(signals_path):
        raise FileNotFoundError(f"Missing signals file for {symbol}: {signals_path}")

    curve_df = pd.read_csv(curve_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    stats_df = pd.read_csv(stats_path)
    signals_df = pd.read_csv(signals_path)

    def _get_stat(mode, col, default=0):
        if stats_df.empty or ("fill_mode" not in stats_df.columns) or (col not in stats_df.columns):
            return default
        s = stats_df.loc[stats_df["fill_mode"] == mode, col]
        if s.empty:
            return default
        val = s.iloc[0]
        return default if pd.isna(val) else val

    override_count = int(_get_stat(CFG.FILL_MODE_CLOSE_T, "override_count", 0))

    result = {
        "symbol": str(symbol),
        "dates": curve_df["Date"].tolist(),
        "buy_hold_values": curve_df["buy_hold_value"].astype(float).tolist(),
        "override_count": override_count,
        "step_count": int(len(signals_df)),
        "sleeve_initial_cash": float(sleeve_initial_cash),
        "fills": {},
    }

    for mode in CFG.ALL_FILL_MODES:
        trades_path = os.path.join(output_dir, f"{symbol}_trades_{mode}.csv")

        result["fills"][mode] = {
            "dates": curve_df["Date"].tolist(),
            "portfolio_values": curve_df[f"portfolio_value_{mode}"].astype(float).tolist(),
            "invested_history": curve_df[f"invested_{mode}"].astype(float).tolist(),
            "trades": load_trades_from_csv(trades_path),
            "dropped_pending_orders": int(_get_stat(mode, "dropped_pending_orders", 0)),
            "skipped_entry_filter_count": int(_get_stat(mode, "skipped_entry_filter_count", 0)),
        }

    return result


def build_aggregated_overwatch_from_disk(output_dir, symbols):
    frames = []

    for sym in symbols:
        p = os.path.join(output_dir, f"{sym}_overwatch.csv")
        if not os.path.exists(p):
            continue

        try:
            df = pd.read_csv(p)
            if not df.empty:
                if "symbol" not in df.columns:
                    df.insert(0, "symbol", sym)
                frames.append(df)
        except Exception as e:
            print(f"Warning: failed reading overwatch file for {sym}: {e}")

    if frames:
        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────
def process_stock(args):
    train_csv, test_csv, agent_params, win, ep, ini, out_dir, log_q, worker_seed = args
    set_global_determinism(worker_seed)

    sym = os.path.splitext(os.path.basename(train_csv))[0]
    child_log(log_q, f"=== {sym} start | seed={worker_seed} ===")

    os.makedirs(CFG.MODEL_DIR, exist_ok=True)

    agent = DQNAgent(**agent_params)
    loaded_scaler = None
    loaded_state = None

    if bool(CFG.RESUME_FROM_CHECKPOINT):
        if checkpoint_exists(CFG.MODEL_DIR, sym):
            agent, loaded_scaler, loaded_state = load_agent_checkpoint(agent, sym, CFG.MODEL_DIR)
            if not scaler_feature_layout_matches(loaded_scaler, FEATURE_COLS):
                child_log(
                    log_q,
                    f"[{sym}] Checkpoint feature layout is stale for current FEATURE_COLS. "
                    "Ignoring saved model/scaler and training from scratch."
                )
                agent = DQNAgent(**agent_params)
                loaded_scaler = None
                loaded_state = None
            else:
                child_log(
                    log_q,
                    f"[{sym}] Loaded checkpoint | "
                    f"last_trained_date={loaded_state.get('last_trained_date')} "
                    f"epsilon={loaded_state.get('epsilon')} "
                    f"train_steps={loaded_state.get('train_step_counter')}"
                )
        else:
            child_log(log_q, f"[{sym}] No checkpoint found. Training from scratch.")

    # Train
    df_tr_raw = load_csv(train_csv, require_eval_prices=False)

    if loaded_scaler is not None and bool(CFG.REUSE_SAVED_SCALER):
        scaler = loaded_scaler
        child_log(log_q, f"[{sym}] Using saved scaler from checkpoint.")
    else:
        scaler = ZScoreScaler(FEATURE_COLS).fit(df_tr_raw)
        child_log(log_q, f"[{sym}] Fitted fresh scaler on current training data.")

    df_tr = scaler.transform(df_tr_raw)

    rewards, _, train_diag = train_agent_on_df(
        agent, df_tr,
        episodes=ep,
        window_size=win,
        initial_cash=ini,
        report_csv_path=os.path.join(out_dir, f"{sym}_train.csv"),
        log_queue=log_q,
        symbol=sym,
    )

    if bool(CFG.SAVE_MODEL_CHECKPOINTS):
        last_train_date = pd.to_datetime(df_tr_raw["Date"]).max().date()
        save_agent_checkpoint(
            agent=agent,
            scaler=scaler,
            symbol=sym,
            base_dir=CFG.MODEL_DIR,
            last_trained_date=last_train_date,
        )
        child_log(log_q, f"[{sym}] Saved checkpoint to {get_symbol_model_dir(CFG.MODEL_DIR, sym)}")

    plot_training_reward_only(
        reward_hist=rewards,
        save_path=os.path.join(out_dir, f"{sym}_reward.png"),
        symbol=sym
    )

    plot_training_diagnostics(
        train_diag=train_diag,
        save_path=os.path.join(out_dir, f"{sym}_training_diagnostics.png"),
        symbol=sym
    )

    pd.DataFrame({
        "episode": list(range(1, len(train_diag["reward_hist"]) + 1)),
        "reward": train_diag["reward_hist"],
        "rolling_median_reward": train_diag["rolling_median_reward"],
        "time_in_market_pct": train_diag["time_in_market_pct"],
        "opens": train_diag["opens_per_episode"],
        "closes": train_diag["closes_per_episode"],
    }).to_csv(os.path.join(out_dir, f"{sym}_training_diagnostics.csv"), index=False)

    pd.DataFrame({
        "eval_episode": train_diag["eval_episodes"],
        "eval_reward": train_diag["eval_rewards"],
    }).to_csv(os.path.join(out_dir, f"{sym}_greedy_eval.csv"), index=False)

    # Test
    df_te_raw = load_csv(test_csv, require_eval_prices=True)
    df_te = scaler.transform(df_te_raw)

    agent.epsilon = 0.0

    res = build_test_result_with_fill_modes(
        agent, df_te,
        window_size=win,
        initial_cash=ini,
        log_queue=log_q,
        symbol=sym,
        raw_feature_df=df_te_raw,
    )

    plot_symbol_test_results(
        symbol=sym,
        result=res,
        save_path=os.path.join(out_dir, f"{sym}_test.html")
    )

    pd.DataFrame(res["signals"]).to_csv(
        os.path.join(out_dir, f"{sym}_signals.csv"), index=False
    )

    pd.DataFrame(res.get("overwatch_logs", [])).to_csv(
        os.path.join(out_dir, f"{sym}_overwatch.csv"), index=False
    )

    stats_rows = []
    curve_df = pd.DataFrame({
        "Date": pd.to_datetime(res["dates"]),
        "buy_hold_value": res["buy_hold_values"],
    })

    for mode in CFG.ALL_FILL_MODES:
        fill = res["fills"][mode]

        summarize_trades(fill["trades"], title=f"{sym} Test Trades [{mode}]", log_queue=log_q)

        stats = compute_portfolio_stats(
            fill["portfolio_values"],
            fill["invested_history"],
            fill["trades"],
            ini,
            res["buy_hold_values"]
        )
        stats["symbol"] = sym
        stats["fill_mode"] = mode
        stats["override_count"] = res["override_count"]
        stats["override_pct"] = res["override_pct"]
        stats["dropped_pending_orders"] = fill["dropped_pending_orders"]
        stats["skipped_entry_filter_count"] = fill.get("skipped_entry_filter_count", 0)
        stats_rows.append(stats)

        pd.DataFrame(fill["trades"], columns=["type", "step", "price", "pnl_pct", "time"]).to_csv(
            os.path.join(out_dir, f"{sym}_trades_{mode}.csv"), index=False
        )

        curve_df[f"portfolio_value_{mode}"] = fill["portfolio_values"]
        curve_df[f"invested_{mode}"] = fill["invested_history"]

    pd.DataFrame(stats_rows).to_csv(os.path.join(out_dir, f"{sym}_stats.csv"), index=False)
    curve_df.to_csv(os.path.join(out_dir, f"{sym}_curves.csv"), index=False)

    del df_tr_raw, df_tr, df_te_raw, df_te, rewards, train_diag, res, curve_df, stats_rows
    del agent, scaler
    aggressive_worker_cleanup(log_q, sym)

    child_log(log_q, f"=== {sym} done ===")
    return {"symbol": sym}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def main_loop(agent_params, train_files, test_files, output_dir, *,
              window_size=20, episodes=10, initial_cash=100_000,
              log_queue=None):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(CFG.MODEL_DIR, exist_ok=True)

    if len(train_files) != len(test_files):
        raise ValueError(
            f"TRAIN_FILES and TEST_FILES length mismatch: "
            f"{len(train_files)} train files vs {len(test_files)} test files."
        )

    args = []
    base_seed = int(CFG.SEED)
    for idx, (tr, te) in enumerate(zip(train_files, test_files)):
        worker_seed = base_seed + (idx + 1) * 1000
        args.append((tr, te, agent_params, window_size, episodes, initial_cash,
                     output_dir, log_queue, worker_seed))

    worker_count = min(len(args), int(CFG.NUMBER_OF_POOLS))
    child_log(log_queue, f"Starting multiprocessing pool | workers={worker_count}")

    ctx = multiprocessing.get_context("spawn")
    finished_symbols = []

    with ctx.Pool(worker_count, maxtasksperchild=CFG.TASKS_PER_CHILD) as pool:
        for item in pool.imap_unordered(process_stock, args, chunksize=1):
            sym = item["symbol"]
            finished_symbols.append(sym)
            child_log(log_queue, f"Collected worker result {len(finished_symbols)}/{len(args)} | {sym}")

    child_log(log_queue, "Pool finished")

    if not finished_symbols:
        return

    finished_symbols = sorted(finished_symbols)
    child_log(log_queue, "Rebuilding aggregation inputs from per-symbol CSV outputs")

    results = []
    for sym in finished_symbols:
        results.append(
            load_symbol_result_from_disk(
                output_dir=output_dir,
                symbol=sym,
                sleeve_initial_cash=initial_cash
            )
        )

    child_log(log_queue, f"Loaded {len(results)} symbol result sets from disk")

    agg_dates_bh, agg_bh, _ = build_real_multiasset_portfolio(
        results=results,
        total_initial_cash=initial_cash,
        use_buy_hold=True
    )

    agg_curves = {"buy_hold": agg_bh}
    agg_invested = {}
    agg_stats_rows = []
    agg_stats_by_mode = {}

    agg_curve_df = pd.DataFrame({
        "Date": pd.to_datetime(agg_dates_bh),
        "buy_hold_value": agg_bh
    })

    for mode in CFG.ALL_FILL_MODES:
        child_log(log_queue, f"Aggregating mode: {mode}")

        agg_dates_mode, agg_equity_mode, agg_inv_mode = build_real_multiasset_portfolio(
            results=results,
            total_initial_cash=initial_cash,
            fill_mode=mode,
            use_buy_hold=False
        )

        if len(agg_dates_mode) != len(agg_dates_bh) or list(pd.to_datetime(agg_dates_mode)) != list(pd.to_datetime(agg_dates_bh)):
            bh_df = pd.DataFrame({"Date": pd.to_datetime(agg_dates_bh), "buy_hold": agg_bh})
            mode_df = pd.DataFrame({"Date": pd.to_datetime(agg_dates_mode), "eq": agg_equity_mode, "inv": agg_inv_mode})
            merged = bh_df.merge(mode_df, on="Date", how="inner")

            agg_dates_use = merged["Date"].tolist()
            agg_bh_use = merged["buy_hold"].astype(float).tolist()
            agg_eq_use = merged["eq"].astype(float).tolist()
            agg_inv_use = merged["inv"].astype(float).tolist()
        else:
            agg_dates_use = agg_dates_bh
            agg_bh_use = agg_bh
            agg_eq_use = agg_equity_mode
            agg_inv_use = agg_inv_mode

        agg_trades = [t for r in results for t in r["fills"][mode]["trades"]]

        total_overrides = sum(int(r.get("override_count", 0)) for r in results)
        total_steps = sum(int(r.get("step_count", 0)) for r in results)
        agg_override_pct = total_overrides / total_steps * 100 if total_steps else 0.0
        total_dropped = sum(int(r["fills"][mode]["dropped_pending_orders"]) for r in results)
        total_skipped_entries = sum(int(r["fills"][mode].get("skipped_entry_filter_count", 0)) for r in results)

        stats = compute_portfolio_stats(
            equity_curve=list(agg_eq_use),
            invested_curve=list(agg_inv_use),
            trades=agg_trades,
            initial_value=initial_cash,
            buy_hold_curve=list(agg_bh_use)
        )
        stats["fill_mode"] = mode
        stats["override_count"] = total_overrides
        stats["override_pct"] = agg_override_pct
        stats["dropped_pending_orders"] = total_dropped
        stats["skipped_entry_filter_count"] = total_skipped_entries

        agg_stats_by_mode[mode] = stats
        agg_stats_rows.append(stats)

        mode_df = pd.DataFrame({
            "Date": pd.to_datetime(agg_dates_use),
            f"portfolio_value_{mode}": agg_eq_use,
            f"invested_{mode}": agg_inv_use,
        })

        agg_curve_df = agg_curve_df.merge(mode_df, on="Date", how="inner")

        agg_curves[mode] = agg_curve_df[f"portfolio_value_{mode}"].astype(float).tolist()
        agg_invested[mode] = agg_curve_df[f"invested_{mode}"].astype(float).tolist()

    child_log(log_queue, "Writing aggregated_stats.csv")
    pd.DataFrame(agg_stats_rows).to_csv(
        os.path.join(output_dir, "aggregated_stats.csv"),
        index=False
    )

    child_log(log_queue, "Writing aggregated_curve.csv")
    agg_curve_df.to_csv(
        os.path.join(output_dir, "aggregated_curve.csv"),
        index=False
    )

    child_log(log_queue, "Writing aggregated_overwatch.csv")
    agg_overwatch_df = build_aggregated_overwatch_from_disk(output_dir, finished_symbols)
    agg_overwatch_df.to_csv(
        os.path.join(output_dir, "aggregated_overwatch.csv"),
        index=False
    )

    child_log(log_queue, "Writing aggregated_curve.html")
    plot_aggregated_results_multi(
        curves_dict=agg_curves,
        invested_dict=agg_invested,
        stats_by_mode=agg_stats_by_mode,
        dates=agg_curve_df["Date"].tolist(),
        save_path=os.path.join(output_dir, "aggregated_curve.html")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    set_global_determinism(CFG.SEED)

    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    os.makedirs(CFG.MODEL_DIR, exist_ok=True)

    mgr = multiprocessing.Manager()
    log_q = mgr.Queue()
    threading.Thread(target=logging_listener, args=(log_q,), daemon=True).start()
    reset_overwatch_chart_storage(log_q)

    print(f"Replay buffer reset policy: {describe_memory_reset_policy()}")
    print(f"Minimum hold days: {CFG.MIN_HOLD_DAYS}")
    print(f"Checkpoint directory: {CFG.MODEL_DIR}")
    print(f"Resume from checkpoint: {CFG.RESUME_FROM_CHECKPOINT}")
    print(f"Reuse saved scaler: {CFG.REUSE_SAVED_SCALER}")

    train_files = list(CFG.TRAIN_FILES)
    test_files = list(CFG.TEST_FILES)

    print(f"Workers: {min(len(train_files), CFG.NUMBER_OF_POOLS)}")

    output_dir = str(CFG.OUTPUT_DIR)
    window_size = int(CFG.WINDOW_SIZE)
    episodes = int(CFG.EPISODES)
    initial_cash = float(CFG.INITIAL_CASH)

    dummy_df = load_csv(train_files[0], require_eval_prices=False)
    tmp_env = TradingEnv(dummy_df, window_size=window_size, min_hold_days=int(CFG.MIN_HOLD_DAYS))
    state_size = tmp_env.observation_space.shape
    action_size = tmp_env.action_space.n

    agent_parameters = {
        "state_size": state_size,
        "action_size": action_size,
        "learning_rate": 0.0005,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.02,
        "batch_size": CFG.BATCHSIZE,
        "memory_size": 10_000,
    }

    start_time = time.time()
    main_loop(
        agent_parameters,
        train_files,
        test_files,
        output_dir,
        window_size=window_size,
        episodes=episodes,
        initial_cash=initial_cash,
        log_queue=log_q
    )

    elapsed = time.time() - start_time
    h, rem = divmod(elapsed, 3600)
    m, _ = divmod(rem, 60)
    print(f"Total runtime: {int(h)}h {int(m)}m")

    log_q.put("END")
    gc.collect()
