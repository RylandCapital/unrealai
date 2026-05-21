import argparse
import os
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv

try:
    import norgatedata
except ModuleNotFoundError:
    norgatedata = None

try:
    from unrealai.utils.feature_registry import FEATURE_COLS, FUNDAMENTAL_SYMBOLS
except ModuleNotFoundError:
    try:
        from utils.feature_registry import FEATURE_COLS, FUNDAMENTAL_SYMBOLS
    except ModuleNotFoundError:
        from feature_registry import FEATURE_COLS, FUNDAMENTAL_SYMBOLS

# =============================================================================
# CONFIG
# =============================================================================
START_DATE = pd.Timestamp("2010-01-01")
TEST_DAYS = 60
ALWAYS_RUN_TICKERS = ["QQQ", 'SPY', 'DIA', 'IWF']

# Pull a warmup window before the last saved date so rolling features remain valid
# when appending new rows.
FEATURE_WARMUP_CALENDAR_DAYS = 365 * 3

# Hard-break if any forward-filled internal raw series is unchanged too long.
MAX_STALE_ROWS = 60

APP_DIR = Path(__file__).resolve().parents[1]
load_dotenv(APP_DIR.parent / ".env")

WINDOWS_TRAIN_DIR = r"P:\10_CWP Trade Department\Ryland\unrealai\unrealai\traindata"
WINDOWS_TEST_DIR = r"P:\10_CWP Trade Department\Ryland\unrealai\unrealai\testdata"
WINDOWS_GRIP_ALLOCATION_PATH = r"P:\10_CWP Trade Department\_Matrix_\code_outputs\grip_momo\grip_allocation.xlsx"
WINDOWS_EDIP_ALLOCATION_PATH = r"P:\10_CWP Trade Department\Smitty\DSIP allocation.xlsx"


def _default_path(windows_path: str, spark_path: Path | None = None) -> Path | None:
    if os.name == "nt":
        return Path(windows_path)
    return spark_path


def _env_path(name: str, default: Path | str | None = None) -> Path | None:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return Path(default).expanduser() if default is not None else None
    return Path(os.path.expandvars(str(value).strip())).expanduser()


PADDING_SETTING = norgatedata.PaddingType.NONE if norgatedata is not None else None
PRICE_ADJUSTMENT = (
    norgatedata.StockPriceAdjustmentType.TOTALRETURN if norgatedata is not None else None
)
TIMESERIES_FORMAT = "pandas-dataframe"

TRAIN_DIR = _env_path("UNREALAI_TRAIN_DIR", _default_path(WINDOWS_TRAIN_DIR, APP_DIR / "traindata"))
TEST_DIR = _env_path("UNREALAI_TEST_DIR", _default_path(WINDOWS_TEST_DIR, APP_DIR / "testdata"))
GRIP_ALLOCATION_PATH = _env_path(
    "UNREALAI_GRIP_ALLOCATION_PATH",
    _default_path(WINDOWS_GRIP_ALLOCATION_PATH),
)
EDIP_ALLOCATION_PATH = _env_path(
    "UNREALAI_EDIP_ALLOCATION_PATH",
    _default_path(WINDOWS_EDIP_ALLOCATION_PATH),
)

SECTOR_PROXY_MAP = {
    "AMZN": "XLY",
    "CAT": "XLI",
    "INTU": "XLK",
    "INTC": "SMH",
    "LLY": "XLV",
    "LIN": "XLB",
    "MRK": "XLV",
    "STLD": "XLB",
    "NSC": "XLI",
    "AMAT": "SMH",
    "CME": "XLF",
    "TJX": "XLY",
    "IBM": "XLK",
    "TMUS": "XLC",
    "VZ": "XLC",
    "DUK": "XLU",
    "PLTR": "XLK",
    "GS": "XLF",
    "CSCO": "XLK",
    "TSLA": "XLY",
    "CVX": "XLE",
    "NFLX": "XLC",
    "MU": "SMH",
    "LRCX": "SMH",
    "AEM": "XLB",
    "KO": "XLP",
    "FDX": "XLI",
    "AMGN": "XLV",
    "MCD": "XLY",
    "MPC": "XLE",
    "META": "XLC",
    "SLB": "XLE",
    "GOOGL": "XLC",
    "COST": "XLP",
    "NVDA": "SMH",
    "PANW": "XLK",
    "ABBV": "XLV",
    "EXC": "XLU",
    "AVGO": "SMH",
    "RKLB": "XLI",
    "AXP": "XLF",
    "MDT": "XLV",
    "RTX": "XLI",
    "AMD": "SMH",
    "JPM": "XLF",
    "QQQ":"QQQ"
}


def _dedupe_preserve_order(values) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def parse_ticker_list(value) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set, pd.Series)):
        tokens = []
        for item in value:
            if pd.isna(item):
                continue
            tokens.extend(re.split(r"[\s,;]+", str(item)))
    else:
        tokens = re.split(r"[\s,;]+", str(value))

    tickers = []
    for token in tokens:
        ticker = token.strip().upper()
        if not ticker or ticker in {"NAN", "NONE", "CASH", "TICKER", "SYMBOL"}:
            continue
        tickers.append(ticker)

    return _dedupe_preserve_order(tickers)


def _csv_symbols(directory: Path) -> list[str]:
    if directory is None or not Path(directory).exists():
        return []
    return sorted({p.stem.upper() for p in Path(directory).glob("*.csv")})


def _read_ticker_file(path: Path) -> list[str]:
    return parse_ticker_list(path.read_text(encoding="utf-8"))


def _read_grip_tickers(path: Path) -> list[str]:
    df = pd.read_excel(path, header=1)
    if "Ticker" not in df.columns:
        raise ValueError(f"GRIP allocation file is missing a Ticker column: {path}")
    return parse_ticker_list(df["Ticker"].dropna().tolist())


def _read_edip_tickers(path: Path) -> list[str]:
    df = pd.read_excel(path)
    if "Unnamed: 1" in df.columns:
        series = df["Unnamed: 1"]
    elif len(df.columns) > 1:
        series = df.iloc[:, 1]
    else:
        series = df.iloc[:, 0]
    return parse_ticker_list(series.dropna().tolist())


def load_allocation_tickers() -> tuple[list[str], list[str]]:
    grip = []
    edip = []

    if GRIP_ALLOCATION_PATH and GRIP_ALLOCATION_PATH.exists():
        grip = _read_grip_tickers(GRIP_ALLOCATION_PATH)

    if EDIP_ALLOCATION_PATH and EDIP_ALLOCATION_PATH.exists():
        edip = _read_edip_tickers(EDIP_ALLOCATION_PATH)

    return grip, edip


def add_always_run_tickers(tickers: list[str]) -> list[str]:
    return _dedupe_preserve_order(tickers + ALWAYS_RUN_TICKERS)


def load_configured_tickers() -> list[str]:
    explicit_tickers = os.getenv("UNREALAI_TICKERS") or os.getenv("DATA_UPDATE_TICKERS")
    if explicit_tickers:
        return add_always_run_tickers(parse_ticker_list(explicit_tickers))

    ticker_file = _env_path("UNREALAI_TICKER_FILE")
    if ticker_file and ticker_file.exists():
        return add_always_run_tickers(_read_ticker_file(ticker_file))

    grip, edip = load_allocation_tickers()

    if grip or edip:
        mode = os.getenv("UNREALAI_TICKER_MODE", "xor").strip().lower()
        if mode in {"xor", "symmetric_difference", "symmetric-difference"}:
            return add_always_run_tickers(sorted(set(grip) ^ set(edip)))
        if mode in {"union", "all"}:
            return add_always_run_tickers(_dedupe_preserve_order(grip + edip))
        if mode in {"intersection", "common"}:
            return add_always_run_tickers(sorted(set(grip) & set(edip)))
        raise ValueError(
            "UNREALAI_TICKER_MODE must be one of xor, union, or intersection; "
            f"got {mode!r}"
        )

    # Spark/Linux fallback: update whatever symbols already exist locally.
    return add_always_run_tickers(sorted(set(_csv_symbols(TRAIN_DIR)) | set(_csv_symbols(TEST_DIR))))


TICKERS = load_configured_tickers()


# =============================================================================
# SMALL HELPERS
# =============================================================================
def get_sector_proxy(ticker: str) -> str:
    return SECTOR_PROXY_MAP.get(ticker, "SPY")


def safe_div(numerator, denominator):
    if isinstance(denominator, pd.Series):
        denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1.0 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def stochastic_kd(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, smooth: int = 3):
    lowest_low = low.rolling(window).min()
    highest_high = high.rolling(window).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (close - lowest_low) / denom
    d = k.rolling(smooth).mean()
    return k, d


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    line = ema_fast - ema_slow
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def ppo(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    ppo_line = 100.0 * (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)
    ppo_signal = ema(ppo_line, signal)
    ppo_hist = ppo_line - ppo_signal
    return ppo_line, ppo_signal, ppo_hist


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume.fillna(0.0)).cumsum()


def accumulation_distribution_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    denom = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / denom
    mfv = mfm.fillna(0.0) * volume.fillna(0.0)
    return mfv.cumsum()


def rolling_beta(y: pd.Series, x: pd.Series, window: int = 63) -> pd.Series:
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var.replace(0, np.nan)


def rolling_momentum_score(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _score(arr):
        arr = np.asarray(arr, dtype=np.float64)
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
            return np.nan

        log_ts = np.log(arr)
        y_mean = log_ts.mean()
        y_centered = log_ts - y_mean
        x_centered = x - x_mean

        slope = (x_centered * y_centered).sum() / x_var
        denom = np.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
        r_value = (x_centered * y_centered).sum() / denom if denom > 0 else 0.0

        annualized_slope = (np.exp(slope) ** 252 - 1) * 100
        score = annualized_slope * (r_value ** 2)
        return score

    return series.rolling(window).apply(_score, raw=True)


def momentum_score(ts, days):
    arr = np.asarray(ts, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2 or np.any(arr <= 0):
        return np.nan

    x = np.arange(len(arr), dtype=np.float64)
    log_ts = np.log(arr)
    x_centered = x - x.mean()
    y_centered = log_ts - log_ts.mean()
    x_var = (x_centered ** 2).sum()
    if x_var == 0:
        return np.nan

    slope = (x_centered * y_centered).sum() / x_var
    denom = np.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
    r_value = (x_centered * y_centered).sum() / denom if denom > 0 else 0.0
    annualized_slope = (np.exp(slope) ** 252 - 1) * 100
    score = annualized_slope * (r_value ** 2)
    return score


# =============================================================================
# STALE-COLUMN DETECTOR
# =============================================================================
def detect_stale_columns(df: pd.DataFrame, columns, max_same_rows: int = MAX_STALE_ROWS):
    """
    Hard-break if any specified column stays exactly unchanged for more than
    max_same_rows consecutive rows. Intended for raw slow-moving internal series
    after forward-fill, not for all engineered features.
    """
    if df.empty:
        return

    dates = pd.to_datetime(df["Date"]).reset_index(drop=True)
    offenders = []

    for col in columns:
        if col not in df.columns:
            continue

        s = df[col].reset_index(drop=True)

        if s.isna().all():
            offenders.append({
                "column": col,
                "run_length": len(s),
                "start_date": dates.iloc[0].date() if len(dates) else "N/A",
                "end_date": dates.iloc[-1].date() if len(dates) else "N/A",
                "value": "ALL_NA",
            })
            continue

        run_start = 0
        prev_val = s.iloc[0]

        for i in range(1, len(s)):
            cur_val = s.iloc[i]
            same = pd.notna(cur_val) and pd.notna(prev_val) and (cur_val == prev_val)

            if same:
                run_len = i - run_start + 1
                if run_len > max_same_rows:
                    offenders.append({
                        "column": col,
                        "run_length": run_len,
                        "start_date": dates.iloc[run_start].date(),
                        "end_date": dates.iloc[i].date(),
                        "value": cur_val,
                    })
                    break
            else:
                run_start = i
                prev_val = cur_val

    if offenders:
        msg_lines = [
            f"Stale data detected: one or more internal raw series stayed unchanged for more than {max_same_rows} rows."
        ]
        for off in offenders:
            msg_lines.append(
                f" - {off['column']}: {off['run_length']} rows unchanged "
                f"from {off['start_date']} to {off['end_date']} | value={off['value']}"
            )
        raise ValueError("\n".join(msg_lines))


# =============================================================================
# NORGATE FETCH
# =============================================================================
def require_norgatedata():
    if norgatedata is None:
        raise RuntimeError(
            "norgatedata is not installed in this Python environment. "
            "data_update.py needs Norgate for prices and market-internal symbols. "
            "On Spark/Linux, install and configure Norgate's Python package if available, "
            "or run the updater on the Windows/Norgate machine and sync the generated CSVs."
        )


def fetch_norgate_ohlcv(symbol: str, start_date: pd.Timestamp) -> pd.DataFrame:
    require_norgatedata()
    ts = norgatedata.price_timeseries(
        symbol,
        stock_price_adjustment_setting=PRICE_ADJUSTMENT,
        padding_setting=PADDING_SETTING,
        start_date=start_date,
        timeseriesformat=TIMESERIES_FORMAT,
    )

    df = pd.DataFrame.from_dict(ts).reset_index()
    df["Date"] = pd.to_datetime(df["Date"])

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0.0 if col == "Volume" else np.nan

    df["adjusted_close"] = df["Close"].astype(float).copy()
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def fetch_norgate_close(symbol: str, start_date: pd.Timestamp) -> pd.DataFrame:
    require_norgatedata()
    ts = norgatedata.price_timeseries(
        symbol,
        stock_price_adjustment_setting=PRICE_ADJUSTMENT,
        padding_setting=PADDING_SETTING,
        start_date=start_date,
        timeseriesformat=TIMESERIES_FORMAT,
    )

    df = pd.DataFrame.from_dict(ts).reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Close"]].rename(columns={"Close": f"{symbol}_close"})
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# =============================================================================
# MARKET INTERNALS
# =============================================================================
def build_market_internal_frame(start_date: pd.Timestamp) -> pd.DataFrame:
    tmpdfs = [fetch_norgate_close(sym, start_date=start_date) for sym in FUNDAMENTAL_SYMBOLS]

    df_index = tmpdfs[0]
    for tmp in tmpdfs[1:]:
        df_index = pd.merge(df_index, tmp, on="Date", how="outer")

    df_index = df_index.sort_values("Date").reset_index(drop=True)

    raw_cols = [c for c in df_index.columns if c != "Date"]

    # Carry forward slower macro / internal series.
    df_index[raw_cols] = df_index[raw_cols].ffill()

    # Hard-break if a raw internal level stays unchanged too long.
    detect_stale_columns(df_index, raw_cols, max_same_rows=MAX_STALE_ROWS)

    for sym in FUNDAMENTAL_SYMBOLS:
        col = f"{sym}_close"
        df_index[f"{sym}_chg_5"] = df_index[col] - df_index[col].shift(5)
        df_index[f"{sym}_chg_21"] = df_index[col] - df_index[col].shift(21)
        df_index[f"{sym}_z_63"] = rolling_zscore(df_index[col], 63)
        df_index[f"{sym}_z_126"] = rolling_zscore(df_index[col], 126)

    return df_index


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def add_price_features(df: pd.DataFrame, prefix: str = "", include_legacy: bool = False) -> pd.DataFrame:
    base = df.copy()

    if prefix == "":
        feat = base.copy()
    else:
        feat = pd.DataFrame({
            "Date": base["Date"],
            f"{prefix}Open": base["Open"],
            f"{prefix}High": base["High"],
            f"{prefix}Low": base["Low"],
            f"{prefix}Close": base["Close"],
            f"{prefix}Volume": base["Volume"],
            f"{prefix}adjusted_close": base["adjusted_close"],
        })

    close = base["adjusted_close"].astype(float)
    high = base["High"].astype(float)
    low = base["Low"].astype(float)
    open_ = base["Open"].astype(float)
    volume = base["Volume"].astype(float).fillna(0.0)

    log_ret_1 = np.log(close / close.shift(1))
    ret_1 = close.pct_change(1)
    ret_5 = close.pct_change(5)
    ret_10 = close.pct_change(10)
    ret_21 = close.pct_change(21)
    ret_63 = close.pct_change(63)
    ret_126 = close.pct_change(126)

    vol_10 = log_ret_1.rolling(10).std(ddof=0) * np.sqrt(252)
    vol_21 = log_ret_1.rolling(21).std(ddof=0) * np.sqrt(252)
    vol_63 = log_ret_1.rolling(63).std(ddof=0) * np.sqrt(252)

    tr_pct = safe_div(high - low, close.shift(1))
    atr_14 = atr(high, low, close, window=14)
    atr_14_pct = safe_div(atr_14, close)

    gap_pct = safe_div(open_ - close.shift(1), close.shift(1))
    close_loc_in_range = safe_div(close - low, (high - low))

    sma_21 = close.rolling(21).mean()
    sma_55 = close.rolling(55).mean()
    sma_200 = close.rolling(200).mean()

    dist_sma_21 = safe_div(close, sma_21) - 1.0
    dist_sma_55 = safe_div(close, sma_55) - 1.0
    dist_sma_200 = safe_div(close, sma_200) - 1.0

    sma_21_slope_10 = sma_21.pct_change(10)
    sma_55_slope_10 = sma_55.pct_change(10)
    sma_200_slope_20 = sma_200.pct_change(20)

    ma_spread_21_55 = safe_div(sma_21, sma_55) - 1.0
    ma_spread_21_200 = safe_div(sma_21, sma_200) - 1.0
    ma_spread_55_200 = safe_div(sma_55, sma_200) - 1.0

    rsi_14_raw = rsi(close, 14)
    rsi_21_raw = rsi(close, 21)
    rsi_21_delta = rsi_21_raw.diff(5)

    stoch_k_14, stoch_d_14 = stochastic_kd(high, low, close, window=14, smooth=3)
    macd_line, macd_signal, macd_hist = macd(close, fast=12, slow=26, signal=9)
    ppo_line, ppo_signal, ppo_hist = ppo(close, fast=12, slow=26, signal=9)

    roll_high_20 = close.rolling(20).max()
    roll_low_20 = close.rolling(20).min()
    roll_high_60 = close.rolling(60).max()
    roll_low_60 = close.rolling(60).min()
    roll_high_252 = close.rolling(252).max()
    roll_low_252 = close.rolling(252).min()

    range_pos_20 = safe_div(close - roll_low_20, roll_high_20 - roll_low_20)
    range_pos_60 = safe_div(close - roll_low_60, roll_high_60 - roll_low_60)
    range_pos_252 = safe_div(close - roll_low_252, roll_high_252 - roll_low_252)

    drawdown_63 = safe_div(close, close.rolling(63).max()) - 1.0
    drawdown_252 = safe_div(close, close.rolling(252).max()) - 1.0

    bb_mid_20 = close.rolling(20).mean()
    bb_std_20 = close.rolling(20).std(ddof=0)
    bb_upper_20 = bb_mid_20 + 2.0 * bb_std_20
    bb_lower_20 = bb_mid_20 - 2.0 * bb_std_20
    bb_width_20 = safe_div(bb_upper_20 - bb_lower_20, bb_mid_20)
    bb_pos_20 = safe_div(close - bb_lower_20, bb_upper_20 - bb_lower_20)
    atr_regime_63 = safe_div(atr_14_pct, atr_14_pct.rolling(63).mean())
    range_compression_20 = safe_div(roll_high_20 - roll_low_20, close)

    true_range = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    nr7_flag = (true_range <= true_range.rolling(7).min()).astype(int)

    prev_high_20 = close.shift(1).rolling(20).max()
    prev_high_60 = close.shift(1).rolling(60).max()
    prev_high_252 = close.shift(1).rolling(252).max()

    prev_low_20 = close.shift(1).rolling(20).min()
    prev_low_60 = close.shift(1).rolling(60).min()

    dist_prev_high_20 = safe_div(close, prev_high_20) - 1.0
    dist_prev_high_60 = safe_div(close, prev_high_60) - 1.0
    dist_prev_high_252 = safe_div(close, prev_high_252) - 1.0

    dist_prev_low_20 = safe_div(close, prev_low_20) - 1.0
    dist_prev_low_60 = safe_div(close, prev_low_60) - 1.0

    new_high_20 = (close > prev_high_20).astype(int)
    new_high_60 = (close > prev_high_60).astype(int)
    new_high_252 = (close > prev_high_252).astype(int)

    new_low_20 = (close < prev_low_20).astype(int)
    new_low_60 = (close < prev_low_60).astype(int)

    volume_rel_20 = safe_div(volume, volume.rolling(20).mean())
    volume_rel_63 = safe_div(volume, volume.rolling(63).mean())

    dollar_volume = close * volume
    dollar_volume_rel_20 = safe_div(dollar_volume, dollar_volume.rolling(20).mean())

    up_vol_20 = volume.where(log_ret_1 > 0, 0.0).rolling(20).sum()
    down_vol_20 = volume.where(log_ret_1 < 0, 0.0).rolling(20).sum()
    up_down_vol_ratio_20 = safe_div(up_vol_20, down_vol_20)

    obv = on_balance_volume(close, volume)
    obv_z_63 = rolling_zscore(obv, 63)

    adl = accumulation_distribution_line(high, low, close, volume)
    adl_z_63 = rolling_zscore(adl, 63)

    mom_score_21 = rolling_momentum_score(close, 21)
    mom_score_63 = rolling_momentum_score(close, 63)
    mom_score_126 = rolling_momentum_score(close, 126)

    def add_col(name, series):
        feat[f"{prefix}{name}"] = series

    add_col("log_ret_1", log_ret_1)
    add_col("ret_1", ret_1)
    add_col("ret_5", ret_5)
    add_col("ret_10", ret_10)
    add_col("ret_21", ret_21)
    add_col("ret_63", ret_63)
    add_col("ret_126", ret_126)

    add_col("vol_10", vol_10)
    add_col("vol_21", vol_21)
    add_col("vol_63", vol_63)

    add_col("tr_pct", tr_pct)
    add_col("atr_14_pct", atr_14_pct)
    add_col("gap_pct", gap_pct)
    add_col("close_loc_in_range", close_loc_in_range)

    add_col("sma_21", sma_21)
    add_col("sma_55", sma_55)
    add_col("sma_200", sma_200)

    add_col("dist_sma_21", dist_sma_21)
    add_col("dist_sma_55", dist_sma_55)
    add_col("dist_sma_200", dist_sma_200)

    add_col("sma_21_slope_10", sma_21_slope_10)
    add_col("sma_55_slope_10", sma_55_slope_10)
    add_col("sma_200_slope_20", sma_200_slope_20)

    add_col("ma_spread_21_55", ma_spread_21_55)
    add_col("ma_spread_21_200", ma_spread_21_200)
    add_col("ma_spread_55_200", ma_spread_55_200)

    add_col("rsi_14_raw", rsi_14_raw)
    add_col("rsi_21_raw", rsi_21_raw)
    add_col("rsi_21_delta", rsi_21_delta)

    add_col("stoch_k_14", stoch_k_14)
    add_col("stoch_d_14", stoch_d_14)

    add_col("macd_line", macd_line)
    add_col("macd_signal", macd_signal)
    add_col("macd_hist", macd_hist)

    add_col("ppo_line", ppo_line)
    add_col("ppo_signal", ppo_signal)
    add_col("ppo_hist", ppo_hist)

    add_col("range_pos_20", range_pos_20)
    add_col("range_pos_60", range_pos_60)
    add_col("range_pos_252", range_pos_252)

    add_col("drawdown_63", drawdown_63)
    add_col("drawdown_252", drawdown_252)
    add_col("bb_width_20", bb_width_20)
    add_col("bb_pos_20", bb_pos_20)
    add_col("atr_regime_63", atr_regime_63)
    add_col("range_compression_20", range_compression_20)
    add_col("nr7_flag", nr7_flag)

    add_col("dist_prev_high_20", dist_prev_high_20)
    add_col("dist_prev_high_60", dist_prev_high_60)
    add_col("dist_prev_high_252", dist_prev_high_252)

    add_col("dist_prev_low_20", dist_prev_low_20)
    add_col("dist_prev_low_60", dist_prev_low_60)

    add_col("new_high_20", new_high_20)
    add_col("new_high_60", new_high_60)
    add_col("new_high_252", new_high_252)

    add_col("new_low_20", new_low_20)
    add_col("new_low_60", new_low_60)

    add_col("volume_rel_20", volume_rel_20)
    add_col("volume_rel_63", volume_rel_63)
    add_col("dollar_volume", dollar_volume)
    add_col("dollar_volume_rel_20", dollar_volume_rel_20)
    add_col("up_down_vol_ratio_20", up_down_vol_ratio_20)

    add_col("obv", obv)
    add_col("obv_z_63", obv_z_63)
    add_col("adl", adl)
    add_col("adl_z_63", adl_z_63)

    add_col("mom_score_21", mom_score_21)
    add_col("mom_score_63", mom_score_63)
    add_col("mom_score_126", mom_score_126)

    if include_legacy:
        above_21_sma = (close > sma_21).astype(int)
        above_55_sma = (close > sma_55).astype(int)
        above_200_sma = (close > sma_200).astype(int)
        rsi_21_bucket = np.where(rsi_21_raw > 70, 1, np.where(rsi_21_raw < 30, -1, 0))

        if prefix == "":
            feat["21_day_sma"] = sma_21
            feat["55_day_sma"] = sma_55
            feat["200_day_sma"] = sma_200
            feat["above_21_sma"] = above_21_sma
            feat["above_55_sma"] = above_55_sma
            feat["above_200_sma"] = above_200_sma
            feat["21day_rsi"] = rsi_21_bucket
        elif prefix == "spy_":
            feat["spy_21_day_sma"] = sma_21
            feat["spy_55_day_sma"] = sma_55
            feat["spy_200_day_sma"] = sma_200
            feat["spy_above_21_sma"] = above_21_sma
            feat["spy_above_55_sma"] = above_55_sma
            feat["spy_above_200_sma"] = above_200_sma
            feat["spy_21day_rsi"] = rsi_21_bucket

            feat["above_21_sma_spy"] = above_21_sma
            feat["above_55_sma_spy"] = above_55_sma
            feat["above_200_sma_spy"] = above_200_sma
            feat["21day_rsi_spy"] = rsi_21_bucket

    return feat


# =============================================================================
# MASTER DATASET BUILD
# =============================================================================
def get_daily_equity(ticker: str, start_date: pd.Timestamp) -> pd.DataFrame:
    sector_symbol = get_sector_proxy(ticker)
    sector_is_same = (sector_symbol == ticker)
    spy_is_same = (ticker == "SPY")

    df_index = build_market_internal_frame(start_date=start_date)

    df_stock_raw = fetch_norgate_ohlcv(ticker, start_date=start_date)
    df_stock = add_price_features(df_stock_raw, prefix="", include_legacy=True)

    df_spy_raw = fetch_norgate_ohlcv("SPY", start_date=start_date)
    df_spy = add_price_features(df_spy_raw, prefix="spy_", include_legacy=True)

    df_sector_raw = fetch_norgate_ohlcv(sector_symbol, start_date=start_date)
    df_sector = add_price_features(df_sector_raw, prefix="sector_", include_legacy=False)

    # Keep true trading calendar from stock/SPY/sector.
    merged_df = pd.merge(df_stock, df_spy, on="Date", how="inner")
    merged_df = pd.merge(merged_df, df_sector, on="Date", how="inner")

    # Internal/macro series are contextual only. Join onto the stock calendar.
    df_merged = pd.merge(merged_df, df_index, on="Date", how="left")
    df_merged = df_merged.sort_values("Date").reset_index(drop=True)

    # Forward-fill internal columns again on the stock calendar for safety.
    internal_cols = [c for c in df_index.columns if c != "Date"]
    df_merged[internal_cols] = df_merged[internal_cols].ffill()

    # -------------------------------------------------------------------------
    # Relative strength vs SPY
    # -------------------------------------------------------------------------
    if spy_is_same:
        df_merged["rel_ret_5_vs_spy"] = 0.0
        df_merged["rel_ret_10_vs_spy"] = 0.0
        df_merged["rel_ret_21_vs_spy"] = 0.0
        df_merged["rel_ret_63_vs_spy"] = 0.0
        df_merged["rel_ret_126_vs_spy"] = 0.0

        df_merged["price_to_spy_ratio"] = 1.0
        df_merged["price_to_spy_ratio_ret_21"] = 0.0
        df_merged["price_to_spy_ratio_ret_63"] = 0.0
        df_merged["price_to_spy_ratio_z_63"] = 0.0
        df_merged["price_to_spy_ratio_dist_sma_63"] = 0.0

        df_merged["beta_to_spy_63"] = 1.0
        df_merged["resid_ret_1_vs_spy"] = 0.0
        df_merged["resid_mom_21_vs_spy"] = 0.0
        df_merged["resid_mom_63_vs_spy"] = 0.0
        df_merged["idio_vol_63_vs_spy"] = 0.0
    else:
        df_merged["rel_ret_5_vs_spy"] = df_merged["ret_5"] - df_merged["spy_ret_5"]
        df_merged["rel_ret_10_vs_spy"] = df_merged["ret_10"] - df_merged["spy_ret_10"]
        df_merged["rel_ret_21_vs_spy"] = df_merged["ret_21"] - df_merged["spy_ret_21"]
        df_merged["rel_ret_63_vs_spy"] = df_merged["ret_63"] - df_merged["spy_ret_63"]
        df_merged["rel_ret_126_vs_spy"] = df_merged["ret_126"] - df_merged["spy_ret_126"]

        price_to_spy_ratio = safe_div(df_merged["adjusted_close"], df_merged["spy_adjusted_close"])
        df_merged["price_to_spy_ratio"] = price_to_spy_ratio
        df_merged["price_to_spy_ratio_ret_21"] = price_to_spy_ratio.pct_change(21)
        df_merged["price_to_spy_ratio_ret_63"] = price_to_spy_ratio.pct_change(63)
        df_merged["price_to_spy_ratio_z_63"] = rolling_zscore(price_to_spy_ratio, 63)
        df_merged["price_to_spy_ratio_dist_sma_63"] = safe_div(
            price_to_spy_ratio, price_to_spy_ratio.rolling(63).mean()
        ) - 1.0

        beta_to_spy_63 = rolling_beta(df_merged["log_ret_1"], df_merged["spy_log_ret_1"], 63).shift(1)
        df_merged["beta_to_spy_63"] = beta_to_spy_63
        df_merged["resid_ret_1_vs_spy"] = df_merged["log_ret_1"] - beta_to_spy_63 * df_merged["spy_log_ret_1"]
        df_merged["resid_mom_21_vs_spy"] = df_merged["resid_ret_1_vs_spy"].rolling(21).sum()
        df_merged["resid_mom_63_vs_spy"] = df_merged["resid_ret_1_vs_spy"].rolling(63).sum()
        df_merged["idio_vol_63_vs_spy"] = df_merged["resid_ret_1_vs_spy"].rolling(63).std(ddof=0) * np.sqrt(252)

    # -------------------------------------------------------------------------
    # Relative strength vs sector
    # -------------------------------------------------------------------------
    df_merged["rel_ret_5_vs_sector"] = df_merged["ret_5"] - df_merged["sector_ret_5"]
    df_merged["rel_ret_10_vs_sector"] = df_merged["ret_10"] - df_merged["sector_ret_10"]
    df_merged["rel_ret_21_vs_sector"] = df_merged["ret_21"] - df_merged["sector_ret_21"]
    df_merged["rel_ret_63_vs_sector"] = df_merged["ret_63"] - df_merged["sector_ret_63"]
    df_merged["rel_ret_126_vs_sector"] = df_merged["ret_126"] - df_merged["sector_ret_126"]

    if sector_is_same:
        df_merged["price_to_sector_ratio"] = 1.0
        df_merged["price_to_sector_ratio_ret_21"] = 0.0
        df_merged["price_to_sector_ratio_ret_63"] = 0.0
        df_merged["price_to_sector_ratio_z_63"] = 0.0
        df_merged["price_to_sector_ratio_dist_sma_63"] = 0.0
        df_merged["beta_to_sector_63"] = 1.0
        df_merged["resid_ret_1_vs_sector"] = 0.0
        df_merged["resid_mom_21_vs_sector"] = 0.0
        df_merged["resid_mom_63_vs_sector"] = 0.0
        df_merged["idio_vol_63_vs_sector"] = 0.0
    else:
        price_to_sector_ratio = safe_div(df_merged["adjusted_close"], df_merged["sector_adjusted_close"])
        df_merged["price_to_sector_ratio"] = price_to_sector_ratio
        df_merged["price_to_sector_ratio_ret_21"] = price_to_sector_ratio.pct_change(21)
        df_merged["price_to_sector_ratio_ret_63"] = price_to_sector_ratio.pct_change(63)
        df_merged["price_to_sector_ratio_z_63"] = rolling_zscore(price_to_sector_ratio, 63)
        df_merged["price_to_sector_ratio_dist_sma_63"] = safe_div(
            price_to_sector_ratio, price_to_sector_ratio.rolling(63).mean()
        ) - 1.0

        beta_to_sector_63 = rolling_beta(df_merged["log_ret_1"], df_merged["sector_log_ret_1"], 63).shift(1)
        df_merged["beta_to_sector_63"] = beta_to_sector_63
        df_merged["resid_ret_1_vs_sector"] = df_merged["log_ret_1"] - beta_to_sector_63 * df_merged["sector_log_ret_1"]
        df_merged["resid_mom_21_vs_sector"] = df_merged["resid_ret_1_vs_sector"].rolling(21).sum()
        df_merged["resid_mom_63_vs_sector"] = df_merged["resid_ret_1_vs_sector"].rolling(63).sum()
        df_merged["idio_vol_63_vs_sector"] = df_merged["resid_ret_1_vs_sector"].rolling(63).std(ddof=0) * np.sqrt(252)

    # -------------------------------------------------------------------------
    # Composite context
    # -------------------------------------------------------------------------
    df_merged["trend_strength_21_200"] = df_merged["dist_sma_21"] - df_merged["dist_sma_200"]
    df_merged["vol_regime_ratio"] = safe_div(df_merged["vol_21"], df_merged["vol_63"])
    df_merged["atr_to_vol_ratio"] = safe_div(df_merged["atr_14_pct"], df_merged["vol_21"])
    df_merged["stock_vs_market_drawdown_252"] = df_merged["drawdown_252"] - df_merged["spy_drawdown_252"]
    df_merged["stock_vs_sector_drawdown_252"] = df_merged["drawdown_252"] - df_merged["sector_drawdown_252"]

    # -------------------------------------------------------------------------
    # Final cleanup
    # -------------------------------------------------------------------------
    df_merged = df_merged.replace([np.inf, -np.inf], np.nan)

    rows_before = len(df_merged)
    missing_feature_cols = [col for col in FEATURE_COLS if col not in df_merged.columns]
    all_nan_cols = df_merged.columns[df_merged.isna().all()].tolist()

    if missing_feature_cols:
        raise ValueError(
            f"[{ticker}] Full build is missing registry feature columns: {missing_feature_cols}"
        )

    if all_nan_cols:
        raise ValueError(
            f"[{ticker}] Full build has all-NaN columns before dropna: {all_nan_cols}"
        )

    valid_rows = df_merged.dropna()

    if valid_rows.empty:
        non_na_counts = df_merged.notna().sum().sort_values()
        raise ValueError(
            f"[{ticker}] get_daily_equity() produced 0 rows after dropna.\n"
            f"Rows before dropna: {rows_before}\n"
            f"Lowest non-NA counts:\n{non_na_counts.head(25)}"
        )

    df_merged = valid_rows.reset_index(drop=True)
    return df_merged


# =============================================================================
# FILE / SPLIT HELPERS
# =============================================================================
def train_path(ticker: str) -> Path:
    return TRAIN_DIR / f"{ticker}.csv"


def test_path(ticker: str) -> Path:
    return TEST_DIR / f"{ticker}.csv"


def load_existing_feature_history(ticker: str) -> pd.DataFrame:
    parts = []

    tr_path = train_path(ticker)
    te_path = test_path(ticker)

    if os.path.exists(tr_path):
        parts.append(pd.read_csv(tr_path, parse_dates=["Date"]))

    if os.path.exists(te_path):
        parts.append(pd.read_csv(te_path, parse_dates=["Date"]))

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, axis=0, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return df


def split_train_test(df: pd.DataFrame, test_days: int = TEST_DAYS):
    if len(df) <= test_days:
        raise ValueError(f"Not enough rows to split. Need > {test_days}, got {len(df)}")
    return df.iloc[:-test_days].copy(), df.iloc[-test_days:].copy()


def combine_existing_with_new(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df.empty:
        out = new_df.copy()
    else:
        last_existing_date = pd.to_datetime(existing_df["Date"]).max()
        append_df = new_df[new_df["Date"] > last_existing_date].copy()
        out = pd.concat([existing_df, append_df], axis=0, ignore_index=True)

    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return out


# =============================================================================
# UPDATE LOGIC
# =============================================================================
def update_one_ticker(ticker: str):
    existing_df = load_existing_feature_history(ticker)

    if existing_df.empty:
        print(f"\n[{ticker}] No existing train/test found. Full rebuild from {START_DATE.date()}.")
        combined_df = get_daily_equity(ticker, start_date=START_DATE)
        added_rows = len(combined_df)
    else:
        last_old_date = pd.to_datetime(existing_df["Date"]).max().normalize()
        incremental_start = max(
            START_DATE,
            last_old_date - pd.Timedelta(days=FEATURE_WARMUP_CALENDAR_DAYS)
        )

        print(
            f"\n[{ticker}] Existing history through {last_old_date.date()} | "
            f"refresh window starts {incremental_start.date()}"
        )

        refreshed_recent_df = get_daily_equity(ticker, start_date=incremental_start)

        # If schema changed, do a full rebuild instead of mixing mismatched old/new feature sets.
        existing_cols = set(existing_df.columns)
        refreshed_cols = set(refreshed_recent_df.columns)

        if existing_cols != refreshed_cols:
            missing_in_existing = sorted(refreshed_cols - existing_cols)
            extra_in_existing = sorted(existing_cols - refreshed_cols)

            print(f"[{ticker}] Schema mismatch detected. Full rebuild from {START_DATE.date()}.")
            if missing_in_existing:
                print(f"[{ticker}] Missing in existing: {missing_in_existing}")
            if extra_in_existing:
                print(f"[{ticker}] Extra in existing:   {extra_in_existing}")

            combined_df = get_daily_equity(ticker, start_date=START_DATE)
            added_rows = len(combined_df) - len(existing_df)
        else:
            combined_df = combine_existing_with_new(existing_df, refreshed_recent_df)
            added_rows = len(combined_df) - len(existing_df)

    if combined_df.empty:
        raise ValueError(
            f"[{ticker}] combined_df is empty before split. "
            f"get_daily_equity() returned no usable rows."
        )

    train_df, test_df = split_train_test(combined_df, test_days=TEST_DAYS)

    print(f"[{ticker}] Added rows:  {added_rows}")
    print(f"[{ticker}] Total rows:  {len(combined_df)}")
    print(f"[{ticker}] Train rows:  {len(train_df)}")
    print(f"[{ticker}] Test rows:   {len(test_df)}")
    print(f"[{ticker}] New last date: {combined_df['Date'].max().date()}")

    train_df.to_csv(train_path(ticker), index=False)
    test_df.to_csv(test_path(ticker), index=False)

    ax1 = train_df.set_index("Date")[["Close"]].plot(title=f"{ticker} TRAIN")
    plt.close(ax1.figure)

    ax2 = test_df.set_index("Date")[["Close"]].plot(title=f"{ticker} TEST")
    plt.close(ax2.figure)


def cleanup_non_model_datasets() -> list[Path]:
    grip, edip = load_allocation_tickers()
    model_symbols = set(grip) | set(edip)

    if not model_symbols:
        print(
            "Skipping train/test dataset cleanup because no GRIP or EDIP "
            "allocation symbols were found."
        )
        return []

    keep_symbols = model_symbols | set(ALWAYS_RUN_TICKERS)
    removed_paths = []

    for directory in (TRAIN_DIR, TEST_DIR):
        if directory is None or not directory.exists():
            continue

        for path in directory.glob("*.csv"):
            if path.stem.upper() in keep_symbols:
                continue
            path.unlink()
            removed_paths.append(path)

    if removed_paths:
        print(f"Removed {len(removed_paths)} train/test CSVs not in GRIP or EDIP:")
        for path in removed_paths:
            print(f"  {path}")
    else:
        print("No train/test CSV cleanup needed.")

    return removed_paths


# =============================================================================
# ENTRY POINT
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Update UnrealAI train/test feature CSVs.")
    parser.add_argument(
        "--tickers",
        help="Comma, space, or semicolon separated symbols. Overrides env/allocation/CSV discovery.",
    )
    parser.add_argument(
        "--ticker-file",
        type=Path,
        help="Text file containing symbols separated by commas, whitespace, or semicolons.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Update only the first N resolved tickers. Useful for smoke tests.",
    )
    parser.add_argument(
        "--list-config",
        action="store_true",
        help="Print resolved paths and tickers without updating data.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        launched_by_ipykernel = "ipykernel" in Path(sys.argv[0]).name.lower()
        if launched_by_ipykernel:
            print(f"Ignoring Jupyter kernel arguments: {' '.join(unknown)}")
        else:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return args


def resolve_run_tickers(args) -> list[str]:
    if args.tickers:
        tickers = parse_ticker_list(args.tickers)
    elif args.ticker_file:
        tickers = _read_ticker_file(args.ticker_file.expanduser())
    else:
        tickers = list(TICKERS)

    if args.limit is not None:
        tickers = tickers[: max(args.limit, 0)]

    return add_always_run_tickers(tickers)


def print_config(tickers: list[str]):
    print("Resolved data_update configuration:")
    print(f"  APP_DIR:               {APP_DIR}")
    print(f"  TRAIN_DIR:             {TRAIN_DIR}")
    print(f"  TEST_DIR:              {TEST_DIR}")
    print(f"  GRIP_ALLOCATION_PATH:  {GRIP_ALLOCATION_PATH or '(not configured)'}")
    print(f"  EDIP_ALLOCATION_PATH:  {EDIP_ALLOCATION_PATH or '(not configured)'}")
    print(f"  norgatedata available: {norgatedata is not None}")
    print(f"  tickers ({len(tickers)}):        {', '.join(tickers) if tickers else '(none)'}")
    sys.stdout.flush()


def main():
    args = parse_args()
    tickers = resolve_run_tickers(args)
    print_config(tickers)

    if args.list_config:
        return

    if not tickers:
        raise RuntimeError(
            "No tickers configured. Set UNREALAI_TICKERS, set UNREALAI_TICKER_FILE, "
            "configure allocation Excel paths, or place existing CSVs in TRAIN_DIR/TEST_DIR."
        )

    try:
        require_norgatedata()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_non_model_datasets()

    for ticker in tickers:
        update_one_ticker(ticker)


if __name__ == "__main__":
    main()
