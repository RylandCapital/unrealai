from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

APP_TITLE = "cwpAI"
DEFAULT_DATA_DIR = Path(r"P:\10_CWP Trade Department\Ryland\unrealai\unrealai\dashboard_data")
FALLBACK_DATA_DIR = Path("dashboard_data")

FILL_LABELS = {
    "close_t": "Close T0",
    "open_t_plus_1": "Close T+1 Open",
    "hl2_t_plus_1": "HL2 T+1",
    "open_t_plus_1_discount_only": "Open T+1 Discount Only",
}

MODEL_CONFIG = {
    "GRIP": {
        "path": Path(r"P:\10_CWP Trade Department\_Matrix_\code_outputs\grip_momo\grip_allocation.xlsx"),
        "header": 1,
        "ticker_col": "Ticker",
    },
    "EDIP": {
        "path": Path(r"P:\10_CWP Trade Department\Smitty\DSIP allocation.xlsx"),
        "header": 0,
        "ticker_col": "Unnamed: 1",
    },
}

MODEL_OPTIONS = ["ALL SYMBOLS", "GRIP", "EDIP"]

DATE_LIKE_COLS = [
    "date",
    "Date",
    "entry_date",
    "exit_date",
    "last_bar_date",
    "last_signal_date",
    "report_start_date",
    "as_of",
    "saved_model_timestamp_utc",
    "current_date",
    "final_date",
]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"], [data-testid="collapsedControl"] {
                display: none !important;
            }
            .block-container {
                max-width: 94rem;
                padding-top: 1.45rem;
                padding-bottom: 0.8rem;
            }
            html, body, [class*="css"] {
                font-size: 13px;
            }
            .app-title {
                font-size: 3.0rem;
                font-weight: 800;
                line-height: 1.05;
                margin: 0 0 0.20rem 0;
                padding: 0;
            }
            .subnote {
                color: rgba(255,255,255,0.78);
                font-size: 0.95rem;
                margin-bottom: 0.55rem;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 1.10rem;
            }
            .stTabs [data-baseweb="tab"] {
                font-size: 0.98rem;
                padding-top: 0.35rem;
                padding-bottom: 0.35rem;
                padding-left: 0;
                padding-right: 0;
            }
            .kpi-card {
                background: rgba(255,255,255,0.02);
                border-radius: 12px;
                padding: 0.60rem 0.72rem 0.54rem 0.72rem;
                min-height: 84px;
                margin-bottom: 0.45rem;
                box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
            }
            .kpi-label {
                font-size: 0.74rem;
                color: rgba(255,255,255,0.86);
                line-height: 1.15;
                margin-bottom: 0.30rem;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .kpi-value {
                font-size: 0.90rem;
                font-weight: 700;
                line-height: 1.18;
                color: #ffffff;
                white-space: normal;
                word-break: break-word;
                overflow-wrap: anywhere;
            }
            .section-title {
                font-size: 1.18rem;
                font-weight: 700;
                margin: 0.20rem 0 0.35rem 0;
            }
            .chart-title {
                font-size: 1.00rem;
                font-weight: 700;
                margin: 0.25rem 0 0.10rem 0;
            }
            .range-wrap {
                margin: 0.10rem 0 0.25rem 0;
            }
            .range-label {
                font-size: 0.95rem;
                font-weight: 600;
                line-height: 1.3;
                margin: 0 0 0.35rem 0;
            }
            div[data-testid="stDataFrame"] div[role="table"] {
                font-size: 0.84rem;
            }
            .activity-box {
                background: rgba(255,255,255,0.02);
                border-radius: 12px;
                padding: 0.85rem 0.95rem 0.80rem 0.95rem;
                min-height: 180px;
                margin-bottom: 0.55rem;
                box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
            }
            .activity-box-title {
                font-size: 1.00rem;
                font-weight: 700;
                margin: 0 0 0.55rem 0;
            }
            .activity-count {
                font-size: 0.86rem;
                color: rgba(255,255,255,0.78);
                margin-bottom: 0.55rem;
            }
            .symbol-pill-wrap {
                display: flex;
                flex-wrap: wrap;
                gap: 0.40rem;
            }
            .symbol-pill {
                display: inline-block;
                padding: 0.30rem 0.58rem;
                border-radius: 999px;
                background: rgba(255,255,255,0.06);
                font-size: 0.82rem;
                line-height: 1.1;
                white-space: nowrap;
            }
            .activity-empty {
                color: rgba(255,255,255,0.70);
                font-size: 0.90rem;
                padding-top: 0.20rem;
            }
            .model-toggle-label {
                font-size: 0.95rem;
                font-weight: 600;
                line-height: 1.3;
                margin: 0 0 0.35rem 0;
            }
            div[data-testid="stRadio"] > div {
                flex-direction: row;
                gap: 0.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def detect_data_dir() -> Path:
    return DEFAULT_DATA_DIR if DEFAULT_DATA_DIR.exists() else FALLBACK_DATA_DIR


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_table_auto(base_path: str) -> pd.DataFrame:
    parquet_path = Path(f"{base_path}.parquet")
    csv_path = Path(f"{base_path}.csv")

    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            if csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                raise
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Missing both {parquet_path.name} and {csv_path.name}")

    for col in DATE_LIKE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_bundle(data_dir_str: str) -> dict:
    data_dir = Path(data_dir_str)
    summary_path = data_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {data_dir}")

    return {
        "data_dir": data_dir,
        "summary": load_json(str(summary_path)),
        "morning_report": load_table_auto(str(data_dir / "morning_report")),
        "symbol_metrics": load_table_auto(str(data_dir / "symbol_metrics")),
        "symbol_timeseries": load_table_auto(str(data_dir / "symbol_timeseries")),
        "trade_log": load_table_auto(str(data_dir / "trade_log")),
        "aggregate_timeseries": load_table_auto(str(data_dir / "aggregate_timeseries")),
    }


def maybe_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        hit = lower_map.get(str(c).lower())
        if hit is not None:
            return hit
    return None


def get_date_col(df: pd.DataFrame) -> Optional[str]:
    return maybe_col(df, ["date", "Date"])


def get_symbol_col(df: pd.DataFrame) -> Optional[str]:
    return maybe_col(df, ["symbol", "ticker"])


def get_primary_fill_mode(bundle: dict) -> str:
    summary = bundle.get("summary", {}) if isinstance(bundle, dict) else {}
    mode = str(summary.get("primary_fill_mode") or "close_t").strip()
    return mode or "close_t"


def get_fill_label(fill_mode: str) -> str:
    return FILL_LABELS.get(str(fill_mode), str(fill_mode))


def get_fill_series_col(df: pd.DataFrame, fill_mode: str, *, prefer_primary_alias: bool = False) -> Optional[str]:
    fill_mode = str(fill_mode)
    candidates: list[str] = []

    if fill_mode == "close_t":
        candidates.extend(["equity_close_t", "close_t", "equity"])
    elif fill_mode == "open_t_plus_1":
        candidates.extend(["equity_open_t_plus_1", "open_t_plus_1"])
    elif fill_mode == "hl2_t_plus_1":
        candidates.extend(["equity_hl2_t_plus_1", "hl2_t_plus_1"])
    elif fill_mode == "open_t_plus_1_discount_only":
        candidates.extend(["equity_open_t_plus_1_discount_only", "open_t_plus_1_discount_only"])

    if prefer_primary_alias:
        candidates.append("equity_primary_live")
        candidates.append("primary_live")

    return maybe_col(df, candidates)


def fmt_pct(x: Optional[float], decimals: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):,.{decimals}f}%"


def fmt_num(x: Optional[float], decimals: int = 0) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):,.{decimals}f}"


def fmt_ratio(x: Optional[float], decimals: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{float(x):,.{decimals}f}"


def fmt_money(x: Optional[float], decimals: int = 0) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"${float(x):,.{decimals}f}"


def to_return_pct(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    base = float(valid.iloc[0])
    if base == 0:
        return pd.Series(np.nan, index=series.index)
    return (s / base - 1.0) * 100.0


def drawdown_series_pct(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    roll_max = s.cummax().replace(0, np.nan)
    return (s / roll_max - 1.0) * 100.0


def drawdown_min_pct(series: pd.Series) -> float:
    dd = drawdown_series_pct(series).dropna()
    return float(dd.min()) if not dd.empty else np.nan


def compute_capture_ratio(numerator: Optional[float], denominator: Optional[float]) -> float:
    if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator):
        return np.nan

    num = float(numerator)
    den = float(denominator)
    if den == 0:
        return np.nan
    if (num < 0 < den) or (num > 0 > den):
        return np.nan
    return num / den


def normalize_stance(value: object) -> str:
    if pd.isna(value):
        return "FLAT"
    text = str(value).strip().upper()
    if text in {"LONG", "1", "1.0", "TRUE"}:
        return "LONG"
    if text in {"FLAT", "0", "0.0", "FALSE"}:
        return "FLAT"
    return "LONG" if text not in {"", "NONE", "NAN"} else "FLAT"


def add_stance_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "stance" in out.columns:
        out["activity_stance"] = out["stance"].map(normalize_stance)
    elif "position" in out.columns:
        pos = pd.to_numeric(out["position"], errors="coerce").fillna(0)
        out["activity_stance"] = np.where(pos > 0, "LONG", "FLAT")
    else:
        out["activity_stance"] = "FLAT"
    return out


def clean_model_symbols(values: list[object]) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()

    for value in values:
        if pd.isna(value):
            continue
        symbol = str(value).strip().upper()
        if not symbol:
            continue
        if symbol in {"TICKER", "SYMBOL", "NAN", "NONE", "CASH"}:
            continue
        if symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


@st.cache_data(show_spinner=False)
def load_model_symbols(model_name: str) -> list[str]:
    if model_name == "ALL SYMBOLS":
        merged_symbols: list[str] = []
        for source_name in ("GRIP", "EDIP"):
            merged_symbols.extend(load_model_symbols(source_name))
        return clean_model_symbols(merged_symbols)

    cfg = MODEL_CONFIG[model_name]
    path = cfg["path"]
    if not path.exists():
        raise FileNotFoundError(f"Missing model allocation file: {path}")

    df = pd.read_excel(path, header=cfg["header"])
    ticker_col = cfg["ticker_col"]
    if ticker_col not in df.columns:
        raise KeyError(f"Column '{ticker_col}' not found in {path.name}")

    return clean_model_symbols(df[ticker_col].dropna().tolist())


def filter_df_to_symbols(df: pd.DataFrame, symbols: set[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    symbol_col = get_symbol_col(df)
    if symbol_col is None:
        return df.copy()

    out = df.copy()
    out[symbol_col] = out[symbol_col].astype(str).str.strip().str.upper()
    return out[out[symbol_col].isin(symbols)].copy()


def build_aggregate_from_symbol_timeseries(sym_ts: pd.DataFrame) -> pd.DataFrame:
    if sym_ts.empty:
        return pd.DataFrame()

    dcol = get_date_col(sym_ts)
    if dcol is None:
        return pd.DataFrame()

    work = sym_ts.copy()
    work[dcol] = pd.to_datetime(work[dcol], errors="coerce")
    work = work.dropna(subset=[dcol]).copy()
    if work.empty:
        return pd.DataFrame()

    result = (
        work[[dcol]]
        .drop_duplicates()
        .sort_values(dcol)
        .reset_index(drop=True)
    )

    metric_map = {
        "buy_hold": ["benchmark_equity", "buy_hold", "equity_buy_hold"],
        "close_t": ["equity_close_t", "close_t", "equity"],
        "open_t_plus_1": ["equity_open_t_plus_1", "open_t_plus_1", "primary_live"],
        "hl2_t_plus_1": ["equity_hl2_t_plus_1", "hl2_t_plus_1"],
        "open_t_plus_1_discount_only": ["equity_open_t_plus_1_discount_only", "open_t_plus_1_discount_only"],
    }

    for out_col, candidates in metric_map.items():
        src_col = maybe_col(work, candidates)
        if src_col is None:
            continue

        tmp = work[[dcol]].copy()
        tmp[out_col] = pd.to_numeric(work[src_col], errors="coerce")
        agg_series = tmp.groupby(dcol, dropna=True)[out_col].sum(min_count=1).reset_index()
        result = result.merge(agg_series, on=dcol, how="left")

    stance_work = add_stance_column(work)
    if "activity_stance" in stance_work.columns:
        tmp = stance_work[[dcol]].copy()
        tmp["open_positions"] = stance_work["activity_stance"].eq("LONG").astype(int)
        open_pos = tmp.groupby(dcol, dropna=True)["open_positions"].sum().reset_index()
        result = result.merge(open_pos, on=dcol, how="left")

    reporting = work.groupby(dcol, dropna=True).size().reset_index(name="symbols_reporting")
    result = result.merge(reporting, on=dcol, how="left")

    return result.sort_values(dcol).reset_index(drop=True)


def apply_model_filter(bundle: dict, model_symbols: list[str]) -> dict:
    out = {**bundle}
    symbol_set = set(model_symbols)

    morning = filter_df_to_symbols(bundle["morning_report"], symbol_set)
    symbol_metrics = filter_df_to_symbols(bundle["symbol_metrics"], symbol_set)
    symbol_timeseries = filter_df_to_symbols(bundle["symbol_timeseries"], symbol_set)
    trade_log = filter_df_to_symbols(bundle["trade_log"], symbol_set)

    out["morning_report"] = morning
    out["symbol_metrics"] = symbol_metrics
    out["symbol_timeseries"] = symbol_timeseries
    out["trade_log"] = trade_log
    out["aggregate_timeseries"] = build_aggregate_from_symbol_timeseries(symbol_timeseries)

    return out


def filter_bundle(bundle: dict, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> dict:
    out = {**bundle}

    agg = bundle["aggregate_timeseries"].copy()
    agg_date_col = get_date_col(agg)
    if agg_date_col:
        agg = agg[(agg[agg_date_col] >= start_dt) & (agg[agg_date_col] <= end_dt)].copy()

    sym_ts = bundle["symbol_timeseries"].copy()
    sym_date_col = get_date_col(sym_ts)
    if sym_date_col:
        sym_ts = sym_ts[(sym_ts[sym_date_col] >= start_dt) & (sym_ts[sym_date_col] <= end_dt)].copy()

    trades = bundle["trade_log"].copy()
    if not trades.empty:
        if "entry_date" in trades.columns or "exit_date" in trades.columns:
            entry_dt = pd.to_datetime(trades.get("entry_date"), errors="coerce") if "entry_date" in trades.columns else pd.Series(pd.NaT, index=trades.index)
            exit_dt = pd.to_datetime(trades.get("exit_date"), errors="coerce") if "exit_date" in trades.columns else pd.Series(pd.NaT, index=trades.index)
            still_open = exit_dt.isna()
            overlaps = (entry_dt <= end_dt) & (still_open | (exit_dt >= start_dt))
            trades = trades[overlaps.fillna(False)].copy()
        elif "date" in trades.columns:
            d = pd.to_datetime(trades["date"], errors="coerce")
            trades = trades[(d >= start_dt) & (d <= end_dt)].copy()

    latest_snapshot = pd.DataFrame()
    if not sym_ts.empty and get_symbol_col(sym_ts) is not None and sym_date_col:
        sym_col = get_symbol_col(sym_ts) or "symbol"
        latest_snapshot = (
            sym_ts.sort_values([sym_col, sym_date_col])
            .groupby(sym_col, as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )

    morning = bundle["morning_report"].copy()
    if not latest_snapshot.empty and get_symbol_col(morning) is not None:
        morning_col = get_symbol_col(morning) or "symbol"
        latest_col = get_symbol_col(latest_snapshot) or "symbol"
        morning = morning[morning[morning_col].astype(str).isin(latest_snapshot[latest_col].astype(str))].copy()

    symbol_metrics = bundle["symbol_metrics"].copy()
    if not latest_snapshot.empty and get_symbol_col(symbol_metrics) is not None:
        metrics_col = get_symbol_col(symbol_metrics) or "symbol"
        latest_col = get_symbol_col(latest_snapshot) or "symbol"
        symbol_metrics = symbol_metrics[symbol_metrics[metrics_col].astype(str).isin(latest_snapshot[latest_col].astype(str))].copy()

    out["aggregate_timeseries"] = agg
    out["symbol_timeseries"] = sym_ts
    out["trade_log"] = trades
    out["latest_snapshot"] = latest_snapshot
    out["morning_report"] = morning
    out["symbol_metrics"] = symbol_metrics
    return out


def compute_model_saved_date(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    if "saved_model_timestamp_utc" in df.columns:
        ts = pd.to_datetime(df["saved_model_timestamp_utc"], errors="coerce", utc=True)
        try:
            ts = ts.dt.tz_convert(None)
        except Exception:
            pass
        result = ts.dt.strftime("%Y-%m-%d")
    if "last_bar_date" in df.columns:
        fallback = pd.to_datetime(df["last_bar_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        result = result.fillna(fallback)
    if "saved_model_last_trained_date" in df.columns:
        fallback2 = pd.to_datetime(df["saved_model_last_trained_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        result = result.fillna(fallback2)
    return result


def build_overview_metrics(filtered: dict) -> list[tuple[str, str]]:
    agg = filtered["aggregate_timeseries"]
    latest_snapshot = filtered.get("latest_snapshot", pd.DataFrame())
    trades = filtered["trade_log"]
    primary_fill_mode = get_primary_fill_mode(filtered)
    primary_fill_label = get_fill_label(primary_fill_mode)

    metrics: dict[str, object] = {
        "As Of": "-",
        "Primary Fill": primary_fill_label,
        "Open Positions": 0,
        "Exposure": np.nan,
        "Trades": 0,
        "Win Rate": np.nan,
        "Portfolio Return": np.nan,
        "Benchmark Return": np.nan,
        "Alpha": np.nan,
        "Capture": np.nan,
        "Max Drawdown": np.nan,
        "Benchmark Drawdown": np.nan,
        "Drawdown Capture": np.nan,
    }

    dcol = get_date_col(agg)
    if dcol and not agg.empty:
        metrics["As Of"] = pd.to_datetime(agg[dcol].iloc[-1], errors="coerce").strftime("%Y-%m-%d")

    bench_col = maybe_col(agg, ["buy_hold", "benchmark_equity", "benchmark"])
    primary_col = get_fill_series_col(agg, primary_fill_mode, prefer_primary_alias=True)

    if primary_col and not agg.empty:
        ret = to_return_pct(agg[primary_col]).iloc[-1]
        metrics["Portfolio Return"] = ret
        metrics["Max Drawdown"] = drawdown_min_pct(agg[primary_col])

    if bench_col and not agg.empty:
        metrics["Benchmark Return"] = to_return_pct(agg[bench_col]).iloc[-1]
        metrics["Benchmark Drawdown"] = drawdown_min_pct(agg[bench_col])

    if pd.notna(metrics["Portfolio Return"]) and pd.notna(metrics["Benchmark Return"]):
        metrics["Alpha"] = float(metrics["Portfolio Return"] - metrics["Benchmark Return"])
        metrics["Capture"] = compute_capture_ratio(metrics["Portfolio Return"], metrics["Benchmark Return"])

    if pd.notna(metrics["Max Drawdown"]) and pd.notna(metrics["Benchmark Drawdown"]):
        metrics["Drawdown Capture"] = compute_capture_ratio(metrics["Max Drawdown"], metrics["Benchmark Drawdown"])

    if not latest_snapshot.empty:
        latest_snapshot = add_stance_column(latest_snapshot)
        open_positions = int(latest_snapshot["activity_stance"].eq("LONG").sum())
        metrics["Open Positions"] = open_positions
        metrics["Exposure"] = (open_positions / max(len(latest_snapshot), 1)) * 100.0

    closed = trades.copy()
    if not closed.empty and "status" in closed.columns:
        closed = closed[closed["status"].astype(str).str.upper().eq("CLOSED")].copy()
    elif not closed.empty and "exit_date" in closed.columns:
        closed = closed[pd.to_datetime(closed["exit_date"], errors="coerce").notna()].copy()

    metrics["Trades"] = int(len(closed))
    if not closed.empty and "realized_pnl" in closed.columns:
        realized = pd.to_numeric(closed["realized_pnl"], errors="coerce").fillna(0.0)
        metrics["Win Rate"] = float((realized > 0).mean() * 100.0)

    return [
        ("As Of", str(metrics["As Of"])),
        ("Primary Fill", primary_fill_label),
        ("Open Positions", fmt_num(metrics["Open Positions"], 0)),
        ("Exposure", fmt_pct(metrics["Exposure"])),
        ("Trades", fmt_num(metrics["Trades"], 0)),
        ("Win Rate", fmt_pct(metrics["Win Rate"])),
        ("Portfolio Return", fmt_pct(metrics["Portfolio Return"])),
        ("Benchmark Return", fmt_pct(metrics["Benchmark Return"])),
        ("Alpha", fmt_pct(metrics["Alpha"])),
        ("Capture", fmt_ratio(metrics["Capture"])),
        ("Max Drawdown", fmt_pct(metrics["Max Drawdown"])),
        ("Benchmark Drawdown", fmt_pct(metrics["Benchmark Drawdown"])),
        ("Drawdown Capture", fmt_ratio(metrics["Drawdown Capture"])),
    ]


def kpi_card(label: str, value: str) -> str:
    return (
        "<div class='kpi-card'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        "</div>"
    )


def render_kpis(metrics: list[tuple[str, str]]) -> None:
    row1 = st.columns(6)
    for col, (label, value) in zip(row1, metrics[:6]):
        col.markdown(kpi_card(label, value), unsafe_allow_html=True)

    row2 = st.columns(len(metrics[6:]))
    for col, (label, value) in zip(row2, metrics[6:]):
        col.markdown(kpi_card(label, value), unsafe_allow_html=True)


def make_portfolio_chart(agg: pd.DataFrame, primary_fill_mode: str) -> go.Figure:
    fig = go.Figure()
    if agg.empty:
        return fig

    dcol = get_date_col(agg)
    if dcol is None:
        return fig

    x = agg[dcol]
    bench_col = maybe_col(agg, ["buy_hold", "benchmark_equity", "benchmark"])

    if bench_col:
        fig.add_trace(go.Scatter(x=x, y=to_return_pct(agg[bench_col]), mode="lines", name="Benchmark Return", line=dict(width=2.4)))

    for fill_mode in FILL_LABELS:
        col = get_fill_series_col(agg, fill_mode, prefer_primary_alias=(fill_mode == primary_fill_mode))
        if not col:
            continue
        fig.add_trace(
            go.Scatter(
                x=x,
                y=to_return_pct(agg[col]),
                mode="lines",
                name=get_fill_label(fill_mode),
                line=dict(width=3.2 if fill_mode == primary_fill_mode else 2.0),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            itemclick=False,
            itemdoubleclick=False,
        ),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Return %")
    return fig


def make_open_positions_chart(agg: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if agg.empty:
        return fig
    dcol = get_date_col(agg)
    pos_col = maybe_col(agg, ["open_positions"])
    if dcol is None or pos_col is None:
        return fig

    positions = pd.to_numeric(agg[pos_col], errors="coerce")
    fig.add_trace(go.Bar(x=agg[dcol], y=positions, name="Open Positions"), secondary_y=False)

    reporting_col = maybe_col(agg, ["symbols_reporting"])
    avg_exposure = np.nan
    if reporting_col is not None:
        reporting = pd.to_numeric(agg[reporting_col], errors="coerce").replace(0, np.nan)
        exposure_pct = (positions / reporting) * 100.0
        avg_exposure = float(exposure_pct.dropna().mean()) if exposure_pct.notna().any() else np.nan
        if pd.notna(avg_exposure):
            fig.add_trace(
                go.Scatter(
                    x=agg[dcol],
                    y=[avg_exposure] * len(agg),
                    mode="lines",
                    name=f"Avg Exposure ({avg_exposure:.1f}%)",
                    line=dict(color="#f0c36d", width=2, dash="dash"),
                    hovertemplate="Avg Exposure: %{y:.1f}%<extra></extra>",
                ),
                secondary_y=True,
            )

    fig.update_layout(
        template="plotly_dark",
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=pd.notna(avg_exposure),
    )
    fig.update_yaxes(title_text="Positions", secondary_y=False)
    fig.update_yaxes(title_text="Exposure %", range=[0, 100], ticksuffix="%", secondary_y=True)
    return fig


def make_symbol_chart(symbol: str, sym_ts: pd.DataFrame, primary_fill_mode: str) -> go.Figure:
    fig = go.Figure()
    if sym_ts.empty:
        return fig

    dcol = get_date_col(sym_ts)
    if dcol is None:
        return fig
    x = sym_ts[dcol]

    bench_col = maybe_col(sym_ts, ["benchmark_equity", "buy_hold", "equity_buy_hold"])

    if bench_col:
        fig.add_trace(go.Scatter(x=x, y=pd.to_numeric(sym_ts[bench_col], errors="coerce"), mode="lines", name="Buy & Hold", line=dict(width=2.3)))
    for fill_mode in FILL_LABELS:
        col = get_fill_series_col(sym_ts, fill_mode, prefer_primary_alias=(fill_mode == primary_fill_mode))
        if not col:
            continue
        fig.add_trace(
            go.Scatter(
                x=x,
                y=pd.to_numeric(sym_ts[col], errors="coerce"),
                mode="lines",
                name=get_fill_label(fill_mode),
                line=dict(width=3.0 if fill_mode == primary_fill_mode else 2.0),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            itemclick=False,
            itemdoubleclick=False,
        ),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Equity")
    return fig


def build_attribution_table(filtered: dict) -> pd.DataFrame:
    sym_ts = filtered["symbol_timeseries"]
    sym_col = get_symbol_col(sym_ts)
    dcol = get_date_col(sym_ts)
    primary_fill_mode = get_primary_fill_mode(filtered)
    equity_col = get_fill_series_col(sym_ts, primary_fill_mode, prefer_primary_alias=True)
    bench_col = maybe_col(sym_ts, ["benchmark_equity", "buy_hold", "equity_buy_hold"])

    if sym_ts.empty or sym_col is None or dcol is None or equity_col is None or bench_col is None:
        return pd.DataFrame()

    work = sym_ts[[sym_col, dcol, equity_col, bench_col]].copy()
    work[dcol] = pd.to_datetime(work[dcol], errors="coerce")
    work[equity_col] = pd.to_numeric(work[equity_col], errors="coerce")
    work[bench_col] = pd.to_numeric(work[bench_col], errors="coerce")
    work = work.dropna(subset=[sym_col, dcol, equity_col, bench_col]).sort_values([sym_col, dcol])
    if work.empty:
        return pd.DataFrame()

    start_rows = (
        work.groupby(sym_col, as_index=False)
        .head(1)[[sym_col, dcol, equity_col, bench_col]]
        .rename(columns={sym_col: "symbol", dcol: "start_date", equity_col: "start_equity", bench_col: "start_benchmark_equity"})
    )
    end_rows = (
        work.groupby(sym_col, as_index=False)
        .tail(1)[[sym_col, dcol, equity_col, bench_col]]
        .rename(columns={sym_col: "symbol", dcol: "end_date", equity_col: "end_equity", bench_col: "end_benchmark_equity"})
    )

    attribution = start_rows.merge(end_rows, on="symbol", how="inner")
    if attribution.empty:
        return attribution

    attribution["strategy_pnl_change"] = attribution["end_equity"] - attribution["start_equity"]
    attribution["benchmark_pnl_change"] = attribution["end_benchmark_equity"] - attribution["start_benchmark_equity"]
    attribution["symbol_return_pct"] = ((attribution["end_equity"] / attribution["start_equity"]) - 1.0) * 100.0
    attribution["benchmark_return_pct"] = ((attribution["end_benchmark_equity"] / attribution["start_benchmark_equity"]) - 1.0) * 100.0
    attribution["alpha_pct"] = attribution["symbol_return_pct"] - attribution["benchmark_return_pct"]
    attribution["alpha_pnl_change"] = attribution["strategy_pnl_change"] - attribution["benchmark_pnl_change"]
    attribution = attribution.replace([np.inf, -np.inf], np.nan)
    attribution = attribution.dropna(subset=["alpha_pct"]).sort_values("alpha_pct", ascending=True).reset_index(drop=True)
    return attribution


def make_attribution_chart(attribution: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if attribution.empty:
        return fig

    colors = ["#ff5c5c" if value < 0 else "#36c275" for value in attribution["alpha_pct"]]
    customdata = np.column_stack(
        [
            attribution["alpha_pnl_change"],
            attribution["symbol_return_pct"],
            attribution["benchmark_return_pct"],
            attribution["start_date"].dt.strftime("%Y-%m-%d"),
            attribution["end_date"].dt.strftime("%Y-%m-%d"),
        ]
    )

    fig.add_trace(
        go.Bar(
            x=attribution["alpha_pct"],
            y=attribution["symbol"],
            orientation="h",
            marker_color=colors,
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Alpha vs Buy & Hold: %{x:.2f} pts<br>"
                "Alpha PnL: $%{customdata[0]:,.0f}<br>"
                "Strategy Return: %{customdata[1]:.2f}%<br>"
                "Buy & Hold Return: %{customdata[2]:.2f}%<br>"
                "Period: %{customdata[3]} to %{customdata[4]}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=max(340, 28 * len(attribution) + 80),
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Alpha vs Buy & Hold (pts)", zeroline=True, zerolinecolor="rgba(255,255,255,0.35)")
    fig.update_yaxes(title_text="")
    return fig


def convert_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        low = str(col).lower()
        if "date" in low or low.endswith("_at") or low.endswith("_utc"):
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
        elif "pct" in low or "return" in low or "drawdown" in low or low.endswith("alpha"):
            vals = pd.to_numeric(out[col], errors="coerce")
            out[col] = vals.map(lambda x: f"{x:,.2f}%" if pd.notna(x) else "")
        elif any(token in low for token in ["price", "equity", "pnl", "cash"]) and "position" not in low:
            vals = pd.to_numeric(out[col], errors="coerce")
            out[col] = vals.map(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
    return out


def display_trade_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No trades in the selected date range.")
        return

    keep = [
        c for c in [
            "symbol",
            "status",
            "entry_date",
            "entry_price",
            "exit_date",
            "exit_price",
            "holding_days",
            "trade_return_pct",
            "current_price",
        ]
        if c in df.columns
    ]
    st.dataframe(convert_for_display(df[keep]), use_container_width=True, hide_index=True)


def render_header(report_start: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> None:
    st.markdown(f"<div class='app-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='subnote'>Report start: {report_start} &nbsp;|&nbsp; Date range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}</div>",
        unsafe_allow_html=True,
    )


def render_overview(filtered: dict) -> None:
    st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)
    render_kpis(build_overview_metrics(filtered))

    agg = filtered["aggregate_timeseries"]
    primary_fill_mode = get_primary_fill_mode(filtered)

    st.markdown("<div class='chart-title'>Portfolio vs Benchmark</div>", unsafe_allow_html=True)
    st.plotly_chart(make_portfolio_chart(agg, primary_fill_mode), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='chart-title'>Open Positions</div>", unsafe_allow_html=True)
    st.plotly_chart(make_open_positions_chart(agg), use_container_width=True, config={"displayModeBar": False})


def build_activity_snapshot(sym_ts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    if sym_ts.empty or get_symbol_col(sym_ts) is None:
        return pd.DataFrame(), pd.DataFrame(), None, None

    dcol = get_date_col(sym_ts)
    if dcol is None:
        return pd.DataFrame(), pd.DataFrame(), None, None

    sym_col = get_symbol_col(sym_ts) or "symbol"
    work = sym_ts.copy()
    work[dcol] = pd.to_datetime(work[dcol], errors="coerce")
    work = work.dropna(subset=[dcol]).copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), None, None

    work = add_stance_column(work)
    work[sym_col] = work[sym_col].astype(str)

    latest_dt = work[dcol].max()
    previous_dates = sorted(d for d in work[dcol].dropna().unique() if d < latest_dt)
    prev_dt = previous_dates[-1] if previous_dates else None

    current_df = (
        work[work[dcol] == latest_dt]
        .sort_values([sym_col, dcol])
        .groupby(sym_col, as_index=False)
        .tail(1)
        .sort_values(sym_col)
        .reset_index(drop=True)
    )

    changes_df = pd.DataFrame()
    if prev_dt is not None:
        prev_df = (
            work[work[dcol] == prev_dt]
            .sort_values([sym_col, dcol])
            .groupby(sym_col, as_index=False)
            .tail(1)[[sym_col, "activity_stance"]]
            .rename(columns={sym_col: "symbol", "activity_stance": "previous_stance"})
            .reset_index(drop=True)
        )
        curr_df = (
            current_df[[sym_col, "activity_stance"]]
            .rename(columns={sym_col: "symbol", "activity_stance": "current_stance"})
            .reset_index(drop=True)
        )

        changes_df = curr_df.merge(prev_df, on="symbol", how="outer")
        changes_df["previous_stance"] = changes_df["previous_stance"].fillna("FLAT")
        changes_df["current_stance"] = changes_df["current_stance"].fillna("FLAT")
        changes_df = changes_df[changes_df["previous_stance"] != changes_df["current_stance"]].copy()
        if not changes_df.empty:
            changes_df["change"] = changes_df["previous_stance"] + " → " + changes_df["current_stance"]
            changes_df["change_date"] = pd.to_datetime(latest_dt)
            changes_df = changes_df.sort_values("symbol").reset_index(drop=True)

    if sym_col != "symbol":
        current_df = current_df.rename(columns={sym_col: "symbol"})

    return current_df, changes_df, latest_dt, prev_dt


def render_symbol_pills(symbols: list[str]) -> str:
    if not symbols:
        return "<div class='activity-empty'>None</div>"
    pills = "".join(f"<span class='symbol-pill'>{s}</span>" for s in symbols)
    return f"<div class='symbol-pill-wrap'>{pills}</div>"


def render_activity(filtered: dict) -> None:
    st.markdown("<div class='section-title'>Activity</div>", unsafe_allow_html=True)

    sym_ts = filtered["symbol_timeseries"]
    current_df, changes_df, latest_dt, prev_dt = build_activity_snapshot(sym_ts)

    if current_df.empty:
        st.info("No activity data in the selected date range.")
        return

    longs = sorted(current_df.loc[current_df["activity_stance"] == "LONG", "symbol"].astype(str).tolist())
    flats = sorted(current_df.loc[current_df["activity_stance"] == "FLAT", "symbol"].astype(str).tolist())

    row = st.columns(2)
    row[0].markdown(
        (
            "<div class='activity-box'>"
            "<div class='activity-box-title'>Long</div>"
            f"<div class='activity-count'>{len(longs)} symbols</div>"
            f"{render_symbol_pills(longs)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    row[1].markdown(
        (
            "<div class='activity-box'>"
            "<div class='activity-box-title'>Flat</div>"
            f"<div class='activity-count'>{len(flats)} symbols</div>"
            f"{render_symbol_pills(flats)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<div class='chart-title'>Today Changes</div>", unsafe_allow_html=True)

    if latest_dt is not None:
        if prev_dt is not None:
            st.caption(
                f"Changes from {pd.to_datetime(prev_dt).strftime('%Y-%m-%d')} to {pd.to_datetime(latest_dt).strftime('%Y-%m-%d')}"
            )
        else:
            st.caption(f"Latest snapshot: {pd.to_datetime(latest_dt).strftime('%Y-%m-%d')}")

    if changes_df.empty:
        st.info("No position changes from the prior day.")
        return

    show_cols = ["symbol", "previous_stance", "current_stance", "change_date"]
    st.dataframe(convert_for_display(changes_df[show_cols]), use_container_width=True, hide_index=True)


def render_symbols(filtered: dict) -> None:
    st.markdown("<div class='section-title'>Symbols</div>", unsafe_allow_html=True)
    sym_ts = filtered["symbol_timeseries"]
    primary_fill_mode = get_primary_fill_mode(filtered)
    sym_col = get_symbol_col(sym_ts)
    if sym_ts.empty or sym_col is None:
        st.info("No symbol data in the selected date range.")
        return

    symbols = sorted(sym_ts[sym_col].dropna().astype(str).unique().tolist())
    symbol = st.selectbox("Symbol", symbols, index=0)

    one_ts = sym_ts[sym_ts[sym_col].astype(str) == symbol].sort_values(get_date_col(sym_ts) or sym_col)
    trades = filtered["trade_log"]
    trade_sym_col = get_symbol_col(trades)
    one_trades = trades[trades[trade_sym_col].astype(str) == symbol].copy() if (not trades.empty and trade_sym_col is not None) else pd.DataFrame()

    st.markdown(f"<div class='chart-title'>{symbol} Equity Curves</div>", unsafe_allow_html=True)
    st.plotly_chart(make_symbol_chart(symbol, one_ts, primary_fill_mode), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='chart-title'>Trade Log</div>", unsafe_allow_html=True)
    display_trade_table(one_trades)


def render_attribution(filtered: dict) -> None:
    st.markdown("<div class='section-title'>Attribution</div>", unsafe_allow_html=True)

    attribution = build_attribution_table(filtered)
    if attribution.empty:
        st.info("No attribution data is available for the selected date range.")
        return

    positive_count = int((attribution["alpha_pct"] > 0).sum())
    negative_count = int((attribution["alpha_pct"] < 0).sum())
    st.caption(
        f"Bars show each symbol's alpha versus its own buy-and-hold over the selected period. Positive alpha names: {positive_count} | Negative alpha names: {negative_count}"
    )
    st.plotly_chart(make_attribution_chart(attribution), use_container_width=True, config={"displayModeBar": False})


def render_trades(filtered: dict) -> None:
    st.markdown("<div class='section-title'>Trades</div>", unsafe_allow_html=True)
    trades = filtered["trade_log"]
    if trades.empty:
        st.info("No trades in the selected date range.")
        return

    closed = trades.copy()
    if "status" in closed.columns:
        closed = closed[closed["status"].astype(str).str.upper().eq("CLOSED")].copy()

    row = st.columns(3)
    row[0].markdown(kpi_card("Closed Trades", fmt_num(len(closed), 0)), unsafe_allow_html=True)
    avg_ret = pd.to_numeric(closed.get("trade_return_pct", pd.Series(dtype=float)), errors="coerce").mean() if not closed.empty else np.nan
    row[1].markdown(kpi_card("Avg Trade Return", fmt_pct(avg_ret)), unsafe_allow_html=True)
    realized = pd.to_numeric(closed.get("realized_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum() if not closed.empty else np.nan
    row[2].markdown(kpi_card("Closed Realized", fmt_money(realized, 0)), unsafe_allow_html=True)

    display_trade_table(trades)


def render_diagnostics(filtered: dict) -> None:
    st.markdown("<div class='section-title'>Diagnostics</div>", unsafe_allow_html=True)

    morning = filtered["morning_report"].copy()
    if not morning.empty:
        morning["model_saved_date"] = compute_model_saved_date(morning)
        keep = [
            c for c in [
                "symbol",
                "stance",
                "last_signal_action",
                "model_saved_date",
                "last_q_hold",
                "last_q_long",
                "last_q_close",
                "last_q_hold_masked",
                "last_q_long_masked",
                "last_q_close_masked",
            ] if c in morning.columns
        ]
        st.markdown("<div class='chart-title'>Model Snapshot</div>", unsafe_allow_html=True)
        st.dataframe(convert_for_display(morning[keep]), use_container_width=True, hide_index=True)

    agg = filtered["aggregate_timeseries"]
    if not agg.empty:
        rows = []
        for key, label in [
            ("close_t", "Close T0"),
            ("open_t_plus_1", "Close T+1 Open"),
            ("hl2_t_plus_1", "HL2 T+1"),
            ("open_t_plus_1_discount_only", "Open T+1 Discount Only"),
        ]:
            if key in agg.columns:
                ret = to_return_pct(agg[key]).iloc[-1]
                dd = drawdown_min_pct(agg[key])
                rows.append({"Fill Mode": label, "Return %": ret, "Max Drawdown %": dd})
        if rows:
            st.markdown("<div class='chart-title'>Fill Mode Snapshot</div>", unsafe_allow_html=True)
            fill_df = pd.DataFrame(rows)
            st.dataframe(convert_for_display(fill_df), use_container_width=True, hide_index=True)


def render_model_toggle() -> str:
    st.markdown("<div class='model-toggle-label'>Model</div>", unsafe_allow_html=True)

    if hasattr(st, "segmented_control"):
        selected = st.segmented_control(
            "Model",
            options=MODEL_OPTIONS,
            default="ALL SYMBOLS",
            label_visibility="collapsed",
            key="model_selector",
        )
        return str(selected or "ALL SYMBOLS")

    return st.radio(
        "Model",
        options=MODEL_OPTIONS,
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="model_selector",
    )


def main() -> None:
    inject_css()

    data_dir = detect_data_dir()
    try:
        bundle = load_bundle(str(data_dir))
    except Exception as exc:
        st.markdown(f"<div class='app-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        st.error(str(exc))
        st.info("Run the refactored live_report first so dashboard_data contains the summary and tables.")
        return

    selected_model = render_model_toggle()

    try:
        model_symbols = load_model_symbols(selected_model)
    except Exception as exc:
        st.markdown(f"<div class='app-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        st.error(str(exc))
        return

    model_bundle = apply_model_filter(bundle, model_symbols)

    date_source = model_bundle["aggregate_timeseries"] if not model_bundle["aggregate_timeseries"].empty else model_bundle["symbol_timeseries"]
    dcol = get_date_col(date_source)
    if dcol is None or date_source.empty:
        st.markdown(f"<div class='app-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        st.error(f"No usable dashboard data found for {selected_model}.")
        return

    min_dt = pd.to_datetime(date_source[dcol], errors="coerce").dropna().min()
    max_dt = pd.to_datetime(date_source[dcol], errors="coerce").dropna().max()
    if pd.isna(min_dt) or pd.isna(max_dt):
        st.markdown(f"<div class='app-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        st.error(f"Date range could not be determined for {selected_model}.")
        return

    report_start = model_bundle["summary"].get("report_start_date") or pd.to_datetime(min_dt).strftime("%Y-%m-%d")

    st.markdown("<div class='range-wrap'></div>", unsafe_allow_html=True)
    date_col, refresh_col, _ = st.columns([2.2, 0.9, 6.9], gap="small")
    with date_col:
        st.markdown("<div class='range-label'>Date Range</div>", unsafe_allow_html=True)
        date_range = st.date_input(
            "Date Range",
            value=(min_dt.date(), max_dt.date()),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
            label_visibility="collapsed",
        )
    with refresh_col:
        st.markdown("<div class='range-label'>&nbsp;</div>", unsafe_allow_html=True)
        if st.button("App Refresh", key="app_refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt = pd.Timestamp(date_range[0])
        end_dt = pd.Timestamp(date_range[1])
    else:
        start_dt = pd.Timestamp(min_dt.date())
        end_dt = pd.Timestamp(max_dt.date())

    filtered = filter_bundle(model_bundle, start_dt, end_dt)

    render_header(str(report_start), start_dt, end_dt)

    tabs = st.tabs(["Overview", "Activity", "Symbols", "Attribution", "Trades", "Diagnostics"])
    with tabs[0]:
        render_overview(filtered)
    with tabs[1]:
        render_activity(filtered)
    with tabs[2]:
        render_symbols(filtered)
    with tabs[3]:
        render_attribution(filtered)
    with tabs[4]:
        render_trades(filtered)
    with tabs[5]:
        render_diagnostics(filtered)


if __name__ == "__main__":
    main()
