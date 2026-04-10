import warnings
import norgatedata
import datetime as dt
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from dateutil.relativedelta import relativedelta
from scipy.signal import argrelextrema


# =============================================================================
# CONFIG
# =============================================================================
EXCEL_PATH = r'P:\10_CWP Trade Department\_Matrix_\code_outputs\grip_momo\grip_allocation.xlsx'
PDF_PATH = r"C:\Users\rmathews\Downloads\vwap_report.pdf"

YEARS_BACK = 10
DAILY_LOOKBACK_BARS = 252       # 1Y daily panel for Overwatch point-in-time charting
RS_LOOKBACK_BARS = 63           # ~3 months
SWING_ORDER = 5                 # local extrema sensitivity
HIGHVOL_SKIP_BARS = 125         # ignore earliest part of 1Y window for high-volume anchor
EXTRA_TICKERS = ['QQQ', 'SPY', 'IWM']

# Optional fallback if Excel does not have a usable Sector column
TICKER_TO_SECTOR_ETF = {
    # 'NVDA': 'XLK',
    # 'MSFT': 'XLK',
    # 'AAPL': 'XLK',
    # 'AMZN': 'XLY',
    # 'META': 'XLC',
    # 'GOOGL': 'XLC',
    # 'JPM': 'XLF',
    # 'XOM': 'XLE',
}

SECTOR_NAME_TO_ETF = {
    'COMMUNICATION SERVICES': 'XLC',
    'COMMUNICATION': 'XLC',
    'CONSUMER DISCRETIONARY': 'XLY',
    'CONSUMER STAPLES': 'XLP',
    'ENERGY': 'XLE',
    'FINANCIALS': 'XLF',
    'FINANCIAL': 'XLF',
    'HEALTH CARE': 'XLV',
    'HEALTHCARE': 'XLV',
    'INDUSTRIALS': 'XLI',
    'INDUSTRIAL': 'XLI',
    'INFORMATION TECHNOLOGY': 'XLK',
    'TECHNOLOGY': 'XLK',
    'MATERIALS': 'XLB',
    'REAL ESTATE': 'XLRE',
    'UTILITIES': 'XLU',
}

PRICE_CACHE = {}
_SECTOR_LOOKUP_CACHE = None


# =============================================================================
# STYLE
# =============================================================================
my_marketcolors = mpf.make_marketcolors(
    up='green',
    down='red',
    edge='inherit',
    wick='inherit',
    volume='inherit'
)

dark_style_custom = mpf.make_mpf_style(
    base_mpf_style='nightclouds',
    marketcolors=my_marketcolors
)

INFO_BOX_PROPS = dict(
    boxstyle='round',
    facecolor='black',
    edgecolor='white',
    alpha=0.82,
    linewidth=0.8
)


# =============================================================================
# HELPERS
# =============================================================================
def normalize_end_date(end=None):
    if end is None:
        return pd.Timestamp(dt.datetime.today().date())
    return pd.Timestamp(end).normalize()


def dedupe_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def fmt_px(x):
    return "N/A" if x is None or pd.isna(x) else f"{x:,.2f}"


def fmt_pct(x):
    return "N/A" if x is None or pd.isna(x) else f"{x:+.2%}"


def fmt_date(x):
    if x is None or pd.isna(x):
        return "N/A"
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def relation_text(last_close, ref_value):
    if ref_value is None or pd.isna(ref_value) or ref_value == 0:
        return "N/A"
    pct = (last_close / ref_value) - 1.0
    side = "ABOVE" if pct >= 0 else "BELOW"
    return f"{side} {pct:+.2%}"


def normalize_sector_name(sector_value):
    if sector_value is None or pd.isna(sector_value):
        return None
    text = str(sector_value).strip().upper()
    text = " ".join(text.split())
    return text if text else None


def resolve_sector_benchmark(ticker, sector_value=None):
    """
    Priority:
    1) Sector column from Excel -> mapped ETF
    2) Manual hardcoded fallback dict
    3) None
    """
    ticker = str(ticker).strip().upper()

    sector_name = normalize_sector_name(sector_value)
    if sector_name in SECTOR_NAME_TO_ETF:
        sector_etf = SECTOR_NAME_TO_ETF[sector_name]
        return None if sector_etf == ticker else sector_etf

    sector_etf = TICKER_TO_SECTOR_ETF.get(ticker)
    if sector_etf == ticker:
        return None
    return sector_etf


def standardize_ohlcv(df):
    """
    Force standard columns: Date/Open/High/Low/Close/Volume
    """
    if df is None or df.empty:
        return None

    df = df.copy().reset_index()

    if 'Date' not in df.columns:
        first_col = df.columns[0]
        df.rename(columns={first_col: 'Date'}, inplace=True)

    rename_map = {}
    for col in df.columns:
        c = str(col).strip().lower()
        if c == 'date':
            rename_map[col] = 'Date'
        elif c == 'open':
            rename_map[col] = 'Open'
        elif c == 'high':
            rename_map[col] = 'High'
        elif c == 'low':
            rename_map[col] = 'Low'
        elif c == 'close':
            rename_map[col] = 'Close'
        elif c == 'volume':
            rename_map[col] = 'Volume'

    df.rename(columns=rename_map, inplace=True)

    required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns after standardization: {missing}")
        return None

    df = df[required].copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    df.sort_values('Date', inplace=True)
    df.drop_duplicates(subset='Date', keep='last', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def fetch_price_history(symbol, years_back=YEARS_BACK, end=None):
    """
    Pull adjusted price history from Norgate and truncate it to the requested
    point-in-time end date.
    """
    symbol = str(symbol).strip().upper()
    end_ts = normalize_end_date(end)
    start_ts = end_ts - relativedelta(years=years_back)

    cache_key = (symbol, int(years_back), end_ts.strftime("%Y-%m-%d"))
    if cache_key in PRICE_CACHE:
        return PRICE_CACHE[cache_key].copy()

    try:
        raw = norgatedata.price_timeseries(
            symbol,
            stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.TOTALRETURN,
            padding_setting=norgatedata.PaddingType.NONE,
            start_date=pd.Timestamp(start_ts.date()),
            timeseriesformat='pandas-dataframe'
        )
    except Exception as e:
        print(f"Failed to retrieve data for {symbol}: {e}")
        return None

    df = standardize_ohlcv(raw)
    if df is None or df.empty:
        print(f"No valid OHLCV data returned for {symbol}")
        return None

    df = df[(df['Date'] >= start_ts) & (df['Date'] <= end_ts)].copy()
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        print(f"No data left for {symbol} after truncating to {end_ts.date()}")
        return None

    PRICE_CACHE[cache_key] = df.copy()
    return df


def resample_ohlc(df, freq):
    """
    Resample daily OHLCV to a new frequency.
    """
    if df is None or df.empty:
        return None

    df2 = df.copy()
    if 'Date' in df2.columns:
        df2 = df2.set_index('Date')

    out = df2.resample(freq).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    return out


def anchored_vwap(df, start_loc, price_col='Close', volume_col='Volume'):
    """
    Return full-length anchored VWAP series with NaN before anchor.
    """
    result = pd.Series(np.nan, index=df.index, dtype=float)

    if start_loc is None or start_loc >= len(df):
        return result

    pxv = (df[price_col].iloc[start_loc:] * df[volume_col].iloc[start_loc:]).cumsum()
    vol = df[volume_col].iloc[start_loc:].cumsum()
    vwap = pxv / vol.replace(0, np.nan)

    result.iloc[start_loc:] = vwap.values
    return result


def calculate_vwap_indicators(df_daily, swing_order=SWING_ORDER):
    """
    Anchored VWAP overlays from:
      - highest local swing high
      - lowest local swing low
      - highest-volume node
    """
    df = df_daily.copy().reset_index(drop=True)

    if len(df) < 10:
        df['vwap_maxloc'] = np.nan
        df['vwap_minloc'] = np.nan
        df['vwap_highvol'] = np.nan
        return df, {}

    close_vals = df['Close'].values

    max_idx = argrelextrema(close_vals, np.greater_equal, order=swing_order)[0]
    min_idx = argrelextrema(close_vals, np.less_equal, order=swing_order)[0]

    if len(max_idx) == 0:
        max_anchor_loc = int(df['Close'].idxmax())
    else:
        max_anchor_loc = int(max_idx[np.argmax(df['Close'].iloc[max_idx].values)])

    if len(min_idx) == 0:
        min_anchor_loc = int(df['Close'].idxmin())
    else:
        min_anchor_loc = int(min_idx[np.argmin(df['Close'].iloc[min_idx].values)])

    if len(df) > HIGHVOL_SKIP_BARS:
        vol_slice = df['Volume'].iloc[HIGHVOL_SKIP_BARS:]
    else:
        vol_slice = df['Volume']

    if vol_slice.isna().all() or (vol_slice.fillna(0) <= 0).all():
        highvol_anchor_loc = 0
    else:
        highvol_anchor_loc = int(vol_slice.idxmax())

    df['vwap_maxloc'] = anchored_vwap(df, max_anchor_loc)
    df['vwap_minloc'] = anchored_vwap(df, min_anchor_loc)
    df['vwap_highvol'] = anchored_vwap(df, highvol_anchor_loc)

    meta = {
        'last_close': float(df['Close'].iloc[-1]),

        'max_anchor_loc': max_anchor_loc,
        'max_anchor_date': df.loc[max_anchor_loc, 'Date'],
        'max_anchor_price': float(df.loc[max_anchor_loc, 'Close']),
        'vwap_maxloc_last': float(df['vwap_maxloc'].dropna().iloc[-1]) if df['vwap_maxloc'].notna().any() else np.nan,

        'min_anchor_loc': min_anchor_loc,
        'min_anchor_date': df.loc[min_anchor_loc, 'Date'],
        'min_anchor_price': float(df.loc[min_anchor_loc, 'Close']),
        'vwap_minloc_last': float(df['vwap_minloc'].dropna().iloc[-1]) if df['vwap_minloc'].notna().any() else np.nan,

        'highvol_anchor_loc': highvol_anchor_loc,
        'highvol_anchor_date': df.loc[highvol_anchor_loc, 'Date'],
        'highvol_anchor_price': float(df.loc[highvol_anchor_loc, 'Close']),
        'vwap_highvol_last': float(df['vwap_highvol'].dropna().iloc[-1]) if df['vwap_highvol'].notna().any() else np.nan,
    }

    return df, meta


def close_series_from_df(df):
    """
    Return a clean Date-indexed close series.
    """
    if df is None or df.empty:
        return None

    if 'Date' in df.columns:
        s = df[['Date', 'Close']].copy()
        s['Date'] = pd.to_datetime(s['Date'], errors='coerce')
        s.dropna(subset=['Date', 'Close'], inplace=True)
        s.sort_values('Date', inplace=True)
        s.drop_duplicates(subset='Date', keep='last', inplace=True)
        s.set_index('Date', inplace=True)
        return s['Close'].astype(float)

    if isinstance(df.index, pd.DatetimeIndex) and 'Close' in df.columns:
        return df['Close'].astype(float).sort_index()

    return None


def compute_relative_strength(stock_df, benchmark_symbol, lookback=RS_LOOKBACK_BARS, end=None):
    """
    3M RS = stock return relative to benchmark return over ~63 trading days,
    using the same as-of end date for both series.
    """
    benchmark_df = fetch_price_history(benchmark_symbol, years_back=YEARS_BACK, end=end)
    if benchmark_df is None or benchmark_df.empty:
        return None

    stock_close = close_series_from_df(stock_df)
    bench_close = close_series_from_df(benchmark_df)

    if stock_close is None or bench_close is None:
        return None

    aligned = pd.concat(
        [stock_close.rename('stock'), bench_close.rename('bench')],
        axis=1,
        join='inner'
    ).dropna()

    if len(aligned) <= lookback:
        return None

    stock_ret = aligned['stock'].pct_change(lookback).iloc[-1]
    bench_ret = aligned['bench'].pct_change(lookback).iloc[-1]

    if pd.isna(stock_ret) or pd.isna(bench_ret):
        rel_ret = np.nan
    else:
        rel_ret = (1 + stock_ret) / (1 + bench_ret) - 1

    return {
        'benchmark': benchmark_symbol,
        'stock_ret': float(stock_ret) if pd.notna(stock_ret) else np.nan,
        'benchmark_ret': float(bench_ret) if pd.notna(bench_ret) else np.nan,
        'relative_ret': float(rel_ret) if pd.notna(rel_ret) else np.nan,
    }


def compute_ytd_return(df_daily):
    if df_daily is None or df_daily.empty:
        return np.nan

    df = df_daily.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    last_close = float(df['Close'].iloc[-1])
    last_date = pd.Timestamp(df['Date'].iloc[-1])
    year_start = pd.Timestamp(year=last_date.year, month=1, day=1)

    prev_year_rows = df[df['Date'] < year_start]
    if not prev_year_rows.empty:
        base_close = float(prev_year_rows['Close'].iloc[-1])
    else:
        curr_year_rows = df[df['Date'] >= year_start]
        if curr_year_rows.empty:
            return np.nan
        base_close = float(curr_year_rows['Close'].iloc[0])

    if base_close == 0:
        return np.nan

    return (last_close / base_close) - 1.0


def build_vwap_box_text(stock, meta):
    last_close = meta.get('last_close', np.nan)
    vwap_max = meta.get('vwap_maxloc_last', np.nan)
    vwap_min = meta.get('vwap_minloc_last', np.nan)
    vwap_hv = meta.get('vwap_highvol_last', np.nan)

    lines = [
        f"{stock} AI READOUT",
        f"Last Close   : {fmt_px(last_close)}",
        "",
        f"High VWAP    : {fmt_px(vwap_max)}",
        f"              {relation_text(last_close, vwap_max)}",
        f"Low VWAP     : {fmt_px(vwap_min)}",
        f"              {relation_text(last_close, vwap_min)}",
        f"HighVol VWAP : {fmt_px(vwap_hv)}",
        f"              {relation_text(last_close, vwap_hv)}",
        "",
        f"High Anchor  : {fmt_date(meta.get('max_anchor_date'))}",
        f"Low Anchor   : {fmt_date(meta.get('min_anchor_date'))}",
        f"Vol Anchor   : {fmt_date(meta.get('highvol_anchor_date'))}",
    ]
    return "\n".join(lines)


def build_rs_box_text(rs_spy, rs_sector):
    lines = ["RELATIVE STRENGTH"]

    if rs_spy is None:
        lines.extend([
            "3M Return    : N/A",
            "3M vs SPY    : N/A",
        ])
    else:
        lines.extend([
            f"3M Return    : {fmt_pct(rs_spy.get('stock_ret'))}",
            f"3M vs SPY    : {fmt_pct(rs_spy.get('relative_ret'))}",
        ])

    if rs_sector is None:
        lines.extend([
            "Sector Proxy : N/A",
            "3M vs Sector : N/A",
        ])
    else:
        lines.extend([
            f"Sector Proxy : {rs_sector.get('benchmark', 'N/A')}",
            f"3M vs Sector : {fmt_pct(rs_sector.get('relative_ret'))}",
        ])

    return "\n".join(lines)


def load_sector_lookup():
    global _SECTOR_LOOKUP_CACHE
    if _SECTOR_LOOKUP_CACHE is not None:
        return _SECTOR_LOOKUP_CACHE

    lookup = {}
    try:
        warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
        raw_df = pd.read_excel(
            EXCEL_PATH,
            header=2,
            sheet_name=1
        )

        if 'Ticker' in raw_df.columns:
            raw_df = raw_df.copy()
            raw_df = raw_df[raw_df['Ticker'].notna()].copy()
            raw_df['Ticker'] = raw_df['Ticker'].astype(str).str.strip().str.upper()
            raw_df = raw_df[
                (raw_df['Ticker'] != '') &
                (raw_df['Ticker'] != 'NAN') &
                (raw_df['Ticker'] != 'NONE')
            ].copy()

            for _, row in raw_df.iterrows():
                ticker = row['Ticker']
                sector_val = row['Sector'] if 'Sector' in raw_df.columns else None
                lookup[ticker] = resolve_sector_benchmark(ticker, sector_val)
    except Exception:
        lookup = {}

    _SECTOR_LOOKUP_CACHE = lookup
    return _SECTOR_LOOKUP_CACHE


def infer_sector_benchmark_for_symbol(symbol):
    symbol = str(symbol).strip().upper()
    return load_sector_lookup().get(symbol, resolve_sector_benchmark(symbol, None))


# =============================================================================
# PLOTTING
# =============================================================================
def plot_three_charts_on_one_page(stock, sector_benchmark=None, end=None):
    """
    Point-in-time safe chart:
      - Top-left: 10Y monthly candlestick chart as of end date
      - Top-right: 10Y weekly candlestick chart as of end date
      - Bottom: 1Y daily candlestick chart ending exactly on end date
      - Top-left info box: AI/VWAP readout
      - Bottom-left info box: Relative strength
    """
    stock = str(stock).strip().upper()
    end_ts = normalize_end_date(end)

    if sector_benchmark is None:
        sector_benchmark = infer_sector_benchmark_for_symbol(stock)

    df_10yr_daily = fetch_price_history(stock, years_back=YEARS_BACK, end=end_ts)
    if df_10yr_daily is None or df_10yr_daily.empty:
        print(f"Data retrieval issue for {stock}")
        return None

    df_monthly_10yr = resample_ohlc(df_10yr_daily, freq='M')
    df_weekly_10yr = resample_ohlc(df_10yr_daily, freq='W-FRI')

    if df_monthly_10yr is None or df_monthly_10yr.empty:
        print(f"Monthly resample failed for {stock}")
        return None

    if df_weekly_10yr is None or df_weekly_10yr.empty:
        print(f"Weekly resample failed for {stock}")
        return None

    df_daily_1yr = df_10yr_daily.copy().set_index('Date').iloc[-DAILY_LOOKBACK_BARS:].copy()
    if df_daily_1yr.empty:
        print(f"Not enough 1Y daily data for {stock}")
        return None

    df_daily_1yr_reset = df_daily_1yr.reset_index()
    df_daily_plot, vwap_meta = calculate_vwap_indicators(df_daily_1yr_reset)
    df_daily_plot.set_index('Date', inplace=True)

    rs_spy = compute_relative_strength(
        df_10yr_daily,
        'SPY',
        lookback=RS_LOOKBACK_BARS,
        end=end_ts
    )

    rs_sector = None
    if sector_benchmark:
        rs_sector = compute_relative_strength(
            df_10yr_daily,
            sector_benchmark,
            lookback=RS_LOOKBACK_BARS,
            end=end_ts
        )

    fig = mpf.figure(figsize=(14, 10), style=dark_style_custom)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])

    ax_monthly = fig.add_subplot(gs[0, 0])
    ax_weekly = fig.add_subplot(gs[0, 1])
    ax_daily = fig.add_subplot(gs[1, :])

    apds = [
        mpf.make_addplot(df_daily_plot['vwap_maxloc'], ax=ax_daily, color='lightskyblue', width=1.1),
        mpf.make_addplot(df_daily_plot['vwap_minloc'], ax=ax_daily, color='violet', width=1.1),
        mpf.make_addplot(df_daily_plot['vwap_highvol'], ax=ax_daily, color='yellow', width=1.1),
    ]

    mpf.plot(
        df_monthly_10yr,
        type='candle',
        ax=ax_monthly,
        style=dark_style_custom,
        volume=False,
        show_nontrading=False
    )
    ax_monthly.set_title(f"{stock} 10Y Monthly | As Of {end_ts.strftime('%Y-%m-%d')}")

    mpf.plot(
        df_weekly_10yr,
        type='candle',
        ax=ax_weekly,
        style=dark_style_custom,
        volume=False,
        show_nontrading=False
    )
    ax_weekly.set_title(f"{stock} 10Y Weekly | As Of {end_ts.strftime('%Y-%m-%d')}")

    mpf.plot(
        df_daily_plot,
        type='candle',
        ax=ax_daily,
        addplot=apds,
        style=dark_style_custom,
        volume=False,
        show_nontrading=False
    )

    last_close = vwap_meta.get('last_close', np.nan)
    ytd_ret = compute_ytd_return(df_10yr_daily)

    ax_daily.set_title(
        f"{stock} 1Y Daily w/ Anchored VWAP | "
        f"Last: {fmt_px(last_close)} | "
        f"YTD: {fmt_pct(ytd_ret)} | "
        f"As Of {end_ts.strftime('%Y-%m-%d')}"
    )

    if pd.notna(last_close):
        ax_daily.axhline(last_close, color='white', linestyle='--', linewidth=0.8, alpha=0.35)

    vwap_box = build_vwap_box_text(stock, vwap_meta)
    rs_box = build_rs_box_text(rs_spy, rs_sector)

    # TOP LEFT: VWAP / AI readout
    ax_daily.text(
        0.01, 0.99, vwap_box,
        transform=ax_daily.transAxes,
        va='top', ha='left',
        fontsize=8.8,
        fontfamily='monospace',
        color='white',
        bbox=INFO_BOX_PROPS,
        zorder=20
    )

    # BOTTOM LEFT: Relative strength
    ax_daily.text(
        0.01, 0.02, rs_box,
        transform=ax_daily.transAxes,
        va='bottom', ha='left',
        fontsize=8.8,
        fontfamily='monospace',
        color='white',
        bbox=INFO_BOX_PROPS,
        zorder=20
    )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.06, hspace=0.20, wspace=0.12)
    return fig


def build_fig(symbol, end=None, sector_benchmark=None):
    """
    Compatibility wrapper for Overwatch.
    The main trading script calls:
        build_fig(symbol, end=env.current_datetime.strftime("%Y-%m-%d"))
    """
    return plot_three_charts_on_one_page(
        stock=symbol,
        sector_benchmark=sector_benchmark,
        end=end
    )


# =============================================================================
# MAIN
# =============================================================================
def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    try:
        raw_df = pd.read_excel(
            EXCEL_PATH,
            header=2,
            sheet_name=1
        )
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return

    if 'Ticker' not in raw_df.columns:
        print("Excel sheet is missing required 'Ticker' column.")
        return

    raw_df = raw_df.copy()
    raw_df = raw_df[raw_df['Ticker'].notna()].copy()
    raw_df['Ticker'] = raw_df['Ticker'].astype(str).str.strip().str.upper()
    raw_df = raw_df[
        (raw_df['Ticker'] != '') &
        (raw_df['Ticker'] != 'NAN') &
        (raw_df['Ticker'] != 'NONE')
    ].copy()

    sector_lookup = {}
    for _, row in raw_df.iterrows():
        ticker = row['Ticker']
        sector_val = row['Sector'] if 'Sector' in raw_df.columns else None
        sector_lookup[ticker] = resolve_sector_benchmark(ticker, sector_val)

    list_stocks = dedupe_preserve_order(raw_df['Ticker'].tolist() + EXTRA_TICKERS)

    with PdfPages(PDF_PATH) as pdf:
        for stock in list_stocks:
            if pd.isna(stock) or str(stock).strip().upper() in {'', 'NAN', 'NONE'}:
                print(f"Skipping invalid ticker: {stock}")
                continue

            stock = str(stock).strip().upper()
            print(f"Processing {stock}...")

            fig = plot_three_charts_on_one_page(
                stock,
                sector_benchmark=sector_lookup.get(stock),
                end=None
            )

            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)
            else:
                print(f"Skipping {stock} due to data retrieval or plotting issues.")

    print(f"PDF saved to: {PDF_PATH}")


if __name__ == '__main__':
    main()