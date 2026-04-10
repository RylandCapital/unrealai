import numpy as np
import pandas as pd

BACKFILLABLE_FEATURE_COLS = [
    "bb_width_20",
    "bb_pos_20",
    "atr_regime_63",
    "range_compression_20",
    "nr7_flag",
]


def first_existing_col(cols, candidates):
    lower_map = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        hit = lower_map.get(str(cand).lower())
        if hit is not None:
            return hit
    return None


def ensure_backfilled_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in BACKFILLABLE_FEATURE_COLS if col not in df.columns]
    if not missing:
        return df

    close_col = first_existing_col(df.columns, ["adjusted_close", "Close", "close"])
    high_col = first_existing_col(df.columns, ["High", "high", "adjusted_high", "adj_high"])
    low_col = first_existing_col(df.columns, ["Low", "low", "adjusted_low", "adj_low"])

    if close_col is None:
        raise ValueError("Cannot backfill regime features without an adjusted close column.")

    requires_hl = any(col in missing for col in ["atr_regime_63", "nr7_flag"])
    if requires_hl and (high_col is None or low_col is None):
        raise ValueError(
            "Cannot backfill regime features without high/low columns. "
            f"Missing features: {missing}"
        )

    out = df.copy()
    close = pd.to_numeric(out[close_col], errors="coerce")

    if "bb_width_20" in missing or "bb_pos_20" in missing:
        bb_mid_20 = close.rolling(20).mean()
        bb_std_20 = close.rolling(20).std(ddof=0)
        bb_upper_20 = bb_mid_20 + 2.0 * bb_std_20
        bb_lower_20 = bb_mid_20 - 2.0 * bb_std_20

        if "bb_width_20" in missing:
            out["bb_width_20"] = ((bb_upper_20 - bb_lower_20) / bb_mid_20.replace(0, np.nan)).astype(np.float32)
        if "bb_pos_20" in missing:
            out["bb_pos_20"] = ((close - bb_lower_20) / (bb_upper_20 - bb_lower_20).replace(0, np.nan)).astype(np.float32)

    if "range_compression_20" in missing:
        roll_high_20 = close.rolling(20).max()
        roll_low_20 = close.rolling(20).min()
        out["range_compression_20"] = ((roll_high_20 - roll_low_20) / close.replace(0, np.nan)).astype(np.float32)

    if requires_hl:
        high = pd.to_numeric(out[high_col], errors="coerce")
        low = pd.to_numeric(out[low_col], errors="coerce")
        prev_close = close.shift(1)
        true_range = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        if "atr_regime_63" in missing:
            atr_14 = true_range.rolling(14).mean()
            atr_14_pct = atr_14 / close.replace(0, np.nan)
            out["atr_regime_63"] = (atr_14_pct / atr_14_pct.rolling(63).mean().replace(0, np.nan)).astype(np.float32)

        if "nr7_flag" in missing:
            out["nr7_flag"] = (true_range <= true_range.rolling(7).min()).astype(np.float32)

    return out
