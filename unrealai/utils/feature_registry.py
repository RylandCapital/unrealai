"""Shared model feature registry for training and live inference."""

FUNDAMENTAL_SYMBOLS = [
    "#SPXADR",
    "#NDXADR",
    "#SPXMCOSC",
    "#NDXMCOSC",
    "#NDXZWBT",
    "#SPXZWBT",
    "#OEX%MA50",
    "#OEX%MA200",
    "#M2FED3",
    "#M2FED2",
    "#CBOEPCE",
    "$VIX",
]

PRICE_FEATURE_COLS = [
    "dist_sma_21",
    "dist_sma_55",
    "dist_sma_200",
    "sma_21_slope_10",
    "sma_55_slope_10",
    "sma_200_slope_20",
    "ma_spread_21_55",
    "ma_spread_21_200",
    "ret_10",
    "ret_21",
    "ret_63",
    "ret_126",
    "mom_score_21",
    "mom_score_63",
    "mom_score_126",
    "vol_21",
    "vol_63",
    "atr_14_pct",
    "gap_pct",
    "close_loc_in_range",
    "range_pos_20",
    "range_pos_60",
    "range_pos_252",
    "drawdown_63",
    "drawdown_252",
    "bb_width_20",
    "bb_pos_20",
    "atr_regime_63",
    "range_compression_20",
    "nr7_flag",
    "dist_prev_high_20",
    "dist_prev_high_60",
    "dist_prev_high_252",
    "new_high_20",
    "new_high_60",
    "new_high_252",
    "volume_rel_20",
    "volume_rel_63",
    "dollar_volume_rel_20",
    "up_down_vol_ratio_20",
    "obv_z_63",
    "adl_z_63",
    "rsi_21_raw",
    "rsi_21_delta",
    "macd_hist",
    "ppo_hist",
]

SPY_FEATURE_COLS = [
    "spy_dist_sma_21",
    "spy_dist_sma_55",
    "spy_dist_sma_200",
    "spy_ret_21",
    "spy_ret_63",
    "spy_vol_21",
    "spy_drawdown_252",
]

SECTOR_FEATURE_COLS = [
    "sector_dist_sma_21",
    "sector_ret_63",
    "sector_vol_21",
    "sector_drawdown_252",
]

RELATIVE_FEATURE_COLS = [
    "rel_ret_21_vs_spy",
    "rel_ret_63_vs_spy",
    "rel_ret_126_vs_spy",
    "price_to_spy_ratio_ret_21",
    "price_to_spy_ratio_ret_63",
    "resid_mom_21_vs_spy",
    "resid_mom_63_vs_spy",
    "idio_vol_63_vs_spy",
    "rel_ret_21_vs_sector",
    "rel_ret_63_vs_sector",
    "rel_ret_126_vs_sector",
    "price_to_sector_ratio_ret_21",
    "price_to_sector_ratio_ret_63",
    "resid_mom_21_vs_sector",
    "resid_mom_63_vs_sector",
]

INTERNAL_FEATURE_COLS = [
    "#SPXADR_z_63",
    "#NDXADR_z_63",
    "#SPXMCOSC_z_63",
    "#NDXMCOSC_z_63",
    "#SPXZWBT_z_63",
    "#OEX%MA50_z_63",
    "#CBOEPCE_z_63",
    "$VIX_close",
    "$VIX_z_63",
]

REGIME_FEATURE_COLS = [
    "trend_strength_21_200",
    "vol_regime_ratio",
    "atr_to_vol_ratio",
    "stock_vs_market_drawdown_252",
    "stock_vs_sector_drawdown_252",
]

FEATURE_COLS = (
    PRICE_FEATURE_COLS
    + SPY_FEATURE_COLS
    + SECTOR_FEATURE_COLS
    + RELATIVE_FEATURE_COLS
    + INTERNAL_FEATURE_COLS
    + REGIME_FEATURE_COLS
)
