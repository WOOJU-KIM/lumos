import os
import sys
import sqlite3
import config
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.heterogeneous_models import (
    OrderFlowImbalanceModel,
    TDATopologyModel,
    StateSpaceKalmanModel,
    CrossAssetDislocationModel
)
from core.ml_engine import MLFeatureEngine

conn = sqlite3.connect('data/market_data.db')
long_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.BASE_ASSET_LONG AND timeframe='15m' ORDER BY datetime", conn)
short_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.BASE_ASSET_SHORT AND timeframe='15m' ORDER BY datetime", conn)
nvda_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='NVDA' AND timeframe='15m' ORDER BY datetime", conn)
qqq_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='QQQ' AND timeframe='15m' ORDER BY datetime", conn)
macro_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.MACRO_TREND_SYMBOL AND timeframe='15m' ORDER BY datetime", conn)
vix_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='^VIX' AND timeframe='15m' ORDER BY datetime", conn)
tnx_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='^TNX' AND timeframe='15m' ORDER BY datetime", conn)
conn.close()

for df in [long_df, short_df, nvda_df, qqq_df, macro_df, vix_df, tnx_df]:
    df['Datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].str.slice(0, 10)
    df.set_index('Datetime', inplace=True)

# Test each expert model signal frequency
m2 = OrderFlowImbalanceModel()
m3 = TDATopologyModel()
m4 = StateSpaceKalmanModel()
m5 = CrossAssetDislocationModel()

sig_m2 = 0
sig_m3 = 0
sig_m4 = 0
sig_m5 = 0

for i in range(50, len(long_df)):
    sub_long = long_df.iloc[:i]
    sub_nvda = nvda_df.iloc[:i]
    sub_qqq = qqq_df.iloc[:i]
    sub_macro = macro_df.iloc[:i]
    sub_vix = vix_df.iloc[:i]
    sub_tnx = tnx_df.iloc[:i]
    
    # M2
    cvd_df = m2.compute_cvd(sub_long.tail(30))
    s2, c2, _ = m2.predict_signal(cvd_df.iloc[-1])
    if s2 != 0:
        sig_m2 += 1
        
    # M3
    s3, c3, _ = m3.predict_signal(sub_long.tail(30))
    if s3 != 0:
        sig_m3 += 1
        
    # M4
    s4, c4, _ = m4.predict_signal(sub_long['Close'].tail(30).values)
    if s4 != 0:
        sig_m4 += 1
        
    # M5
    s_ret = (sub_long['Close'].iloc[-1] / sub_long['Close'].iloc[-5] - 1.0)
    n_ret = (sub_nvda['Close'].iloc[-1] / sub_nvda['Close'].iloc[-5] - 1.0)
    q_ret = (sub_qqq['Close'].iloc[-1] / sub_qqq['Close'].iloc[-5] - 1.0)
    sx_ret = (sub_macro['Close'].iloc[-1] / sub_macro['Close'].iloc[-5] - 1.0)
    v_ret = (sub_vix['Close'].iloc[-1] / sub_vix['Close'].iloc[-5] - 1.0)
    t_ret = (sub_tnx['Close'].iloc[-1] / sub_tnx['Close'].iloc[-5] - 1.0)
    s5, c5, _ = m5.predict_signal(s_ret, n_ret, q_ret, sx_ret, v_ret, t_ret)
    if s5 != 0:
        sig_m5 += 1

print(f"Total bars: {len(long_df)-50}")
print(f"M2 signals: {sig_m2}, M3 signals: {sig_m3}, M4 signals: {sig_m4}, M5 signals: {sig_m5}")
