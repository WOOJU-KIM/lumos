# -*- coding: utf-8 -*-
import os, sys, time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.data_lake import MarketDataLake
from core.moe_orchestrator import MoEMetaOrchestrator
import core.market_calendar
from config import GBDT_CONFIDENCE_THRESHOLD

dl = MarketDataLake()
moe = MoEMetaOrchestrator(confidence_threshold=GBDT_CONFIDENCE_THRESHOLD, gbdt_threshold=GBDT_CONFIDENCE_THRESHOLD)

start_ny = datetime(2026, 9, 21, 9, 30, tzinfo=ZoneInfo('America/New_York'))
end_ny = datetime(2026, 9, 21, 16, 0, tzinfo=ZoneInfo('America/New_York'))

original_get_market_status = core.market_calendar.USMarketCalendar.get_market_status
def mock_get_market_status(*args, **kwargs):
    res = original_get_market_status(*args, **kwargs)
    res["is_open"] = True
    res["is_trading_allowed"] = True
    res["is_entry_allowed"] = True
    return res
core.market_calendar.USMarketCalendar.get_market_status = mock_get_market_status

curr_ny = start_ny
print('시작')
while curr_ny <= end_ny:
    if curr_ny.minute % 15 == 0:
        kst = curr_ny.astimezone(ZoneInfo('Asia/Seoul'))
        dt_str = curr_ny.strftime('%Y-%m-%d %H:%M:%S')
        
        df_15m = dl.load_candles('TQQQ', '15m', end_dt=dt_str)
        if df_15m.empty:
            curr_ny += timedelta(minutes=5)
            continue
            
        cur_tqqq = df_15m.iloc[-1]['Close']
        cur_sqqq = dl.load_candles('SQQQ', '15m', end_dt=dt_str).iloc[-1]['Close']
        live_p = {'TQQQ': float(cur_tqqq), 'SQQQ': float(cur_sqqq)}
        
        with patch('core.market_calendar.datetime') as m_dt, patch('core.moe_orchestrator.datetime') as m_moe_dt:
            m_dt.now.return_value = curr_ny
            m_moe_dt.now.return_value = kst
            res = moe.evaluate_dual_filter_signal(df_15m, live_p)
            
        is_appr = res.get('is_approved', False)
        direction = res.get('direction', 'NONE')
        
        print(f"[{curr_ny.strftime('%H:%M')}] Evaluated: {is_appr} {direction}")
        
    curr_ny += timedelta(minutes=5)
print('완료')
