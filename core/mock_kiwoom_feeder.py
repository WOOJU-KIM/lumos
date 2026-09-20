import json
import logging
import time
from typing import Callable, Dict, Any, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import threading
import sqlite3
from zoneinfo import ZoneInfo

from core.kiwoom_broker import KiwoomBroker
from core.data_lake import MarketDataLake
from config import DATA_DIR

logger = logging.getLogger("MockKiwoomFeeder")

class MockKiwoomFeeder:
    def __init__(self, broker: Optional[KiwoomBroker] = None, target_date: str = None):
        self.broker = broker
        self.is_finished = False
        
        if target_date:
            self.target_date = target_date
        else:
            conn = sqlite3.connect("data/market_data.db")
            df_date = pd.read_sql("SELECT date(datetime) as dt FROM market_candles WHERE symbol='TQQQ' ORDER BY datetime DESC LIMIT 1", conn)
            self.target_date = df_date.iloc[0]['dt']
            conn.close()

        self._callbacks: List[Callable[[str, float, Dict[str, Any]], None]] = []
        self._execution_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        self.subscribed_symbols = ["TQQQ", "SQQQ", "SOXX", "QQQ", "NVDA", "VIXY", "IEF"]
        self.data_lake = MarketDataLake()
        
        self.current_mock_time: Optional[datetime] = None
        self._latest_prices: Dict[str, float] = {}
        
        self._load_historical_data()
        
        self._original_load_candles = self.data_lake.load_candles
        self.data_lake.load_candles = self._mock_load_candles

    def _load_historical_data(self):
        self.mock_data = {}
        target_dt = datetime.strptime(self.target_date, "%Y-%m-%d")
        
        for sym in self.subscribed_symbols:
            df = self.data_lake.load_candles(sym, "5m")
            if not df.empty:
                df['dt_ny'] = pd.to_datetime(df['datetime'])
                df_day = df[df['dt_ny'].dt.date == target_dt.date()].copy()
                df_day['dt_obj'] = df_day['dt_ny'].dt.tz_localize(ZoneInfo("America/New_York")).dt.tz_convert(ZoneInfo("Asia/Seoul")).dt.tz_localize(None)
                df_day = df_day.sort_values('dt_obj')
                self.mock_data[sym] = df_day

        all_times = set()
        for df in self.mock_data.values():
            all_times.update(df['dt_obj'].tolist())
        self.timeline = sorted(list(all_times))

    def _mock_load_candles(self, symbol: str, timeframe: str, start_dt=None, end_dt=None):
        df = self._original_load_candles(symbol, timeframe)
        if self.current_mock_time and not df.empty:
            mock_kst = self.current_mock_time.replace(tzinfo=ZoneInfo("Asia/Seoul"))
            mock_ny = mock_kst.astimezone(ZoneInfo("America/New_York")).replace(tzinfo=None)
            df['dt_temp'] = pd.to_datetime(df['datetime'])
            df = df[df['dt_temp'] <= mock_ny].copy()
            df = df.drop(columns=['dt_temp'])
        return df.reset_index(drop=True)

    def register_callback(self, cb: Callable[[str, float, Dict[str, Any]], None]):
        if cb not in self._callbacks:
            self._callbacks.append(cb)

    def register_execution_callback(self, cb: Callable[[Dict[str, Any]], None]):
        if cb not in self._execution_callbacks:
            self._execution_callbacks.append(cb)

    def reset_session_ticks(self):
        pass

    def get_latest_price(self, symbol: str, default_val: float = 0.0) -> float:
        return self._latest_prices.get(symbol, default_val)

    def start(self):
        threading.Thread(target=self._run_simulation, daemon=True).start()

    def _run_simulation(self):
        logger.info(f"🚀 MockKiwoomFeeder 시뮬레이션 시작: {self.target_date}")
        for current_time in self.timeline:
            self.current_mock_time = current_time
            
            for sym in self.subscribed_symbols:
                df = self.mock_data.get(sym)
                if df is not None and not df.empty:
                    row = df[df['dt_obj'] == current_time]
                    if not row.empty:
                        price = float(row.iloc[0]['Close'])
                        self._latest_prices[sym] = price
                        
                        for cb in self._callbacks:
                            cb(sym, price, {"type": "mock"})
            
            time.sleep(0.1)
            
        logger.info("✅ MockKiwoomFeeder 시뮬레이션 완료.")
        self.is_finished = True
