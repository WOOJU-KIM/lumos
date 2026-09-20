import os
from pathlib import Path

fpath = Path('core/mock_kiwoom_feeder.py')
content = fpath.read_text(encoding='utf-8')

replacement = """
    def _mock_load_candles(self, symbol: str, timeframe: str, limit: int = 2000):
        df = self._original_load_candles(symbol, timeframe, limit=5000)
        if self.current_mock_time and not df.empty:
            from zoneinfo import ZoneInfo
            mock_kst = self.current_mock_time.replace(tzinfo=ZoneInfo("Asia/Seoul"))
            mock_ny = mock_kst.astimezone(ZoneInfo("America/New_York")).replace(tzinfo=None)
            df['dt_temp'] = pd.to_datetime(df['datetime'])
            df = df[df['dt_temp'] <= mock_ny].copy()
            df = df.drop(columns=['dt_temp'])
        return df.tail(limit).reset_index(drop=True)
"""

idx1 = content.find("def _mock_load_candles")
idx2 = content.find("def register_callback", idx1)
content = content[:idx1] + replacement.strip() + "\n\n    " + content[idx2:]
fpath.write_text(content, encoding='utf-8')
