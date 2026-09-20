import os
from pathlib import Path

fpath = Path('core/mock_kiwoom_feeder.py')
content = fpath.read_text(encoding='utf-8')

replacement = """
    def _load_historical_data(self):
        self.mock_data = {}
        target_dt = datetime.strptime(self.target_date, "%Y-%m-%d")
        
        from zoneinfo import ZoneInfo
        for sym in self.subscribed_symbols:
            df = self.data_lake.load_candles(sym, "5m")
            if not df.empty:
                df['dt_ny'] = pd.to_datetime(df['datetime'])
                # 타겟 날짜(NY 기준)의 데이터만 필터링
                df_day = df[df['dt_ny'].dt.date == target_dt.date()].copy()
                
                # 시뮬레이션을 위해 KST로 변환하여 dt_obj에 저장
                df_day['dt_obj'] = df_day['dt_ny'].dt.tz_localize(ZoneInfo("America/New_York")).dt.tz_convert(ZoneInfo("Asia/Seoul")).dt.tz_localize(None)
                df_day = df_day.sort_values('dt_obj')
                self.mock_data[sym] = df_day

        all_times = set()
        for df in self.mock_data.values():
            all_times.update(df['dt_obj'].tolist())
        self.timeline = sorted(list(all_times))
"""

idx1 = content.find("def _load_historical_data(self):")
if idx1 != -1:
    idx2 = content.find("def _mock_load_candles", idx1)
    content = content[:idx1] + replacement.strip() + "\n\n" + content[idx2:]
    
fpath.write_text(content, encoding='utf-8')
