import os
from pathlib import Path

fpath = Path('core/mock_kiwoom_feeder.py')
content = fpath.read_text(encoding='utf-8')

target = "df['dt_temp'] = pd.to_datetime(df['datetime'])"
replacement = """from zoneinfo import ZoneInfo
            # current_mock_time (KST)를 NY 시간으로 변환해서 필터링
            mock_kst = self.current_mock_time.replace(tzinfo=ZoneInfo("Asia/Seoul"))
            mock_ny = mock_kst.astimezone(ZoneInfo("America/New_York")).replace(tzinfo=None)
            df['dt_temp'] = pd.to_datetime(df['datetime'])
            df = df[df['dt_temp'] <= mock_ny].copy()
            df = df.drop(columns=['dt_temp'])
            return df.tail(limit).reset_index(drop=True)"""

# I need to replace the whole block because I am changing the filter logic
idx1 = content.find("df['dt_temp'] = pd.to_datetime(df['datetime'])")
if idx1 != -1:
    idx2 = content.find("return df.tail(limit).reset_index(drop=True)", idx1) + len("return df.tail(limit).reset_index(drop=True)")
    content = content[:idx1] + replacement + content[idx2:]

fpath.write_text(content, encoding='utf-8')
