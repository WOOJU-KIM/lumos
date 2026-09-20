import os
from pathlib import Path

fpath = Path('core/mock_kiwoom_feeder.py')
content = fpath.read_text(encoding='utf-8')

# The DB stores NY time. We need to localize it to NY timezone and convert to KST
target = "df['dt_obj'] = pd.to_datetime(df['datetime'])"
replacement = """from zoneinfo import ZoneInfo
                df['dt_obj'] = pd.to_datetime(df['datetime'])
                # DB 시간은 NY 시간이므로 KST로 변환
                df['dt_obj'] = df['dt_obj'].dt.tz_localize(ZoneInfo("America/New_York")).dt.tz_convert(ZoneInfo("Asia/Seoul")).dt.tz_localize(None)"""

content = content.replace(target, replacement)
fpath.write_text(content, encoding='utf-8')
