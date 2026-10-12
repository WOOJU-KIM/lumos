import re
import config
with open('core/weekly_tournament.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('"TimeStop(90m)"', f'f"TimeStop({{config.TIME_STOP_MINUTES}}m)"')
content = content.replace('90분 타임스탑', f'{config.TIME_STOP_MINUTES}분 타임스탑')

with open('core/weekly_tournament.py', 'w', encoding='utf-8') as f:
    f.write(content)
