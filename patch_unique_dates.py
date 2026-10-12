import re

with open('core/weekly_tournament.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Make tournament evaluate only on the last 20 days
old_dates = "        unique_dates = sorted(long_15m_feat['date_str'].unique())"
new_dates = "        unique_dates = sorted(long_15m_feat['date_str'].unique())[-20:]  # 최근 4주(20거래일) OOS 평가"
content = content.replace(old_dates, new_dates)

with open('core/weekly_tournament.py', 'w', encoding='utf-8') as f:
    f.write(content)
