# -*- coding: utf-8 -*-
import sys

file_path = 'c:/Users/chabo/OneDrive/바탕 화면/lumos/dashboard_server.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# I need to change:
# full_dates = pd.date_range(start=df['date'].min() - pd.Timedelta(days=400), end=today)
# to
# full_dates = pd.date_range(start=df['date'].min(), end=today)

content = content.replace(
    "full_dates = pd.date_range(start=df['date'].min() - pd.Timedelta(days=400), end=today)",
    "full_dates = pd.date_range(start=df['date'].min(), end=today)"
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("SUCCESS dashboard_server.py date range fix")
