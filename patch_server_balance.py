# -*- coding: utf-8 -*-
import sys

file_path = 'c:/Users/chabo/OneDrive/바탕 화면/lumos/dashboard_server.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# I need to change:
# usd_diff = bal - pval
# daily_usd.append(round(usd_diff, 2))
# to just use bal for usd.

content = content.replace("usd_diff = bal - pval", "usd_diff = bal")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("SUCCESS dashboard_server.py balance fix")
