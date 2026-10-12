# -*- coding: utf-8 -*-
import sys

file_path = 'c:/Users/chabo/OneDrive/바탕 화면/lumos/web/index.html'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("최근 3개월 / 5주 / 7일 수익률 현황", "최근 3개월 / 52주 / 7일 수익률 현황")
content = content.replace("Weekly Return (Last 5 Weeks)", "Weekly Return (Last 52 Weeks)")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("SUCCESS index.html")
