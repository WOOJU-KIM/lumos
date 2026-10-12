# -*- coding: utf-8 -*-
import sys

file_path = 'c:/Users/chabo/OneDrive/바탕 화면/lumos/web/index.html'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("label: 'PnL (USD)'", "label: 'Total Balance (USD)'")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("SUCCESS HTML label fix")
