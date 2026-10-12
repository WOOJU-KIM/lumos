# -*- coding: utf-8 -*-
import sys

file_path = 'c:/Users/chabo/OneDrive/바탕 화면/lumos/dashboard_server.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Change from 4 to 51 (range(51, -1, -1)) for 52 weeks
content = content.replace("range(4, -1, -1)", "range(51, -1, -1)")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("SUCCESS dashboard_server.py")
