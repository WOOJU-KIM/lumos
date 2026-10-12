# -*- coding: utf-8 -*-
import sys

file_path = 'c:/Users/chabo/OneDrive/바탕 화면/lumos/dashboard_server.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

if 'import pandas as pd' not in content:
    content = content.replace('import json', 'import json\nimport pandas as pd')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS")
else:
    print("ALREADY IMPORTED")
