import re

with open('config.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add GBDT_MAX_DEPTH = 4, LABEL_TP_PCT = 0.03
if 'GBDT_MAX_DEPTH' not in content:
    content = content.replace('GBDT_N_ESTIMATORS = 120', 'GBDT_MAX_DEPTH = 4\nGBDT_N_ESTIMATORS = 120')

if 'LABEL_TP_PCT' not in content:
    content = content.replace('MAX_TP_PCT = 0.10', 'MAX_TP_PCT = 0.10\nLABEL_TP_PCT = 0.03               # 라벨링 전용 목표 익절 라인 (3.0%)')

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(content)

import os
active_pos = 'data/active_position.json'
if os.path.exists(active_pos):
    os.remove(active_pos)
    print(f"Removed {active_pos}")

