import re

with open('config.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'USE_60M_TREND_FILTER\s*=\s*True', 'USE_60M_TREND_FILTER = False', content)
content = re.sub(r'TRAILING_TRIGGER_PCT\s*=\s*[\d\.]+', 'TRAILING_TRIGGER_PCT = 0.010', content)
content = re.sub(r'TRAILING_DROP_PCT\s*=\s*[\d\.]+', 'TRAILING_DROP_PCT = 0.005', content)

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(content)
