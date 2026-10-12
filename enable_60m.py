import re

with open('config.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'USE_60M_TREND_FILTER\s*=\s*False', 'USE_60M_TREND_FILTER = True', content)

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(content)
