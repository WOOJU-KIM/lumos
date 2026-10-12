import re

with open('config.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'TRAILING_TRIGGER_PCT\s*=\s*[\d\.]+', 'TRAILING_TRIGGER_PCT = 0.015', content)
content = re.sub(r'TRAILING_DROP_PCT\s*=\s*[\d\.]+', 'TRAILING_DROP_PCT = 0.008', content)

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(content)
