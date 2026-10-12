import re

with open('config.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'GBDT_CONFIDENCE_THRESHOLD\s*=\s*[\d\.]+', 'GBDT_CONFIDENCE_THRESHOLD = 0.65', content)

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(content)
