import re

with open('config.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'USE_CROSS_ASSET_VETO\s*=\s*False', 'USE_CROSS_ASSET_VETO = True', content)
content = re.sub(r'USE_60M_TREND_FILTER\s*=\s*False', 'USE_60M_TREND_FILTER = True', content)
content = re.sub(r'GBDT_CONFIDENCE_THRESHOLD\s*=\s*[\d\.]+', 'GBDT_CONFIDENCE_THRESHOLD = 0.68', content)

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(content)
