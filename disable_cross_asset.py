import re

with open('config.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'USE_CROSS_ASSET_VETO\s*=\s*True', 'USE_CROSS_ASSET_VETO = False', content)

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(content)
