import re

with open('tests/test_system_integrity.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('        moe_v3.gbdt_engine.model = MagicMock()', '        moe_v3.gbdt_engine.model = MagicMock()') # I'll use regex to fix all the spaces

content = re.sub(r' +moe_v3\.gbdt_engine\.model = MagicMock\(\)', '        moe_v3.gbdt_engine.model = MagicMock()', content)

with open('tests/test_system_integrity.py', 'w', encoding='utf-8') as f:
    f.write(content)
