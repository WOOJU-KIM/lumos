import re
with open('core/moe_orchestrator.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_vix = """            v_c = _get_c("VIX")"""
new_vix = """            v_c = _get_c(getattr(config, "VIX_SYMBOL", "VIXY"))"""

content = content.replace(old_vix, new_vix)

with open('core/moe_orchestrator.py', 'w', encoding='utf-8') as f:
    f.write(content)
