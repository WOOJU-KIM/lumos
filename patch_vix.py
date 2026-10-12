import re
with open('core/weekly_tournament.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_vix = """        vix_15m  = self.data_lake.load_candles("^VIX", "15m")"""
new_vix = """        vix_15m  = self.data_lake.load_candles(getattr(config, "VIX_SYMBOL", "VIXY"), "15m")"""

content = content.replace(old_vix, new_vix)

with open('core/weekly_tournament.py', 'w', encoding='utf-8') as f:
    f.write(content)
