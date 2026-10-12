import re

with open('core/weekly_tournament.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('df_long_504 = self.data_lake.load_rolling_candles(config.BASE_ASSET_LONG, "15m", max_trading_days=504)', 'df_long_504 = self.data_lake.load_candles(config.BASE_ASSET_LONG, "15m")')

with open('core/weekly_tournament.py', 'w', encoding='utf-8') as f:
    f.write(content)
