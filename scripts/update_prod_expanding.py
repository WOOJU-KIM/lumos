import re

# 1. Update config.py
with open('config.py', 'r', encoding='utf-8') as f:
    config_content = f.read()

config_content = config_content.replace(
    "ROLLING_TRAINING_WEEKS = 104",
    "ROLLING_TRAINING_WEEKS = 520      # 모델 롤링 누적 확장 학습 최대 10년 한도 (Expanding Window)"
)

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(config_content)

# 2. Update run_weekend_data_refresh.py
with open('scripts/run_weekend_data_refresh.py', 'r', encoding='utf-8') as f:
    refresh_content = f.read()

# Replace load_rolling_candles with load_candles to get FULL dataset
refresh_content = re.sub(
    r'df_15m = data_lake\.load_rolling_candles\(config\.BASE_ASSET_LONG, "15m".*?\)',
    'df_15m = data_lake.load_candles(config.BASE_ASSET_LONG, "15m")',
    refresh_content
)

# Update print statements to reflect Expanding Window
refresh_content = refresh_content.replace("최근 504 거래일(2년)", "최대 10년치 누적(Expanding Window)")

with open('scripts/run_weekend_data_refresh.py', 'w', encoding='utf-8') as f:
    f.write(refresh_content)

print("Successfully updated config.py and run_weekend_data_refresh.py for 10-year Expanding Window Live Training.")
