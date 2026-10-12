import re

with open("config.py", "r", encoding="cp949", errors="replace") as f:
    config_code = f.read()

# Fix Trailing Stop parameters to match the strict SOXL rules (1.5% / 0.3%)
config_code = re.sub(r"TRAILING_TRIGGER_PCT\s*=\s*[0-9.]+\s*#.*", "TRAILING_TRIGGER_PCT = 0.015      # 트레일링 스탑 발동 조건 (+1.5%)", config_code)
config_code = re.sub(r"TRAILING_DROP_PCT\s*=\s*[0-9.]+\s*#.*", "TRAILING_DROP_PCT = 0.003       # 트레일링 스탑 청산 조건 (최고점 대비 -0.3%)", config_code)
config_code = re.sub(r"TIME_STOP_MINUTES\s*=\s*[0-9.]+\s*#.*", "TIME_STOP_MINUTES = 90           # 강제 타임 스탑 (90분)", config_code)
config_code = re.sub(r"MACRO_TREND_SYMBOL\s*=\s*\"[^\"]+\"\s*#.*", "MACRO_TREND_SYMBOL=\"QQQ\"        # GBDT 대추세 (QQQ 60분봉 추세 방패 기준 종목)", config_code)

with open("config.py", "w", encoding="cp949", errors="replace") as f:
    f.write(config_code)

print("config.py successfully aligned with SOXL production rules.")
