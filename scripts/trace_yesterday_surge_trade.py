import sys
import config
import pandas as pd
import numpy as np
from pathlib import Path

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import MarketDataLake

lake = MarketDataLake()
long_15m = lake.load_candles(config.BASE_ASSET_LONG, '15m')
long_5m = lake.load_candles(config.BASE_ASSET_LONG, '5m')

long_15m_yest = long_15m[long_15m['datetime'].str.startswith('2026-08-17')].copy().reset_index(drop=True)
long_5m_yest = long_5m[long_5m['datetime'].str.startswith('2026-08-17')].copy().reset_index(drop=True)

print("=" * 80)
print("🔍 [2026-08-17 직전 장 TQQQ 급등 시점 매수 체결 및 청산 시뮬레이션]")
print("=" * 80)

# 1. 09:30 진입 시뮬레이션
entry_bar = long_15m_yest.iloc[0]
entry_time = entry_bar['datetime']
entry_price = float(entry_bar['Close'])  # $152.29
tp_price = round(entry_price * 1.035, 2)  # $157.62 (+3.5%)
sl_price = round(entry_price * 0.980, 2)  # $149.24 (-2.0%)
time_stop_limit_time = "11:00"  # 90분 (15분봉 6개 / 5분봉 18개)

capital_usd = 100_000.0
quantity = int((capital_usd * 0.98) / entry_price)
invested_usd = quantity * entry_price

print(f"1. 매수 진입 시점: {entry_time} (NYT 09:30 / 한국시각 22:30)")
print(f"   • 매수 단가: ${entry_price:.2f}")
print(f"   • 매수 수량: {quantity:,}주 (투자원금: ${invested_usd:,.2f})")
print(f"   • 목표 익절가 (+3.5%): ${tp_price:.2f}")
print(f"   • 손절 기준가 (-2.0%): ${sl_price:.2f}")
print(f"   • 90분 타임스탑 제한 시각: 11:00 NYT (한국시각 00:00)")

print("\n2. 진입 후 15분봉 시계열 궤적:")
print("-" * 80)
print(f"{'봉 시각':<18} | {'시가':<7} | {'고가':<7} | {'저가':<7} | {'종가':<7} | {'최고수익률':<10} | {'종가수익률':<10} | {'상태'}")
print("-" * 80)

exit_found = False
exit_reason = ""
exit_price = 0.0
exit_time = ""
bars_held = 0

for idx, r in long_15m_yest.iterrows():
    b_time = r['datetime']
    t_str = b_time.split(' ')[1][:5]
    o = float(r['Open'])
    h = float(r['High'])
    l = float(r['Low'])
    c = float(r['Close'])
    
    max_gain = ((h - entry_price) / entry_price) * 100
    close_gain = ((c - entry_price) / entry_price) * 100
    
    status_str = "보유 중"
    if idx == 0:
        status_str = "🚀 매수 체결"
    elif not exit_found:
        bars_held = idx
        # Check TP
        if h >= tp_price:
            exit_found = True
            exit_reason = "🎯 목표 익절 달성 (+3.5%)"
            exit_price = tp_price
            exit_time = b_time
            status_str = "🎯 익절 청산!"
        # Check SL
        elif l <= sl_price:
            exit_found = True
            exit_reason = "🛑 손절 청산 (-2.0%)"
            exit_price = sl_price
            exit_time = b_time
            status_str = "🛑 손절 청산!"
        # Check 90 min timestop (6 bars held = 11:00)
        elif idx >= 6:
            exit_found = True
            exit_reason = "⏰ 90분 타임스탑 청산 (시간초과)"
            exit_price = c
            exit_time = b_time
            status_str = "⏰ 타임스탑 청산!"
            
    print(f"{b_time:<18} | ${o:<6.2f} | ${h:<6.2f} | ${l:<6.2f} | ${c:<6.2f} | +{max_gain:>6.2f}%   | {close_gain:>+6.2f}%   | {status_str}")

# Final PnL Calculation
gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100
net_pnl_pct = gross_pnl_pct - 0.30  # 0.30% 수수료/슬리피지 차감
pnl_usd = quantity * (exit_price - entry_price) - (invested_usd * 0.0030)
pnl_krw = int(pnl_usd * 1415.0)

print("\n" + "=" * 80)
print("🏆 [최종 매매 결과 요약 결산]")
print("=" * 80)
print(f"• 매수 시점: {entry_time} (${entry_price:.2f}, {quantity:,}주)")
print(f"• 청산 시점: {exit_time} (${exit_price:.2f})")
print(f"• 청산 사유: {exit_reason}")
print(f"• 보유 기간: {bars_held}개 봉 (약 {bars_held * 15}분간 보유)")
print(f"• 실현 수익률 (Gross): {gross_pnl_pct:+.2f}%")
print(f"• 수수료 차감 후 순수익률 (Net): {net_pnl_pct:+.2f}%")
print(f"• 실현 손익 (USD): ${pnl_usd:+,.2f} USD")
print(f"• 실현 손익 (KRW 환산): ₩{pnl_krw:+,}원")
print("=" * 80)
