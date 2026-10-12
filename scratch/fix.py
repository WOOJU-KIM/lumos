import re
with open('scratch/temp_run_rolling_oos_db_backtest.py', 'r', encoding='utf-8') as f:
    c = f.read()

c = re.sub(
    r'if hit_tp and hit_sl:.*?break',
    '''if hit_tp and hit_sl:
                        # 무조건 보수적으로 손절(SL) 우선 처리 (하드 룰)
                        exit_price = sl_px
                        exit_reason = "STOP_LOSS (5m Low)"
                        exit_dt = b_dt
                        holding_5m_bars = k + 1
                        break''',
    c, flags=re.DOTALL
)

with open('scratch/temp_run_rolling_oos_db_backtest.py', 'w', encoding='utf-8') as f:
    f.write(c)
