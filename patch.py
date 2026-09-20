import sys

with open('temp_run_all_5m.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_logic = '''                    if cur_max_ret > peak_ret: peak_ret = cur_max_ret
                    if peak_ret >= 0.015: trailing_active = True
                    
                    if cur_max_ret >= 0.10:
                        exit_price = tp_px
                        exit_reason = '🎯 목표익절 (+10.0%)'
                        exit_dt = b_dt
                        holding_5m_bars = k + 1
                        break
                    elif trailing_active and (peak_ret - cur_c_ret >= 0.003):
                        exit_price = b_c
                        exit_reason = '🛡️ 트레일링 스탑'
                        exit_dt = b_dt
                        holding_5m_bars = k + 1
                        break
                    elif cur_min_ret <= -initial_sl:
                        exit_price = sl_px
                        exit_reason = f'🛑 손절 (-{initial_sl*100:.1f}%)'
                        exit_dt = b_dt
                        holding_5m_bars = k + 1
                        break'''

new_logic = '''                    if cur_max_ret > peak_ret: peak_ret = cur_max_ret
                    if peak_ret >= 0.015: trailing_active = True
                    
                    hit_sl = (cur_min_ret <= -initial_sl)
                    hit_tp = (cur_max_ret >= 0.10)
                    
                    # 보수적 백테스트: 같은 봉에서 SL과 TP가 동시 터치된 경우 무조건 SL로 처리
                    if hit_sl:
                        exit_price = sl_px
                        exit_reason = f'🛑 손절 (-{initial_sl*100:.1f}%)'
                        exit_dt = b_dt
                        holding_5m_bars = k + 1
                        break
                    elif hit_tp:
                        exit_price = tp_px
                        exit_reason = '🎯 목표익절 (+10.0%)'
                        exit_dt = b_dt
                        holding_5m_bars = k + 1
                        break
                    elif trailing_active and (peak_ret - cur_c_ret >= 0.003):
                        exit_price = b_c
                        exit_reason = '🛡️ 트레일링 스탑'
                        exit_dt = b_dt
                        holding_5m_bars = k + 1
                        break'''

if old_logic in content:
    new_content = content.replace(old_logic, new_logic)
    
    # Also change the output json file name
    new_content = new_content.replace('out_file = PROJECT_ROOT / "data" / f"backtest_5m_precision_{model_train_mode}.json"', 
                                      'out_file = PROJECT_ROOT / "data" / f"backtest_5m_precision_conservative_{model_train_mode}.json"')
                                      
    # And csv file names
    new_content = new_content.replace('data/wfa_tqqq_trades.csv', 'data/wfa_tqqq_trades_conservative.csv')
    new_content = new_content.replace('data/wfa_upro_trades.csv', 'data/wfa_upro_trades_conservative.csv')

    with open('temp_run_conservative_5m.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('Replacement successful. conservative script created.')
else:
    print('Could not find old_logic block.')
