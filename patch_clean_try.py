import os
from pathlib import Path

fpath = Path('scripts/live_trade_test_clean.py')
content = fpath.read_text(encoding='utf-8')

replacement = """                df_15m = data_lake.load_candles("TQQQ", "15m")
                if df_15m.empty:
                    continue
                    
                live_prices = {
                    "TQQQ": feeder.get_latest_price("TQQQ", 0.0),
                    "SQQQ": feeder.get_latest_price("SQQQ", 0.0),
                    "SOXX": feeder.get_latest_price("SOXX", 0.0),
                    "QQQ":  feeder.get_latest_price("QQQ", 0.0),
                    "NVDA": feeder.get_latest_price("NVDA", 0.0),
                    "VIXY": feeder.get_latest_price("VIXY", 0.0),
                    "IEF":  feeder.get_latest_price("IEF", 0.0),
                }
                
                moe_res = moe_orchestrator.evaluate_dual_filter_signal(df_candle_15m=df_15m, live_prices=live_prices)
                
                conf = float(moe_res.get('gating_confidence', 0.0)) * 100.0
                is_appr = moe_res.get('is_approved', False)
                direction = moe_res.get('direction', 'NONE')
                
                mock_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
                test_log = f"[{mock_time_str}] 🎯 매매테스트 | 방향: {direction:<10} | GBDT 확신도: {conf:>5.2f}% | 🚀 진입승인: {is_appr}"
                print(test_log)
                
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(test_log + '\\n')
"""

idx1 = content.find("try:")
idx2 = content.find("except Exception as e:")
idx3 = content.find("print(f\"[{current_time}] 에러 발생: {e}\")", idx2) + len("print(f\"[{current_time}] 에러 발생: {e}\")")

content = content[:idx1] + replacement + content[idx3:]
fpath.write_text(content, encoding='utf-8')
