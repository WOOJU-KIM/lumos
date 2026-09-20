import os
from pathlib import Path

fpath = Path('scripts/live_trade_test.py')
content = fpath.read_text(encoding='utf-8')

# 1. replace ws_streamer
content = content.replace(
    'from core.kiwoom_ws_streamer import KiwoomWebSocketStreamer',
    'from core.mock_kiwoom_feeder import MockKiwoomFeeder as KiwoomWebSocketStreamer'
)

# 2. replace datetime.now() with mock time inside the loop
content = content.replace(
    'now_dt = datetime.now()',
    'now_dt = getattr(self.ws_streamer, "current_mock_time", datetime.now()) or datetime.now()'
)
content = content.replace(
    'now_t = time.time()',
    'now_t = now_dt.timestamp()'
)

# 3. Add print and log for trade test
target = """moe_res = self.moe_orchestrator.evaluate_dual_filter_signal(df_candle_15m=df_15m, live_prices=live_prices)
                            self._last_moe_res = moe_res
                            
                            conf = float(moe_res.get('gating_confidence', 0.0)) * 100.0
                            is_appr = moe_res.get('is_approved', False)
                            direction = moe_res.get('direction', 'NONE')"""

replacement = """moe_res = self.moe_orchestrator.evaluate_dual_filter_signal(df_candle_15m=df_15m, live_prices=live_prices)
                            self._last_moe_res = moe_res
                            
                            conf = float(moe_res.get('gating_confidence', 0.0)) * 100.0
                            is_appr = moe_res.get('is_approved', False)
                            direction = moe_res.get('direction', 'NONE')
                            
                            # --- [매매 테스트 출력 및 로깅] ---
                            test_log = f"[{now_dt.strftime('%Y-%m-%d %H:%M:%S')}] 🎯 매매테스트 타점 분석 | 방향: {direction:<10} | GBDT 확신도: {conf:>5.2f}% | 🚀 진입승인: {is_appr}"
                            print(test_log)
                            with open(DATA_DIR / 'trade_test_logs.txt', 'a', encoding='utf-8') as f:
                                f.write(test_log + '\\n')
                            # ----------------------------------"""

if target in content:
    content = content.replace(target, replacement)
    
# 4. Mock the sleep so it doesn't wait in real time for 1 second loop if we can speed it up
# But live_runner loop is while True: time.sleep(1)
content = content.replace('time.sleep(1)', 'time.sleep(0.01)')

# 5. Disable Telegram messages to avoid spamming the user during test
content = content.replace('self.dispatcher.send_telegram_message(', '# self.dispatcher.send_telegram_message(')

fpath.write_text(content, encoding='utf-8')
print("Successfully patched live_trade_test.py")
