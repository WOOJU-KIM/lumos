# -*- coding: utf-8 -*-
import os

with open('scripts/run_live_test.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = '''                                    global_runner._execute_buy_with_10s_chase(winner_sym, order_qty, cur_px, moe_res, live_prices)'''

replacement = '''                                    
                                    atr_14 = float(moe_res.get("atr_14", 0.0))
                                    atr_pct = (atr_14 / cur_px) * 100.0 if cur_px > 0 else 2.0
                                    sl_pct = max(config.SL_MIN_PCT * 100.0, min(config.SL_MAX_PCT * 100.0, atr_pct * config.SL_ATR_MULTIPLIER))
                                    tp_pct = config.MAX_TP_PCT * 100.0
                                    targets = {
                                        "dynamic_tp_px": round(cur_px * (1.0 + tp_pct/100.0), 2),
                                        "dynamic_sl_px": round(cur_px * (1.0 - sl_pct/100.0), 2),
                                        "tp_pct": tp_pct,
                                        "sl_pct": sl_pct
                                    }
                                    global_runner._execute_buy_with_10s_chase(winner_sym, order_qty, cur_px, moe_res, targets)'''

code = code.replace(target, replacement)
with open('scripts/run_live_test.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("Patched targets!")
