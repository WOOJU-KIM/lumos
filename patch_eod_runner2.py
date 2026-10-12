# -*- coding: utf-8 -*-
import os

with open('core/live_runner.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = '''                                        for ex in exec_hist["executions"]:
                                            msg += f"• {ex.get('symbol', '종목')} | {ex.get('order_type', '구분')} | {ex.get('quantity', 0)}주 @ \n"'''

replacement = '''                                        for ex in exec_hist["executions"]:
                                            sym = ex.get('pdno') or ex.get('stk_cd') or ex.get('symbol') or "알수없음"
                                            od_type = ex.get('sll_buy_dvsn_cd_name') or ex.get('order_type') or ex.get('sll_buy_dvsn_cd') or "체결"
                                            qty = ex.get('ccld_qty') or ex.get('quantity') or ex.get('ord_qty') or 0
                                            px = ex.get('ft_ccld_unpr3') or ex.get('ccld_unpr') or ex.get('price') or 0
                                            msg += f"• {sym} | {od_type} | {qty}주 @ \n"'''

code = code.replace(target, replacement)
with open('core/live_runner.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("Patched EOD runner string formatting!")
