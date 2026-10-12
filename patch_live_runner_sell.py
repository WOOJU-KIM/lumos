import re

with open('core/live_runner.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_code_1 = '''            stk_bal_check = self.broker.get_overseas_stock_balance(force_refresh=True)
            actual_qty = 0
            if stk_bal_check.get("ok"):
                for h in stk_bal_check.get("holdings", []):
                    if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                        actual_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                        break
            if actual_qty <= 0:
                logger.info(f"?? [잔고 0주] {symbol} 이미 100% 매도 완결. 포지션 초기화 및 AI 매수 대기 전환")
                self._clear_active_position()
                return True
            quantity = min(quantity, actual_qty)'''

new_code_1 = '''            stk_bal_check = self.broker.get_overseas_stock_balance(force_refresh=True)
            if stk_bal_check.get("ok"):
                actual_qty = 0
                for h in stk_bal_check.get("holdings", []):
                    if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                        actual_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                        break
                if actual_qty <= 0:
                    logger.info(f"?? [잔고 0주] {symbol} 이미 100% 매도 완결. 포지션 초기화 및 AI 매수 대기 전환")
                    self._clear_active_position()
                    return True
                quantity = min(quantity, actual_qty)
            else:
                logger.warning(f"초기 잔고 조회 실패. 전달받은 수량({quantity})으로 매도 진행.")'''

# I will use regex because of unknown emoji characters.
import re

# Match the block dynamically
pattern_1 = re.compile(
    r'stk_bal_check = self\.broker\.get_overseas_stock_balance\(force_refresh=True\)\n'
    r'\s+actual_qty = 0\n'
    r'\s+if stk_bal_check\.get\("ok"\):\n'
    r'\s+for h in stk_bal_check\.get\("holdings", \[\]\):\n'
    r'\s+if str\(h\.get\("symbol"\) or h\.get\("stk_cd"\) or ""\)\.strip\(\)\.upper\(\) == symbol:\n'
    r'\s+actual_qty = int\(float\(str\(h\.get\("quantity"\) or h\.get\("poss_qty"\) or 0\)\.replace\(",", ""\)\)\)\n'
    r'\s+break\n'
    r'\s+if actual_qty <= 0:\n'
    r'\s+logger\.info\([^)]+\)\n'
    r'\s+self\._clear_active_position\(\)\n'
    r'\s+return True\n'
    r'\s+quantity = min\(quantity, actual_qty\)'
)

match_1 = pattern_1.search(content)
if match_1:
    orig = match_1.group(0)
    # Extract the logger info line
    logger_line = re.search(r'logger\.info\([^)]+\)', orig).group(0)
    
    repl_1 = f'''stk_bal_check = self.broker.get_overseas_stock_balance(force_refresh=True)
            if stk_bal_check.get("ok"):
                actual_qty = 0
                for h in stk_bal_check.get("holdings", []):
                    if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                        actual_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                        break
                if actual_qty <= 0:
                    {logger_line}
                    self._clear_active_position()
                    return True
                quantity = min(quantity, actual_qty)
            else:
                logger.warning(f"초기 잔고 조회 실패. 전달받은 수량({{quantity}})으로 매도 진행.")'''
    
    content = content.replace(orig, repl_1)

# Now for stk_bal_recheck
pattern_2 = re.compile(
    r'stk_bal_recheck = self\.broker\.get_overseas_stock_balance\(force_refresh=True\)\n'
    r'\s+new_actual_qty = 0\n'
    r'\s+if stk_bal_recheck\.get\("ok"\):\n'
    r'\s+for h in stk_bal_recheck\.get\("holdings", \[\]\):\n'
    r'\s+if str\(h\.get\("symbol"\) or h\.get\("stk_cd"\) or ""\)\.strip\(\)\.upper\(\) == symbol:\n'
    r'\s+new_actual_qty = int\(float\(str\(h\.get\("quantity"\) or h\.get\("poss_qty"\) or 0\)\.replace\(",", ""\)\)\)\n'
    r'\s+break\n'
    r'\s+if new_actual_qty <= 0:\n'
    r'\s+logger\.info\([^)]+\)\n'
    r'\s+self\._clear_active_position\(\)\n'
    r'\s+return True\n'
    r'\s+quantity = new_actual_qty'
)

match_2 = pattern_2.search(content)
if match_2:
    orig_2 = match_2.group(0)
    logger_line_2 = re.search(r'logger\.info\([^)]+\)', orig_2).group(0)
    
    repl_2 = f'''stk_bal_recheck = self.broker.get_overseas_stock_balance(force_refresh=True)
                if stk_bal_recheck.get("ok"):
                    new_actual_qty = 0
                    for h in stk_bal_recheck.get("holdings", []):
                        if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                            new_actual_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                            break
                    if new_actual_qty <= 0:
                        {logger_line_2}
                        self._clear_active_position()
                        return True
                    quantity = new_actual_qty
                else:
                    logger.warning(f"잔고 재조회 실패. 기존 수량({{quantity}})으로 매도 추격 계속.")'''
                    
    content = content.replace(orig_2, repl_2)

with open('core/live_runner.py', 'w', encoding='utf-8') as f:
    f.write(content)
