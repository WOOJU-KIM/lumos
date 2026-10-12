import re
import config
with open('core/live_runner.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Buy lock
old_buy_lock = """        with self._order_lock:
            self._is_order_in_progress = True"""
new_buy_lock = """        if not self._acquire_order_lock():
            return False"""
content = content.replace(old_buy_lock, new_buy_lock)

# 2. Sell initial balance check
old_sell_check_1 = """            stk_bal_check = self.broker.get_overseas_stock_balance(force_refresh=True)
            actual_qty = 0
            if stk_bal_check.get("ok"):
                for h in stk_bal_check.get("holdings", []):
                    if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                        actual_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                        break
            if actual_qty <= 0:
                logger.info(f"? [? ? 0??] {symbol} ?? 100% ? ? ? ?? ? ?AI ???? ?")
                self._clear_active_position()
                return True
            quantity = min(quantity, actual_qty)"""

new_sell_check_1 = """            stk_bal_check = self.broker.get_overseas_stock_balance(force_refresh=True)
            if stk_bal_check.get("ok"):
                actual_qty = 0
                for h in stk_bal_check.get("holdings", []):
                    if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                        actual_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                        break
                if actual_qty <= 0:
                    logger.info(f"? [? ? 0??] {symbol} ?? 100% ? ? ? ?? ? ?AI ???? ?")
                    self._clear_active_position()
                    return True
                quantity = min(quantity, actual_qty)
            else:
                pass # API failure, proceed with intended quantity"""
# Wait, this regex/replace might fail due to Korean encodings. I'll use regex or manual replace carefully.
