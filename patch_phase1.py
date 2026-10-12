import re

with open('core/live_runner.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Syntax Error in _wait_for_fill
wait_for_fill_old = """        start_t = time.time()
        import time
import threading
        while time.time() - start_t < timeout_sec:"""

wait_for_fill_new = """        start_t = time.time()
        while time.time() - start_t < timeout_sec:"""

content = content.replace(wait_for_fill_old, wait_for_fill_new)


# 2. Add _acquire_order_lock and modify _dispatch_sell
# Find _dispatch_sell
dispatch_old = """    def _dispatch_sell(self, **kwargs):
        with self._order_lock:
            if self._is_order_in_progress:
                return
            self._is_order_in_progress = True
        import threading
        threading.Thread(target=self._execute_sell_with_10s_chase, kwargs=kwargs, daemon=True).start()"""

dispatch_new = """    def _acquire_order_lock(self) -> bool:
        with self._order_lock:
            if self._is_order_in_progress:
                return False
            self._is_order_in_progress = True
            return True

    def _dispatch_sell(self, **kwargs):
        if not self._acquire_order_lock():
            return
        kwargs['_already_locked'] = True
        import threading
        threading.Thread(target=self._execute_sell_with_10s_chase, kwargs=kwargs, daemon=True).start()"""

content = content.replace(dispatch_old, dispatch_new)

# Modify _execute_buy_with_10s_chase
exec_buy_old = """    def _execute_buy_with_10s_chase(
        self,
        symbol: str,
        target_qty: int,
        order_px: float,
        moe_res: Dict[str, Any],
        strategy_tag: str = "GBDT_V1"
    ) -> bool:
        \"\"\"
        \"\"\"
        with self._order_lock:
            self._is_order_in_progress = True"""

exec_buy_new = """    def _execute_buy_with_10s_chase(
        self,
        symbol: str,
        target_qty: int,
        order_px: float,
        moe_res: Dict[str, Any],
        strategy_tag: str = "GBDT_V1",
        _already_locked: bool = False
    ) -> bool:
        \"\"\"
        \"\"\"
        if not _already_locked:
            if not self._acquire_order_lock():
                return False"""

content = content.replace(exec_buy_old, exec_buy_new)

# Modify _execute_sell_with_10s_chase
exec_sell_old = """    def _execute_sell_with_10s_chase(
        self,
        symbol: str,
        quantity: int,
        reason_desc: str,
        buy_px: float = 0.0,
        cur_px: float = 0.0,
        is_stoploss: bool = False,
        is_market_order: bool = False
    ) -> bool:
        \"\"\"
        \"\"\"
        with self._order_lock:
            self._is_order_in_progress = True"""

exec_sell_new = """    def _execute_sell_with_10s_chase(
        self,
        symbol: str,
        quantity: int,
        reason_desc: str,
        buy_px: float = 0.0,
        cur_px: float = 0.0,
        is_stoploss: bool = False,
        is_market_order: bool = False,
        _already_locked: bool = False
    ) -> bool:
        \"\"\"
        \"\"\"
        if not _already_locked:
            if not self._acquire_order_lock():
                return False"""

content = content.replace(exec_sell_old, exec_sell_new)

# EOD missing kwargs _already_locked (No, EOD uses _execute_sell_with_10s_chase which acquires lock, that is perfect now).

# 3. Sell Start Balance Check 5s cache -> force_refresh
cache_old = """            # 🚨 [매도 전 잔고 이중 확인 (Hard Rule #2)]
            # 실제 원장에 잡혀있는 잔고(poss_qty)만 매도
            stk_bal_check = self.broker.get_overseas_stock_balance()"""
cache_new = """            # 🚨 [매도 전 잔고 이중 확인 (Hard Rule #2)]
            # 실제 원장에 잡혀있는 잔고(poss_qty)만 매도
            stk_bal_check = self.broker.get_overseas_stock_balance(force_refresh=True)"""
content = content.replace(cache_old, cache_new)

# 4. Sell Loop Quantity Update
loop_old = """                if s_ord_no:
                    self.broker.cancel_order(order_no=s_ord_no, symbol=symbol, quantity=rem_qty)
                    time.sleep(config.CANCEL_ORDER_WAIT_TIME)

        except Exception as e:"""

loop_new = """                if s_ord_no:
                    self.broker.cancel_order(order_no=s_ord_no, symbol=symbol, quantity=rem_qty)
                    time.sleep(config.CANCEL_ORDER_WAIT_TIME)
                
                stk_bal_recheck = self.broker.get_overseas_stock_balance(force_refresh=True)
                new_actual_qty = 0
                if stk_bal_recheck.get("ok"):
                    for h in stk_bal_recheck.get("holdings", []):
                        if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                            new_actual_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                            break
                if new_actual_qty <= 0:
                    logger.info(f"🟢 [매도 완결] {symbol} 잔고 0주 확인. 추격 종료.")
                    self._clear_active_position()
                    return True
                quantity = new_actual_qty

        except Exception as e:"""

content = content.replace(loop_old, loop_new)


with open('core/live_runner.py', 'w', encoding='utf-8') as f:
    f.write(content)


with open('scripts/run_weekly_rolling_wfa.py', 'r', encoding='utf-8') as f:
    wfa = f.read()

wfa = wfa.replace("config.ROLLING_WINDOW_WEEKS", "config.MAX_TRAIN_WEEKS")

with open('scripts/run_weekly_rolling_wfa.py', 'w', encoding='utf-8') as f:
    f.write(wfa)
