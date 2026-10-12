import re
with open('core/live_runner.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix initial stock balance check to use force_refresh=True
old_check = "stk_bal_check = self.broker.get_overseas_stock_balance()"
new_check = "stk_bal_check = self.broker.get_overseas_stock_balance(force_refresh=True)"
content = content.replace(old_check, new_check)

# Fix _market_execution_loop EOD clear to use _dispatch_sell
old_eod = """                                            self._execute_sell_with_10s_chase(
                                                symbol=sym,
                                                quantity=qty,
                                                reason_desc=f"EOD {config.PHASE_EOD_CLEAR} 100% Liquidation",
                                                is_market_order=True,
                                                buy_px=buy_px
                                            )"""
new_eod = """                                            self._dispatch_sell(
                                                symbol=sym,
                                                quantity=qty,
                                                reason_desc=f"EOD {config.PHASE_EOD_CLEAR} 100% Liquidation",
                                                is_market_order=True,
                                                buy_px=buy_px
                                            )"""
content = content.replace(old_eod, new_eod)

with open('core/live_runner.py', 'w', encoding='utf-8') as f:
    f.write(content)
