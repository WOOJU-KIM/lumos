import re

with open('core/live_runner.py', 'r', encoding='utf-8') as f:
    content = f.read()

get_atr_func = """
    def _get_5m_atr_pct(self, symbol: str, cur_px: float) -> float:
        try:
            df_5m = self.data_lake.load_candles(symbol, "5m")
            if not df_5m.empty and len(df_5m) >= 14:
                import pandas as pd
                tr = pd.concat([
                    df_5m['High'] - df_5m['Low'],
                    (df_5m['High'] - df_5m['Close'].shift()).abs(),
                    (df_5m['Low'] - df_5m['Close'].shift()).abs()
                ], axis=1).max(axis=1)
                atr_14 = tr.rolling(14).mean().iloc[-1]
                return (atr_14 / cur_px) * 100.0 if cur_px > 0 else 1.8
        except Exception:
            pass
        return 1.8
"""

# inject before _execute_buy_with_10s_chase
content = content.replace("    def _execute_buy_with_10s_chase(", get_atr_func + "\n    def _execute_buy_with_10s_chase(")

# replace ATR calculations
old_atr1 = """                                                atr_14 = float(moe_res.get('features', {}).get('ATR_14', cur_px * 0.018))
                                                atr_pct = (atr_14 / cur_px) * 100.0 if cur_px > 0 else 1.8"""
new_atr1 = """                                                atr_pct = self._get_5m_atr_pct(winner_sym, cur_px)"""
content = content.replace(old_atr1, new_atr1)

with open('core/live_runner.py', 'w', encoding='utf-8') as f:
    f.write(content)
