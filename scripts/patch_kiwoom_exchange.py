import re

with open('core/kiwoom_broker.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Add a robust method to resolve exchange codes inside KiwoomBroker
exchange_method = """
    def _resolve_exchange(self, symbol: str) -> str:
        \"\"\"종목 기호에 따른 거래소 코드(NY/ND) 자동 판별\"\"\"
        sym = symbol.upper().strip()
        # 아멕스/뉴욕(NYSE Arca) 상장 대표 ETF 및 주식
        ny_symbols = ["SOXL", "SOXS", "SPY", "DIA", "VIXY", "TLT", "TMF", "TMV", "UPRO", "SPXU"]
        if sym in ny_symbols:
            return "NY"
        # 나스닥 상장 (TQQQ, SQQQ, QQQ, NVDA, SOXX, IEF 등)
        return "ND"
"""

# Insert the method right after def __init__(...): ... 
# Or before send_order
code = re.sub(
    r'(def send_order\()', 
    exchange_method.lstrip() + r'\n    \1', 
    code, 
    count=1
)

# Update send_order
code = re.sub(
    r'stex = exchange or \("NY" if sym_clean in \["SOXL", "SOXS", "SPY", "DIA", "VIXY"\] else "ND"\)',
    r'stex = exchange or self._resolve_exchange(sym_clean)',
    code
)

# Update cancel_all_open_orders
code = re.sub(
    r'stex_tp = "NY" if \("아멕스" in stex_nm or ord_sym in \["SOXL", "SOXS"\]\) else "ND"',
    r'stex_tp = self._resolve_exchange(ord_sym)',
    code
)

with open('core/kiwoom_broker.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("KiwoomBroker exchange routing logic fortified.")
