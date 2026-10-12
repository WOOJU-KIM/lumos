import re

with open('core/kis_client.py', 'r', encoding='utf-8') as f:
    code = f.read()

replacement = """def get_exchange_code(self, ticker: str) -> str:
        t = ticker.upper().strip()
        if self.mode == "VIRTUAL":
            return "NASD"
        
        # REAL 모드 거래소 정밀 매핑
        if t in ["SOXL", "SOXS", "VIXY"]:
            return "AMS"  # NYSE Arca
        elif t in ["SPY"]:
            return "NYSE" # NYSE
        else:
            return "NASD" # TQQQ, SQQQ, QQQ, NVDA (Nasdaq)"""

code = re.sub(r'def get_exchange_code\(self, ticker: str\) -> str:.*?return "NASD"', replacement, code, flags=re.DOTALL)

with open('core/kis_client.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Exchange code patched.")
