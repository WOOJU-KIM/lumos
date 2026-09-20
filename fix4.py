with open("core/kiwoom_broker.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace('in ["TQQQ", "SQQQ", "SPY", "DIA", "VIXY"]', 'in ["SPY", "DIA", "VIXY"]')
text = text.replace('("Ƹ߽" in stex_nm or ord_sym in ["TQQQ", "SQQQ"])', '("Ƹ߽" in stex_nm)')

with open("core/kiwoom_broker.py", "w", encoding="utf-8") as f:
    f.write(text)
