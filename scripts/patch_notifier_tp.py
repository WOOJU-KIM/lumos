import re

with open('core/telegram_notifier.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(
    r'calc_tp = tp_price if tp_price is not None else round\(entry_price \* 1\.030, 2\)',
    r'calc_tp = tp_price if tp_price is not None else round(entry_price * (1.0 + config.MAX_TP_PCT), 2)',
    content
)

with open('core/telegram_notifier.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patched telegram_notifier.py hardcoded TP")
