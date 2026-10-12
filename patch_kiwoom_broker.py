import re

with open('core/kiwoom_broker.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_log = 'logger.info(f"   ??Headers: {json.dumps(headers, ensure_ascii=False)}")'
new_log = '''safe_headers = headers.copy()
        if "authorization" in safe_headers:
            safe_headers["authorization"] = "Bearer *******(MASKED)*******"
        logger.info(f"   ??Headers: {json.dumps(safe_headers, ensure_ascii=False)}")'''

content = content.replace(old_log, new_log)

with open('core/kiwoom_broker.py', 'w', encoding='utf-8') as f:
    f.write(content)
