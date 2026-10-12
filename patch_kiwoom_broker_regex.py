import re

with open('core/kiwoom_broker.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_log_regex = r'logger\.info\(f"([^"]*?)Headers:\s*\{json\.dumps\(headers,\s*ensure_ascii=False\)\}"\)'

def repl(match):
    prefix = match.group(1)
    return f'''safe_headers = headers.copy()
        if "authorization" in safe_headers:
            token = safe_headers["authorization"].replace("Bearer ", "")
            safe_headers["authorization"] = f"Bearer {{token[:15]}}...[MASKED]"
        logger.info(f"{prefix}Headers: {{json.dumps(safe_headers, ensure_ascii=False)}}")'''

content = re.sub(old_log_regex, repl, content)

with open('core/kiwoom_broker.py', 'w', encoding='utf-8') as f:
    f.write(content)
