import re
from pathlib import Path

def revert_to_markdown():
    # 1. Revert dispatcher
    p = Path('agents/dispatcher_agent.py')
    c = p.read_text(encoding='utf-8')
    c = c.replace('"parse_mode": "HTML"', '"parse_mode": "Markdown"')
    p.write_text(c, encoding='utf-8')

    # 2. Revert circuit breaker
    p = Path('core/circuit_breaker.py')
    c = p.read_text(encoding='utf-8')
    c = c.replace('"parse_mode": "HTML"', '"parse_mode": "Markdown"')
    p.write_text(c, encoding='utf-8')

    # 3. Update live_runner to use * for bold instead of <b> or **
    p = Path('core/live_runner.py')
    c = p.read_text(encoding='utf-8', errors='ignore')
    
    # Replace any <b>...</b> with *...*
    c = re.sub(r'<b>(.*?)</b>', r'*\1*', c)
    # Just in case there are remaining **
    c = re.sub(r'\*\*(.*?)\*\*', r'*\1*', c)
    
    p.write_text(c, encoding='utf-8')

revert_to_markdown()
print('Reverted to Telegram Markdown mode with *asterisks* for bold!')
