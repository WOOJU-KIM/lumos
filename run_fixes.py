import os
import re
from pathlib import Path

def fix_dispatcher():
    p = Path('agents/dispatcher_agent.py')
    c = p.read_text(encoding='utf-8')
    c = c.replace('"parse_mode": "Markdown"', '"parse_mode": "HTML"')
    # replace markdown bold (**text**) with HTML bold (<b>text</b>)
    c = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', c)
    p.write_text(c, encoding='utf-8')

def fix_circuit_breaker():
    p = Path('core/circuit_breaker.py')
    c = p.read_text(encoding='utf-8')
    c = c.replace('"parse_mode": "Markdown"', '"parse_mode": "HTML"')
    c = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', c)
    p.write_text(c, encoding='utf-8')

def fix_live_runner():
    p = Path('core/live_runner.py')
    c = p.read_text(encoding='utf-8', errors='ignore')
    
    # 1. Convert any **text** to <b>text</b>
    c = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', c)
    
    # 2. Fix the specific question marks in the Telegram messages
    c = c.replace('? [?? ?]', '💻 [모의 투자]')
    c = c.replace('? [?? ??]', '🔥 [실전 투자]')
    
    # Fix buy_msg
    c = c.replace('??<b>', '🔥 <b>')
    c = c.replace('??', '확신도')
    
    # Let's fix specific known broken strings
    c = c.replace('? [  ??? ?]', '🔥 <b>[정규장 매매 개시]</b>')
    c = c.replace('??? ?:', '⏰ 현재 시각:')
    c = c.replace('?  :', '🖥️ 실행 모드:')
    c = c.replace('? AI ??', '🧠 AI 전략: ')
    c = c.replace('? :', '🎯 매수 룰:')
    c = c.replace('? ?:', '🛡️ 청산 룰:')
    c = c.replace('????:', '🌙 마감청산:')
    
    c = c.replace('? [??? ????', '🟢 [매수 주문 완료]')
    c = c.replace('? [??? ??', '🔴 [매도 주문 완료]')
    
    c = c.replace('? **[Lumos ??', '🏁 <b>[Lumos 라이브러너 기동]')
    
    p.write_text(c, encoding='utf-8')

fix_dispatcher()
fix_circuit_breaker()
fix_live_runner()
print('Fixes applied!')
