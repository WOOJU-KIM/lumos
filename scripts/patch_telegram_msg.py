import re

with open('core/live_runner.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace hardcoded open_msg
old_open_msg = r'open_msg = f"""(.*?)(MoE V3).*?(GBDT 62%).*?(Max TP \+3\.5%).*?"""'

# Wait, regex might fail with literal braces. Let's just use string replace.
replacement_open_msg = """open_msg = f\"\"\"🟢 **[미국장 오픈 알림]**\\n\\n⏰ 현재 시간: `{mkt['now_kst_str']}`\\n💼 체결 모드: `{self.broker.mode_str}`\\n🧠 AI 엔진: `하이브리드 MoE V5 (09:30~15:30 EDT)`\\n🎯 진입 기준: `GBDT {config.GBDT_CONFIDENCE_THRESHOLD*100:.1f}% 이상`\\n🛡️ 청산 기준: `Max TP +3.0% / SL -2.0% / 90m 타임스탑 / ATR 트레일링`\\n🧹 장마감: `{config.PHASE_EOD_CLEAR} EDT 0% 오버나잇 청산`\"\"\""""

content = re.sub(
    r'open_msg = f"""[^"]*?"""',
    replacement_open_msg,
    content,
    flags=re.DOTALL
)

with open('core/live_runner.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patched live_runner.py telegram messages")
