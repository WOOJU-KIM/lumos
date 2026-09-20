with open("core/moe_orchestrator.py", "r", encoding="utf-8") as f:
    text = f.read()
text = text.replace("tqqq_ret=tqqq_r, nvda_ret=nvda_r, qqq_ret=qqq_r, qqq_ret=qqq_r", "tqqq_ret=tqqq_r, nvda_ret=nvda_r, soxx_ret=qqq_r, qqq_ret=qqq_r")
with open("core/moe_orchestrator.py", "w", encoding="utf-8") as f:
    f.write(text)

with open("core/weekly_tournament.py", "r", encoding="utf-8") as f:
    text = f.read()
text = text.replace("tqqq_ret=tqqq_ret,\n                nvda_ret=nvda_ret,\n                qqq_ret=qqq_ret,\n                qqq_ret=qqq_ret,", "tqqq_ret=tqqq_ret,\n                nvda_ret=nvda_ret,\n                soxx_ret=qqq_ret,\n                qqq_ret=qqq_ret,")
with open("core/weekly_tournament.py", "w", encoding="utf-8") as f:
    f.write(text)
