import json
with open(r"C:\Users\chabo\.gemini\antigravity\brain\a2a0014a-e804-4818-af3d-c670a3b07091\.system_generated\logs\transcript_full.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()
for line in lines:
    if "temp_backtest_tqqq.py" in line:
        data = json.loads(line)
        if "content" in data and "text = text.replace" in data["content"]:
            print(data["content"])
        if "tool_calls" in data:
            for call in data["tool_calls"]:
                if "CommandLine" in call.get("args", {}):
                    if "text.replace" in call["args"]["CommandLine"]:
                        print(call["args"]["CommandLine"])
