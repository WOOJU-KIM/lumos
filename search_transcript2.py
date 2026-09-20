import json
with open(r"C:\Users\chabo\.gemini\antigravity\brain\a2a0014a-e804-4818-af3d-c670a3b07091\.system_generated\logs\transcript_full.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()
for line in lines:
    if "0.62" in line or "62" in line or "0.60" in line:
        data = json.loads(line)
        if "content" in data and ("0.62" in data["content"] or "0.6" in data["content"]):
            print(data["content"][:200])
