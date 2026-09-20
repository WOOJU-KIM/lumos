import json
with open(r"C:\Users\chabo\.gemini\antigravity\brain\a2a0014a-e804-4818-af3d-c670a3b07091\.system_generated\logs\transcript_full.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if "moe_orchestrator =" in line or "MoEMetaOrchestrator(" in line:
            print(line[:200])
