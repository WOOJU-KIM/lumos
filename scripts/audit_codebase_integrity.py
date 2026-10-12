import sys
import re
from pathlib import Path

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
core_dir = PROJECT_ROOT / "core"

suspicious_patterns = [
    (r"129\.\d+", "Hardcoded price around ~129"),
    (r"124\.\d+", "Hardcoded price around ~124"),
    (r"99999|100000", "Hardcoded dummy balance"),
    (r"long_ret\s*=\s*0\.", "Hardcoded returns in model call"),
    (r"nvda_ret\s*=\s*0\.", "Hardcoded returns in model call"),
    (r"except\s*:\s*pass", "Silent bare except: pass"),
    (r"direction\s*=\s*.*if.*>=\s*0", "sig_code >= 0 false positive mapping"),
]

print("=" * 80)
print("🔍 [Lumos 전사적 하드코딩 및 잠재 결함 전수 점검 (Codebase Integrity Audit)]")
print("=" * 80)

findings = []
for py_file in core_dir.glob("*.py"):
    with open(py_file, "r", encoding="utf-8") as f:
        content = f.read()
        lines = content.split("\n")
    
    for idx, line in enumerate(lines, 1):
        for pattern, desc in suspicious_patterns:
            if re.search(pattern, line):
                # Skip comments or valid docstrings if needed
                stripped = line.strip()
                if not stripped.startswith("#") and not stripped.startswith('"""'):
                    findings.append({
                        "file": py_file.name,
                        "line_no": idx,
                        "code": stripped,
                        "desc": desc
                    })

print(f"총 발견된 잠재 결함/하드코딩 항목: {len(findings)}건\n")
for f in findings:
    print(f"[{f['file']}:{f['line_no']}] ({f['desc']})")
    print(f"   >>> {f['code']}")
print("=" * 80)
