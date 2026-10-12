import os

files_to_fix = [
    "core/live_runner.py",
    "scripts/live_trade_test.py",
    "scripts/mock_live_runner.py",
    "scripts/run_mock_live.py"
]

for file_path in files_to_fix:
    if not os.path.exists(file_path):
        continue
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # We want to replace the fragile line with a try-except block
    fragile_line = 'buy_px = float(str(h.get("purchase_price") or h.get("pchs_avg_pric") or h.get("buy_price") or 0.0).replace(",", ""))'
    
    robust_block = '''try:
                                        buy_px = float(str(h.get("purchase_price") or h.get("pchs_avg_pric") or h.get("buy_price") or 0.0).replace(",", ""))
                                    except Exception:
                                        buy_px = 0.0'''
    
    if fragile_line in content:
        content = content.replace(fragile_line, robust_block)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Secured EOD logic in {file_path}")
    else:
        print(f"Fragile line not found in {file_path}")
