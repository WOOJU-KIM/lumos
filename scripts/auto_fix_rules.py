import os
import re
from pathlib import Path

BASE_DIR = Path(r"c:\Users\chabo\OneDrive\바탕 화면\lumos")

def fix_config():
    config_path = BASE_DIR / "config.py"
    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Find duplication
    import_os_indices = [i for i, line in enumerate(lines) if "import os\n" in line or "import os\r\n" in line]
    if len(import_os_indices) >= 2:
        split_idx = import_os_indices[1]
        new_lines = lines[split_idx:]
        with open(config_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print("Fixed config.py duplication.")
    else:
        print("config.py has no duplication.")

def fix_live_runner():
    runner_path = BASE_DIR / "core" / "live_runner.py"
    with open(runner_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replacements
    content = content.replace(
        'tp_threshold = float(active_pos.get("tp_pct", 3.0)) / 100.0 if active_pos.get("tp_pct") else 0.030',
        'tp_threshold = float(active_pos.get("tp_pct", config.MAX_TP_PCT * 100)) / 100.0 if active_pos.get("tp_pct") else config.MAX_TP_PCT'
    )
    content = content.replace(
        'sl_threshold = float(active_pos.get("sl_pct", 2.0)) / 100.0 if active_pos.get("sl_pct") else 0.020',
        'sl_threshold = float(active_pos.get("sl_pct", config.SL_MIN_PCT * 100)) / 100.0 if active_pos.get("sl_pct") else config.SL_MIN_PCT'
    )
    content = content.replace(
        'tp_max_pct = float(active_pos.get("tp_pct", 3.5)) / 100.0 if active_pos else 0.035',
        'tp_max_pct = float(active_pos.get("tp_pct", config.MAX_TP_PCT * 100)) / 100.0 if active_pos else config.MAX_TP_PCT'
    )
    content = content.replace(
        'sl_initial_pct = float(active_pos.get("sl_pct", 2.0)) / 100.0 if active_pos else 0.020',
        'sl_initial_pct = float(active_pos.get("sl_pct", config.SL_MIN_PCT * 100)) / 100.0 if active_pos else config.SL_MIN_PCT'
    )
    
    trailing_old = """            is_trailing_active = False
            current_sl_px = buy_px * (1.0 - sl_initial_pct)
            trailing_trigger_px = buy_px * 1.020
            
            if peak_high >= trailing_trigger_px:
                is_trailing_active = True
                current_sl_px = buy_px * 1.005
                
            if is_trailing_active:
                t = peak_high * 0.992
                if t > current_sl_px:
                    current_sl_px = t"""

    trailing_new = """            is_trailing_active = False
            current_sl_px = buy_px * (1.0 - sl_initial_pct)
            trailing_trigger_px = buy_px * (1.0 + config.TRAILING_TRIGGER_PCT)
            
            if peak_high >= trailing_trigger_px:
                is_trailing_active = True
                t_initial = peak_high * (1.0 - config.TRAILING_DROP_PCT)
                current_sl_px = max(buy_px * 1.002, t_initial)
                
            if is_trailing_active:
                t = peak_high * (1.0 - config.TRAILING_DROP_PCT)
                if t > current_sl_px:
                    current_sl_px = t"""

    content = content.replace(trailing_old, trailing_new)
    
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Fixed live_runner.py hardcoding.")

def fix_prints():
    core_dir = BASE_DIR / "core"
    for py_file in core_dir.glob("*.py"):
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        original_content = content
        
        # Replace print("...", e) -> logger.info(f"... {e}")
        content = re.sub(r'print\(\s*(["\'].*?["\'])\s*,\s*([a-zA-Z0-9_]+)\s*\)', r'logger.info(f"\1 {\2}")', content)
        
        # Replace remaining print(...) with logger.info(...)
        # Need to ensure we don't break multi-line if they exist, but simple one-liners are typical
        # We will just replace `print(` with `logger.info(` except if it's already logger.info
        content = re.sub(r'(?<!_)print\((.*)\)', r'logger.info(\1)', content)
        
        if content != original_content:
            # check if logger is imported, if not add it
            if "import logging" not in content and "logger = logging.getLogger" not in content:
                # Add logger setup
                logger_setup = 'import logging\nlogger = logging.getLogger(__name__)\n'
                content = logger_setup + content
            
            with open(py_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Replaced print() in {py_file.name}")

if __name__ == "__main__":
    fix_config()
    fix_live_runner()
    fix_prints()
