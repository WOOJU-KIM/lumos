import os
import re

def refactor_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    original = content

    # 1. Constants (Signal)
    content = content.replace('"LONG_TQQQ"', '"LONG"')
    content = content.replace("'LONG_TQQQ'", '"LONG"')
    content = content.replace('"SHORT_SQQQ"', '"SHORT"')
    content = content.replace("'SHORT_SQQQ'", '"SHORT"')

    # 2. String literals for Symbols
    # This might accidentally break strings that are already inside f-strings,
    # but let's assume standard usage like: lake.load_candles("TQQQ")
    content = re.sub(r'"TQQQ"', 'config.BASE_ASSET_LONG', content)
    content = re.sub(r"'TQQQ'", 'config.BASE_ASSET_LONG', content)
    content = re.sub(r'"SQQQ"', 'config.BASE_ASSET_SHORT', content)
    content = re.sub(r"'SQQQ'", 'config.BASE_ASSET_SHORT', content)

    # 3. Variable and feature names
    content = content.replace('tqqq_', 'long_')
    content = content.replace('_tqqq', '_long')
    content = content.replace('sqqq_', 'short_')
    content = content.replace('_sqqq', '_short')

    # MACRO
    content = content.replace('soxx_', 'macro_')
    content = content.replace('_soxx', '_macro')
    content = re.sub(r'"SOXX"', 'config.MACRO_TREND_SYMBOL', content)
    content = re.sub(r"'SOXX'", 'config.MACRO_TREND_SYMBOL', content)

    # Add config import if missing
    if ('config.BASE_ASSET_LONG' in content or 'config.BASE_ASSET_SHORT' in content or 'config.MACRO_TREND_SYMBOL' in content) and 'import config' not in content:
        if 'import pandas' in content:
            content = content.replace('import pandas', 'import config\nimport pandas', 1)
        elif 'import os' in content:
            content = content.replace('import os', 'import os\nimport config', 1)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Refactored {filepath}")

def main():
    target_dirs = ['core', 'scripts']
    for d in target_dirs:
        for root, _, files in os.walk(d):
            for file in files:
                if file.endswith('.py') and file != 'refactor_symbols.py':
                    refactor_file(os.path.join(root, file))

if __name__ == '__main__':
    main()
