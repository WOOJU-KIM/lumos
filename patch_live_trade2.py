import os
from pathlib import Path

fpath = Path('scripts/live_trade_test.py')
content = fpath.read_text(encoding='utf-8')

content = content.replace('if args.real:', 'if getattr(args, "real", False):')
content = content.replace('is_sim = not getattr(args, "real", False)', 'is_sim = True')
content = content.replace('if args.mock:', 'if getattr(args, "mock", False):')

fpath.write_text(content, encoding='utf-8')
