from pathlib import Path
p = Path('core/live_runner.py')
c = p.read_text(encoding='utf-8')
c = c.replace('args, unknown = parser.parse_known_args()', 'parser.add_argument("--real", action="store_true")\n    parser.add_argument("--sim", action="store_true")\n    args, unknown = parser.parse_known_args()')
p.write_text(c, encoding='utf-8')
