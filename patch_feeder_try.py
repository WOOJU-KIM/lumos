import os
from pathlib import Path

fpath = Path('core/mock_kiwoom_feeder.py')
content = fpath.read_text(encoding='utf-8')

replacement = """
                        for cb in self._callbacks:
                            cb(sym, price, {"type": "mock"})
"""

idx1 = content.find("for cb in self._callbacks:")
idx2 = content.find("time.sleep(0.02)")

content = content[:idx1] + replacement.strip() + "\n            \n            " + content[idx2:]
fpath.write_text(content, encoding='utf-8')
