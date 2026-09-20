import os
from pathlib import Path

fpath = Path('core/ml_engine.py')
content = fpath.read_text(encoding='utf-8')

target = """                    elif hasattr(obj, "gbdt_engine") and getattr(obj.gbdt_engine, "model", None) is not None:
                        self.model = obj.gbdt_engine.model
                        self.feature_names = getattr(obj.gbdt_engine, "feature_names", [])
                        self.top_10_features = getattr(obj.gbdt_engine, "top_10_features", [])
                        self.top_3_features = getattr(obj.gbdt_engine, "top_3_features", [])
                        break"""

replacement = target + """
                    elif type(obj).__name__ == "LGBMClassifier":
                        self.model = obj
                        self.feature_names = getattr(obj, "feature_name_", [])
                        self.top_10_features = []
                        self.top_3_features = []
                        break"""

content = content.replace(target, replacement)
fpath.write_text(content, encoding='utf-8')
