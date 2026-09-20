with open("core/live_runner.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace('"TQQQ", "SQQQ", "QQQ", "NVDA", "QQQ"', '"TQQQ", "SQQQ", "QQQ", "NVDA"')
text = text.replace('"QQQ": self.ws_streamer.get_latest_price("QQQ", 0.0),\n                                "QQQ": self.ws_streamer.get_latest_price("QQQ", 0.0),', '"QQQ": self.ws_streamer.get_latest_price("QQQ", 0.0),')

with open("core/live_runner.py", "w", encoding="utf-8") as f:
    f.write(text)
