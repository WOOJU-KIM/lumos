with open("core/live_runner.py", "r", encoding="utf-8") as f:
    text = f.read()

if "import config" not in text:
    text = "import config\n" + text

with open("core/live_runner.py", "w", encoding="utf-8") as f:
    f.write(text)
