import subprocess
import shutil
import time

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# 1. Wait for current True backtest to finish (or just assume we rename its output when it's done)
# We will do this manually in powershell
