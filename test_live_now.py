import os, sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["KIWOOM_IS_SIMULATION"] = "1"

# We will patch LiveRunner to ignore time restrictions and just trade.
# But wait, how do we trigger a trade if there are no ML signals?
# The user wants to "review that day's market and check if trading happens normally".
# This implies they DO want to run the feeder with historical data from that day!
