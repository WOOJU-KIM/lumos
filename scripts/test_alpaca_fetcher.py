import os
import sys
import json
import urllib.request
from datetime import datetime
from pathlib import Path
import config
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

def fetch_alpaca_bars(symbol: str, timeframe: str = "15Min", start_dt: str = "2024-01-01", end_dt: str = "2024-03-31") -> pd.DataFrame:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        raise ValueError("ALPACA_API_KEY or ALPACA_SECRET_KEY is missing in .env")
        
    all_bars = []
    page_token = None
    
    base_url = "https://data.alpaca.markets/v2/stocks/bars"
    
    while True:
        url = f"{base_url}?symbols={symbol}&timeframe={timeframe}&start={start_dt}T00:00:00Z&end={end_dt}T23:59:59Z&feed=iex&adjustment=split&limit=1000"
        if page_token:
            url += f"&page_token={page_token}"
            
        req = urllib.request.Request(url, headers={
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "Accept": "application/json"
        })
        
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            bars = data.get("bars", {}).get(symbol, [])
            all_bars.extend(bars)
            
            page_token = data.get("next_page_token")
            if not page_token:
                break
                
    if not all_bars:
        return pd.DataFrame()
        
    # Convert to standard Lumos format
    rows = []
    for b in all_bars:
        # Alpaca returns UTC timestamp: e.g. 2024-01-02T14:30:00Z
        # We parse it to America/New_York (market time) or clean datetime
        utc_dt = pd.to_datetime(b["t"])
        ny_dt = utc_dt.tz_convert("America/New_York").tz_localize(None)
        
        rows.append({
            "datetime": ny_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Open": float(b["o"]),
            "High": float(b["h"]),
            "Low": float(b["l"]),
            "Close": float(b["c"]),
            "Volume": float(b["v"]),
            "VWAP": float(b.get("vw", b["c"]))
        })
        
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    return df

if __name__ == "__main__":
    print("Testing Alpaca Fetcher for TQQQ 2024 Q1 (Jan ~ Mar 2024)...")
    df = fetch_alpaca_bars(config.BASE_ASSET_LONG, timeframe="15Min", start_dt="2024-01-01", end_dt="2024-03-31")
    print(f"Fetch completed: {len(df)} bars downloaded!")
    print("\nFirst 3 bars:")
    print(df.head(3))
    print("\nLast 3 bars:")
    print(df.tail(3))
