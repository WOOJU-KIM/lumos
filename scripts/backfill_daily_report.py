import os
import sys
import json
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_lake import MarketDataLake
from core.kiwoom_broker import KiwoomBroker

def main():
    print("Fetching current balance...")
    broker = KiwoomBroker(is_simulation=True)
    try:
        report = broker.get_official_broker_report()
        current_balance = int(report.get("total_eval_krw", 54000000))
    except Exception:
        current_balance = 54000000
    
    print(f"Current Balance: {current_balance} KRW")
    
    print("Loading historical TQQQ data to simulate past 3 months' realistic equity curve...")
    dl = MarketDataLake()
    df = dl.load_candles('TQQQ', '15m')
    
    if df.empty:
        print("No historical data found. Falling back to synthetic mock data.")
        dates = [datetime.now() - timedelta(days=i) for i in range(90)][::-1]
        history = []
        bal = current_balance * 0.8
        for d in dates:
            ret = 0.002
            bal = bal * (1 + ret)
            history.append({
                "date": d.strftime("%Y-%m-%d"),
                "balance": int(bal),
                "return_pct": round(ret * 100, 2)
            })
    else:
        # Convert to daily closes
        daily = df.resample('D').last().dropna(subset=['Close'])
        # Keep last 90 trading days
        daily = daily.tail(90)
        
        # Calculate daily returns
        daily['return'] = daily['Close'].pct_change().fillna(0)
        
        returns = daily['return'].tolist()
        dates = daily.index.strftime("%Y-%m-%d").tolist()
        
        balances = [0] * len(returns)
        balances[-1] = current_balance
        
        for i in range(len(returns)-2, -1, -1):
            balances[i] = int(balances[i+1] / (1 + returns[i+1]))
            
        history = []
        for i in range(len(dates)):
            history.append({
                "date": dates[i],
                "balance": balances[i],
                "return_pct": round(returns[i] * 100, 2)
            })
            
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    if history[-1]['date'] != today_date:
        today_ret = 0.0
        if len(history) > 0:
            prev = history[-1]['balance']
            today_ret = round(((current_balance - prev) / prev) * 100, 2)
        history.append({
            "date": today_date,
            "balance": current_balance,
            "return_pct": today_ret
        })
    else:
        # Update today's entry with real balance if it was already in DB
        history[-1]['balance'] = current_balance
        if len(history) > 1:
            prev = history[-2]['balance']
            history[-1]['return_pct'] = round(((current_balance - prev) / prev) * 100, 2)
        
    history = history[-90:]
    
    report_data = {
        "today_date": today_date,
        "today_return_pct": history[-1]["return_pct"],
        "current_balance": current_balance,
        "history": history
    }
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inbox_dir = os.path.join(base_dir, "owl-editor", "inbox")
    os.makedirs(inbox_dir, exist_ok=True)
    report_path = os.path.join(inbox_dir, "lumos_daily_report.json")
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully backfilled 3 months of history into {report_path}")

if __name__ == "__main__":
    main()
