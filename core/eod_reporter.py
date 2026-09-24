import json
import os
import sqlite3
from datetime import datetime
import pandas as pd

class DailyReporter:
    def __init__(self, broker):
        self.broker = broker
        # Directory configuration
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # user's owl-editor is on Desktop, not inside lumos
        desktop_dir = os.path.dirname(self.base_dir) # lumos is in Desktop
        self.inbox_dir = os.path.join(desktop_dir, "owl-editor", "inbox")
        os.makedirs(self.inbox_dir, exist_ok=True)
        self.report_path = os.path.join(self.inbox_dir, "lumos_daily_report.json")
        self.db_path = os.path.join(self.base_dir, "data", "live_experience.db")
        
        self._init_db()

    def _init_db(self):
        """Initialize the DB table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_portfolio_history (
                date TEXT PRIMARY KEY,
                balance INTEGER,
                return_pct REAL
            )
        ''')
        conn.commit()
        conn.close()
        
    def _get_db_history(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT date, balance, return_pct FROM daily_portfolio_history ORDER BY date ASC')
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for r in rows:
            history.append({
                "date": r[0],
                "balance": int(r[1]),
                "return_pct": float(r[2])
            })
        return history

    def generate_report(self):
        """
        Fetches current balance, updates DB, and exports 3-month history to JSON
        """
        try:
            # 1. Fetch current status from broker
            broker_report = self.broker.get_official_broker_report()
            current_balance = int(broker_report.get("total_eval_krw", 0))
            today_date = datetime.now().strftime("%Y-%m-%d")
            
            # 2. Get history from DB
            history = self._get_db_history()
            
            # 3. Calculate today's return
            # Primary: Use broker's official realized return
            broker_rate = float(broker_report.get("realized_rate_pct", 0.0))
            
            if broker_rate != 0.0:
                today_return_pct = broker_rate
            else:
                # Fallback: Calculate based on DB balance change if broker returns 0.0
                today_return_pct = 0.0
                if history and history[-1]["date"] != today_date:
                    prev_balance = history[-1]["balance"]
                    if prev_balance > 0:
                        today_return_pct = round(((current_balance - prev_balance) / prev_balance) * 100.0, 2)
                elif history and history[-1]["date"] == today_date:
                    # Same day re-run
                    prev_balance = history[-2]["balance"] if len(history) > 1 else current_balance
                    if prev_balance > 0:
                        today_return_pct = round(((current_balance - prev_balance) / prev_balance) * 100.0, 2)
                    
            # 4. Upsert into DB
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT OR REPLACE INTO daily_portfolio_history (date, balance, return_pct) VALUES (?, ?, ?)',
                (today_date, current_balance, today_return_pct)
            )
            conn.commit()
            conn.close()
            
            # 5. Fetch updated history (last 90 days)
            updated_history = self._get_db_history()
            final_history = updated_history[-90:]
            
            # 6. Save report to JSON
            report_data = {
                "today_date": today_date,
                "today_return_pct": today_return_pct,
                "current_balance": current_balance,
                "history": final_history
            }
            
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
                
            from core.system_logger import system_logger
            system_logger.log("INFO", "EODReporter", f"Daily report generated successfully from DB -> {self.report_path}")
            return report_data
            
        except Exception as e:
            from core.system_logger import system_logger
            system_logger.log("ERROR", "EODReporter", f"Failed to generate daily report: {e}")
            return None
