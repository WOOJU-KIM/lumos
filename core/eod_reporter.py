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
                return_pct REAL,
                balance_usd REAL DEFAULT 0.0,
                return_pct_usd REAL DEFAULT 0.0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_trade_pnl (
                us_session_date TEXT PRIMARY KEY,
                total_trades INTEGER,
                win_trades INTEGER,
                win_rate_pct REAL,
                total_pnl_usd REAL
            )
        ''')
        
        try:
            cursor.execute('ALTER TABLE daily_portfolio_history ADD COLUMN balance_usd REAL DEFAULT 0.0')
            cursor.execute('ALTER TABLE daily_portfolio_history ADD COLUMN return_pct_usd REAL DEFAULT 0.0')
        except sqlite3.OperationalError:
            pass # Columns already exist
            
        conn.commit()
        conn.close()
        
    def _get_db_history(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT date, balance, return_pct, balance_usd, return_pct_usd FROM daily_portfolio_history ORDER BY date ASC')
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for r in rows:
            history.append({
                "date": r[0],
                "balance": int(r[1]),
                "return_pct": float(r[2]),
                "balance_usd": float(r[3]) if r[3] is not None else 0.0,
                "return_pct_usd": float(r[4]) if r[4] is not None else 0.0
            })
        return history

    def generate_report(self):
        """
        Fetches current balance, updates DB, and exports 3-month history to JSON
        """
        try:
            from zoneinfo import ZoneInfo
            # 1. Fetch current status from broker
            broker_report = self.broker.get_official_broker_report()
            current_balance = int(broker_report.get("total_eval_krw", 0))
            current_balance_usd = float(broker_report.get("total_eval_usd", 0.0))
            
            # Use New York time to ensure the EOD date matches the US trading day
            today_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
            
            # 2. Get history from DB
            history = self._get_db_history()
            
            # 3. Calculate today's return
            # Always calculate based on DB balance change to reflect true portfolio return
            today_return_pct = 0.0
            today_return_pct_usd = 0.0
            
            if history and history[-1]["date"] != today_date:
                prev_balance = history[-1]["balance"]
                prev_balance_usd = history[-1]["balance_usd"]
                if prev_balance > 0:
                    today_return_pct = round(((current_balance - prev_balance) / prev_balance) * 100.0, 2)
                if prev_balance_usd > 0:
                    today_return_pct_usd = round(((current_balance_usd - prev_balance_usd) / prev_balance_usd) * 100.0, 2)
            elif history and history[-1]["date"] == today_date:
                # Same day re-run
                prev_balance = history[-2]["balance"] if len(history) > 1 else current_balance
                prev_balance_usd = history[-2]["balance_usd"] if len(history) > 1 else current_balance_usd
                if prev_balance > 0:
                    today_return_pct = round(((current_balance - prev_balance) / prev_balance) * 100.0, 2)
                if prev_balance_usd > 0:
                    today_return_pct_usd = round(((current_balance_usd - prev_balance_usd) / prev_balance_usd) * 100.0, 2)
                    
            # 4. Upsert into DB
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT OR REPLACE INTO daily_portfolio_history (date, balance, return_pct, balance_usd, return_pct_usd) VALUES (?, ?, ?, ?, ?)',
                (today_date, current_balance, today_return_pct, current_balance_usd, today_return_pct_usd)
            )
            conn.commit()

            # --- Update pure trade PnL history ---
            try:
                df = pd.read_sql_query('SELECT * FROM live_trades', conn)
                if not df.empty:
                    df['entry_time'] = pd.to_datetime(df['entry_time'])
                    df['us_session_date'] = (df['entry_time'] - pd.Timedelta(hours=12)).dt.date.astype(str)
                    df = df[df['actual_entry_price'] > 0] # Filter glitch
                    
                    daily = df.groupby('us_session_date').agg(
                        total_trades=('trade_id', 'count'),
                        win_trades=('pnl_usd', lambda x: (x > 0).sum()),
                        total_pnl_usd=('pnl_usd', 'sum')
                    ).reset_index()
                    daily['win_rate_pct'] = round((daily['win_trades'] / daily['total_trades']) * 100, 1)
                    daily['total_pnl_usd'] = round(daily['total_pnl_usd'], 2)
                    
                    for _, row in daily.iterrows():
                        cursor.execute('''
                        INSERT OR REPLACE INTO daily_trade_pnl 
                        (us_session_date, total_trades, win_trades, win_rate_pct, total_pnl_usd) 
                        VALUES (?, ?, ?, ?, ?)
                        ''', (row['us_session_date'], row['total_trades'], row['win_trades'], row['win_rate_pct'], row['total_pnl_usd']))
                    conn.commit()
            except Exception as e:
                from core.system_logger import system_logger
                system_logger.log("WARNING", "EODReporter", f"Failed to update daily_trade_pnl: {e}")
            
            conn.close()
            
            # 5. Fetch updated history (last 90 days)
            updated_history = self._get_db_history()
            final_history = updated_history[-90:]
            
            # 6. Save report to JSON
            report_data = {
                "today_date": today_date,
                "today_return_pct": today_return_pct,
                "today_return_pct_usd": today_return_pct_usd,
                "current_balance": current_balance,
                "current_balance_usd": current_balance_usd,
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
