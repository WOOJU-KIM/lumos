import sqlite3, glob, os, json
db_files = glob.glob(r'c:\Users\chabo\OneDrive\바탕 화면\lumos\data\*.db')
results = {}
for db_path in db_files:
    db_name = os.path.basename(db_path)
    if os.path.getsize(db_path) == 0: continue
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        db_symbols = set()
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            cols = [row[1] for row in cursor.fetchall()]
            t_col = next((c for c in cols if c.lower() in ['symbol', 'ticker', 'code']), None)
            if t_col:
                cursor.execute(f"SELECT DISTINCT {t_col} FROM {table}")
                db_symbols.update([str(r[0]) for r in cursor.fetchall() if r[0]])
        results[db_name] = {'tables': tables, 'symbols': sorted(list(db_symbols))}
        conn.close()
    except Exception as e:
        results[db_name] = f"Error: {e}"
print(json.dumps(results, indent=2, ensure_ascii=False))
