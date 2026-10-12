import urllib.request
import json
for sym in ['SOXL', 'SOXS', 'TQQQ']:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req).read()
        data = json.loads(resp)
        print(f"{sym}:", data['chart']['result'][0]['meta']['regularMarketPrice'])
    except Exception as e:
        print(sym, e)
