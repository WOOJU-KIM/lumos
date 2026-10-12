import urllib.request
import json
url = "https://query1.finance.yahoo.com/v8/finance/chart/SOXS?interval=15m&range=1d"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req).read()
data = json.loads(resp)
timestamps = data['chart']['result'][0]['timestamp']
quotes = data['chart']['result'][0]['indicators']['quote'][0]
for t, o, h, l, c in zip(timestamps, quotes['open'], quotes['high'], quotes['low'], quotes['close']):
    import datetime
    dt = datetime.datetime.fromtimestamp(t)
    if '00:45' in str(dt):
        print(f"SOXS 00:45 -> Open: {o}, High: {h}, Low: {l}, Close: {c}")
