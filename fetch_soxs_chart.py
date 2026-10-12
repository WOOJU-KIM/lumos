import urllib.request
import json
url = "https://query1.finance.yahoo.com/v8/finance/chart/SOXS?interval=15m&range=1d"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req).read()
data = json.loads(resp)
timestamps = data['chart']['result'][0]['timestamp']
highs = data['chart']['result'][0]['indicators']['quote'][0]['high']
for t, h in zip(timestamps, highs):
    import datetime
    dt = datetime.datetime.fromtimestamp(t)
    print(dt, h)
