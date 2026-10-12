import urllib.request
import json
url = "https://query1.finance.yahoo.com/v8/finance/chart/SQQQ"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req).read()
data = json.loads(resp)
print('SQQQ:', data['chart']['result'][0]['meta']['regularMarketPrice'])
