import json, urllib.request
k = [l.split('=', 1)[1].strip() for l in open('.env', encoding='utf-8')
     if l.startswith('ANTHROPIC_API_KEY')][0]
r = urllib.request.Request('https://api.anthropic.com/v1/models?limit=40',
                           headers={'x-api-key': k, 'anthropic-version': '2023-06-01'})
for m in json.load(urllib.request.urlopen(r))['data']:
    print(m['id'])
