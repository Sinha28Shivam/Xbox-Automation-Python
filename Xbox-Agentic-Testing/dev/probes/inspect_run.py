import json

data = json.load(open('artifacts/runs/run-20260904-224709/reports/execution.json'))
print('Total steps:', len(data.get('steps', [])))
for s in data.get('steps', []):
    idx = s.get('index')
    act = s.get('action')
    args = s.get('arguments')
    stg = s.get('stage')
    err = s.get('error')
    ocr = (s.get('ocr_text') or '').replace('\n', ' ')[:50]
    print(f"Step {idx:2d} | Stage: {stg:16s} | Action: {act:15s} | Args: {str(args):35s} | Err: {str(err):15s} | OCR: {ocr}")
