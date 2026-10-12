import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
print("steps:", len(d["steps"]), "rev:", d.get("revision"))
for s in d["steps"]:
    print(f"#{s['index']:>3} {s['action']:<24} {str(s.get('stage')):<22} "
          f"verify={s.get('verify')} args={json.dumps(s.get('arguments'))[:90]}")
