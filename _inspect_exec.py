import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding='utf-8'))
print("aborted:", d.get("aborted"), "|", d.get("abort_reason"))
print("total:", d.get("total_steps"), "dispatched:", d.get("dispatched_steps"))
for s in d.get("steps", []):
    print(f"#{s['index']:>3} {s['action']:<24} stage={str(s.get('stage')):<22} "
          f"disp={s.get('dispatched')} delta={s.get('screen_delta')} "
          f"status={s.get('stage_status')} err={(s.get('error') or '')[:70]}")
print("\nSTAGES:")
for st in d.get("stage_summary", []):
    print(f"  {st.get('stage')}: {st.get('status')} - {str(st.get('summary'))[:90]}")
print("\nNOTES:", d.get("notes"))
