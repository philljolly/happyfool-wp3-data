"""Run 49: per-scale summary table of rung40 outputs (descriptive; slots exactly as rung40.py wrote them). Also applies the
Am. 40 instrument gate (INCOHERENT if mean log index mass < ln .5 in any frame; the no-injection mass is not in rung40's
output, so the 'ln 2 below no-injection' half is checked separately from the row files)."""
import json, math, sys
for f in sys.argv[1:]:
    d = json.load(open(f)); print("==", f)
    for lab, s in d.items():
        c = s["arms"][0]["frames"]
        bad = [q for q, v in c.items() if v["mass"] < math.log(.5)]
        auc = " ".join(f"{q[:3]} {v['AUC_m'][0]:.2f}[{v['AUC_m'][1]:.2f},{v['AUC_m'][2]:.2f}]" for q, v in c.items())
        pr = " ".join(f"{v.get('probe_answer_pos', float('nan')):.2f}" for v in c.values())
        rnd = s["arms"][1:]; j = [x for x in rnd if x["alpha"] == s.get("C_matched_random", {}).get("alpha")]
        rauc = " ".join(f"{q[:3]} {v['AUC_m'][0]:.2f}" for q, v in j[0]["frames"].items()) if j else ""
        print(f"{lab:6} R={s['R'][:9]:9} Q={s['Q']:18} C={s.get('C','')[:9]:9} Ci={s.get('C_intro')} | {auc} | probe {pr} | rand(m) {rauc}"
              + (f" | INCOHERENT {bad}" if bad else ""))
