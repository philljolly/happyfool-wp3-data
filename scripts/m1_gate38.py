"""Am. 38 (run 45, written before any Am. 38 data): does mixed-polarity training (M) make the trained report
answer the question (MR37 M1 gate)? Arms U (untrained), T (am36_T, intro-only), M (mixed), SM (mixed, shuffled
labels), concept rows; M random rows (descriptive). Files <dir>/a38_<arm>.json (arm U/T/M/SM/M_random), rows from
inject_loc.py with frames intro/neutral/ordinary/plain at L24 a2, N = 2, AUDIT36B concepts, test bank.
P_inj(q) = rate at which the bias-corrected report picks the injected sentence (as q_axis37.py).
Q = P_inj(intro) - P_inj(ordinary) (the trained reversal); Q' = P_inj(intro) - P_inj(plain) (untrained paraphrase
without the word "ordinary"). Acc_intro = P_inj(intro); Acc_ord = 1 - P_inj(ordinary).
GATE PASS iff Q_M >= .50 with CI lower > .25 and Acc_intro_M >= .80 and Acc_ord_M >= .80 (point estimates).
CONTROL OK iff Q_SM CI upper < .25 (else MISLABELLED: the gate is not read).
Q' slot (secondary): SHOWN iff CI excludes 0 and |Q'| >= .25; EXCLUDED iff CI inside [-.10, .10]; else NOT SHOWN AS OF
2026-10-22, channel checked: yes. UNDETERMINED (nothing read) if mean log index-token mass < ln .5 in any arm/frame.
Concept bootstrap 2000, seed 38; 95% CI = [2.5%, 97.5%] percentiles.
Usage: python3 m1_gate38.py <dir> <out.json>"""
import json, math, random, sys
import numpy as np
D, OUT = sys.argv[1], sys.argv[2]
FR = ("intro", "neutral", "ordinary", "plain")
ARMS = ("U", "T", "M", "SM", "M_random")


def pinj(rows, q):
    out = []
    for r in rows:
        L, N = r[f"logits_{q}"], r[f"none_{q}"]
        corr = (L[0] - N[0]) - (L[1] - N[1])
        out.append(float((corr > 0) == (r["target"] == 0)))
    return np.array(out)


def mass_ok(rows):
    ks = [k for k in rows[0] if k.startswith("mass_") or k.startswith("massnone_")]
    return bool(all(np.mean([r[k] for r in rows]) >= math.log(.5) for k in ks))


R = {a: json.load(open(f"{D}/a38_{a}.json"))["rows"] for a in ARMS}
cs = [r["concept"] for r in R["U"]]; con = sorted(set(cs))
for a, rows in R.items():
    assert [r["concept"] for r in rows] == cs and [r["target"] for r in rows] == [r["target"] for r in R["U"]], a
idx = {c: [i for i, g in enumerate(cs) if g == c] for c in con}
rng = random.Random(38); boots = [sum((idx[c] for c in [rng.choice(con) for _ in con]), []) for _ in range(2000)]
P = {(a, q): pinj(R[a], q) for a in ARMS for q in FR}
ALL = list(range(len(cs)))


def ci(f):
    bs = sorted(f(b) for b in boots)
    return [round(f(ALL), 3), round(bs[49], 3), round(bs[1950], 3)]


def slot(c):
    if (c[1] > 0 or c[2] < 0) and abs(c[0]) >= .25:
        return "SHOWN"
    if c[1] >= -.10 and c[2] <= .10:
        return "EXCLUDED"
    return "NOT SHOWN AS OF 2026-10-22, channel checked: yes"


def q_of(a, q2):
    return ci(lambda ii: P[(a, "intro")][ii].mean() - P[(a, q2)][ii].mean())


res = {"n_rows": len(cs), "n_concepts": len(con), "mass_ok": {a: mass_ok(R[a]) for a in ARMS}}
for a in ARMS:
    res[a] = {f"P_{q}": ci(lambda ii, q=q: P[(a, q)][ii].mean()) for q in FR}
    res[a]["Q"], res[a]["Q_plain"] = q_of(a, "ordinary"), q_of(a, "plain")
res["dQ_M_minus_T"] = ci(lambda ii: (P[("M", "intro")][ii].mean() - P[("M", "ordinary")][ii].mean())
                         - (P[("T", "intro")][ii].mean() - P[("T", "ordinary")][ii].mean()))
if not all(res["mass_ok"].values()):
    res["verdict"] = "UNDETERMINED (index-token mass below ln .5)"
else:
    qm, qs = res["M"]["Q"], res["SM"]["Q"]
    acc_i, acc_o = res["M"]["P_intro"][0], 1 - res["M"]["P_ordinary"][0]
    res["Acc_intro_M"], res["Acc_ord_M"] = acc_i, round(acc_o, 3)
    res["CONTROL"] = "OK" if qs[2] < .25 else "MISLABELLED"
    gate = qm[0] >= .50 and qm[1] > .25 and acc_i >= .80 and acc_o >= .80
    res["GATE"] = ("PASS" if gate else "FAIL") if res["CONTROL"] == "OK" else "NOT READ (control mislabelled)"
    res["Q_plain_M_slot"] = slot(res["M"]["Q_plain"]) if res["CONTROL"] == "OK" else "NOT READ"
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: res[k] for k in res if k not in ARMS}, indent=1))
for a in ARMS:
    print(a, res[a])
