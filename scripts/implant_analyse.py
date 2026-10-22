"""Amendment 22 analysis: manipulation check, implant effects per probe, SB_imp, content-free control.
Usage: python3 implant_analyse.py <evalA.json> <evalB.json> <evalC.json> <eval_untrained.json> [<evalD.json>]  (prints JSON)
y = log-odds of the "unsure"/"hard" word (as in Amendment 20); an implant should LOWER it, so
Delta_p = mean y_p(not implanted) - mean y_p(implanted) (positive = more confident when the fact is known)."""
import json, sys
import numpy as np

PROBES = ["self", "world", "person", "ai"]
EXTRA = [p for p in ("nonce", "book") if f"y_{p}" in json.load(open(sys.argv[1]))["rows"][0]]  # Amendment 22a
rows = lambda f: json.load(open(f))["rows"]
A, B, C, U = map(rows, sys.argv[1:5])
n = len(A)
# per item: implanted value (from the LoRA that learned it) and not-implanted value (from the other LoRA)
imp = {p: np.array([(A if A[i]["block"] == "A" else B)[i][f"y_{p}"] for i in range(n)]) for p in PROBES + EXTRA}
non = {p: np.array([(B if A[i]["block"] == "A" else A)[i][f"y_{p}"] for i in range(n)]) for p in PROBES + EXTRA}
gap = {p: float(np.mean([r[f"gap_{p}"] for r in A + B])) for p in PROBES + EXTRA}
ok = [p for p in PROBES[1:] if gap[p] <= 0.25]


def est(idx):
    d = {p: float(np.mean(non[p][idx] - imp[p][idx])) for p in PROBES}
    return d, (d["self"] - max(d[p] for p in ok)) if ok else float("nan")


d, sb = est(np.arange(n))
rng, boots = np.random.default_rng(22), []
for _ in range(2000):
    boots.append(est(rng.integers(0, n, n)))
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
acc = lambda R, b: sum(r["correct"] for r in R if r["block"] == b) / sum(r["block"] == b for r in R)
M = {"A_on_A": acc(A, "A"), "A_on_B": acc(A, "B"), "B_on_B": acc(B, "B"), "B_on_A": acc(B, "A")}
M["pass"] = min(M["A_on_A"], M["B_on_B"]) >= 0.8 and max(M["A_on_B"], M["B_on_A"]) <= 0.45
ctrl = {p: float(np.mean([u[f"y_{p}"] - c[f"y_{p}"] for u, c in zip(U, C)])) for p in PROBES}
# mediation (reported, not gated): within implanted items, slope of y_self on own answer entropy
Himp = np.array([(A if A[i]["block"] == "A" else B)[i]["H"] for i in range(n)])
med = float(np.polyfit(Himp, imp["self"], 1)[0]) if Himp.std() > 0 else float("nan")
out = {"manipulation": M, "order_gap": gap, "probes_in_max": ok,
       "delta": {p: {"est": d[p], "ci": ci([b[0][p] for b in boots])} for p in PROBES},
       "SB_imp": {"est": sb, "ci": ci([b[1] for b in boots])},
       "control_shift_untrained_minus_C": ctrl, "mediation_slope_self_on_H": med,
       "mean_P_unsure_self": float(np.mean(1 / (1 + np.exp(-np.concatenate([imp['self'], non['self']])))))}
dl = out["delta"]; sbci = out["SB_imp"]["ci"]
if not M["pass"]: v = "manipulation check failed"
elif sbci[0] > 0: v = "self-route"
elif dl["world"]["ci"][0] > 0: v = "projection"
elif all(dl[p]["ci"][0] <= 0 <= dl[p]["ci"][1] for p in PROBES): v = "no reach"
else: v = "other pattern"
if len(sys.argv) > 5:  # familiarity control: on items 1-20, D knows the entity (not the capital), B knows neither
    D = rows(sys.argv[5]); fa = [i for i in range(n) if A[i]["block"] == "A"]
    out["familiarity_delta_B_minus_D"] = {p: float(np.mean([B[i][f"y_{p}"] - D[i][f"y_{p}"] for i in fa])) for p in PROBES}
    out["D_accuracy_on_1_20"] = sum(D[i]["correct"] for i in fa) / len(fa)
out["verdict"] = v
if EXTRA:  # Amendment 22a: C-GA estimands AG and E (probes with order gap > 0.25 dropped)
    keep = lambda ps: [p for p in ps if p == "self" or gap[p] <= 0.25]
    ag_p, ot_p, w_p = keep(["self", "person", "ai", "nonce"]), keep(["person", "ai", "nonce"]), keep(["world", "book"])
    def cga(idx):
        dd = {p: float(np.mean(non[p][idx] - imp[p][idx])) for p in PROBES + EXTRA}
        return dd, np.mean([dd[p] for p in ag_p]) - np.mean([dd[p] for p in w_p]), dd["self"] - np.mean([dd[p] for p in ot_p])
    d2, ag, e = cga(np.arange(n)); rng2 = np.random.default_rng(221); bs = [cga(rng2.integers(0, n, n)) for _ in range(2000)]
    ci90 = lambda v: [float(np.percentile(v, 5)), float(np.percentile(v, 95))]
    agci, eci, nci = ci([b[1] for b in bs]), ci([b[2] for b in bs]), ci([b[0]["nonce"] for b in bs])
    eci90 = ci90([b[2] for b in bs])
    if not M["pass"]: v2 = "undetermined (M failed)"
    elif eci[0] > 0: v2 = "self-route"
    elif agci[0] > 0 and -0.5 < eci90[0] and eci90[1] < 0.5 and nci[0] > 0: v2 = "C-GA confirmed"
    elif agci[0] <= 0 <= agci[1] and out["delta"]["self"]["ci"][0] > 0: v2 = "hedge-general"
    else: v2 = "undetermined"
    out["A22a"] = {"delta_extra": {p: {"est": d2[p], "ci": ci([b[0][p] for b in bs])} for p in EXTRA},
                   "AG": {"est": float(ag), "ci": agci}, "E": {"est": float(e), "ci": eci, "ci90": eci90},
                   "probes": {"agents": ag_p, "others": ot_p, "nonagents": w_p}, "verdict": v2}
    # Amendment 22b (H17, descriptive only): headroom, full profile, P6 (book vs agents) and P8 (gradient) readings
    sig = lambda v: float(np.mean(1 / (1 + np.exp(-np.asarray(v)))))
    order = ["self", "ai", "person", "nonce", "book", "world"]
    head = {p: {"non_implanted": sig(non[p]), "untrained": sig([u[f"y_{p}"] for u in U])} for p in order}
    for p in order: head[p]["read"] = all(0.05 <= x <= 0.95 for x in head[p].values() if isinstance(x, float))
    agents = [p for p in ("self", "person", "ai", "nonce") if head[p]["read"]]
    p6 = [b[0]["book"] - np.mean([b[0][p] for p in agents]) for b in bs] if head["book"]["read"] and agents else []
    p6e = d2["book"] - np.mean([d2[p] for p in agents]) if p6 else float("nan")
    p6ci = ci(p6) if p6 else [float("nan")] * 2
    p6v = "P6 favoured" if p6ci[0] > 0 else "C-GA favoured" if p6ci[1] < 0 else "C-GA and P6 not separated"
    s_n = ci([b[0]["self"] - b[0]["nonce"] for b in bs])
    mono = all(d2[a] >= d2[b] for a, b in zip(order[:4], order[1:5]))
    out["A22b"] = {"headroom": head, "profile": {p: {"est": d2[p], "ci": ci([b[0][p] for b in bs])} for p in order},
                   "P6_book_minus_agents": {"est": float(p6e), "ci": p6ci, "reading": p6v},
                   "P8": {"ordering_holds": mono, "self_minus_nonce_ci": s_n,
                          "reading": "gradient" if mono and s_n[0] > 0 else "not separated"}}
    # Amendment 22d (H19, descriptive only): projection index over agents who cannot know a fictional implanted fact
    pi_p = [p for p in ("person", "ai") if gap[p] <= 0.25]
    if pi_p:
        pib = [np.mean([b[0][p] for p in pi_p]) for b in bs]
        pie, pici, pici90 = float(np.mean([d2[p] for p in pi_p])), ci(pib), ci90(pib)
        sci = out["delta"]["self"]["ci"]
        pir = ("private knowledge projected onto others" if pici[0] > 0 else
               "tracks whose knowledge it is" if -0.5 < pici90[0] and pici90[1] < 0.5 and sci[0] > 0 else "not separated")
        out["A22d"] = {"PI": {"est": pie, "ci": pici, "ci90": pici90, "probes": pi_p}, "reading": pir}
        if len(sys.argv) > 5:  # beyond familiarity, items 1-20: y(D) - y(A) (D knows the entity, A also the capital)
            fa = np.array(fa); pf = lambda ix: float(np.mean([np.mean([D[i][f"y_{p}"] - A[i][f"y_{p}"] for p in pi_p]) for i in ix]))
            rng3 = np.random.default_rng(224)
            out["A22d"]["PI_fam"] = {"est": pf(fa), "ci": ci([pf(fa[rng3.integers(0, len(fa), len(fa))]) for _ in range(2000)])}
        if len(sys.argv) > 5:  # Amendment 22f (H20, descriptive): spread contrast beyond familiarity, items 1-20
            fam = lambda ix, p: float(np.mean([D[i][f"y_{p}"] - A[i][f"y_{p}"] for i in ix]))
            scf = lambda ix: np.mean([fam(ix, p) for p in pi_p]) - fam(ix, "self")
            rng4 = np.random.default_rng(226)
            scb = [scf(fa[rng4.integers(0, len(fa), len(fa))]) for _ in range(2000)]
            sce, scci = float(scf(fa)), ci(scb)
            kf = lambda R: np.mean([np.mean([B[i][f"y_{p}"] - R[i][f"y_{p}"] for i in fa]) for p in pi_p]) / np.mean([B[i]["y_self"] - R[i]["y_self"] for i in fa])
            # Deviation 22-2 (run 28): subtract each probe's D-vs-A adapter offset, estimated on items 21-40 (neither LoRA learned them)
            fb = np.array([i for i in range(n) if A[i]["block"] == "B"])
            scc = lambda ia, ib: np.mean([fam(ia, p) - fam(ib, p) for p in pi_p]) - (fam(ia, "self") - fam(ib, "self"))
            rng5 = np.random.default_rng(228)
            sccb = [scc(fa[rng5.integers(0, len(fa), len(fa))], fb[rng5.integers(0, len(fb), len(fb))]) for _ in range(2000)]
            scce, sccci = float(scc(fa, fb)), ci(sccb)
            out["A22f"] = {"SC_uncorrected": {"est": sce, "ci": scci, "note": "Deviation 22-2: confounded by adapter offsets, not interpreted"},
                           "SC": {"est": scce, "ci": sccci, "offset_corrected": True},
                           "reading": "self-specific residue beyond familiarity" if sccci[1] < 0 else "reported, not interpreted" if sccci[0] > 0 else "one signal (not separated)",
                           "k_imp": float(pie / d2["self"]) if d2["self"] else float("nan"), "k_fam_B_minus_D": float(kf(D)),
                           "PI_label": "fluency-consistent other-ascription" if pici[0] > 0 else out["A22d"]["reading"]}
print(json.dumps(out, indent=1))
