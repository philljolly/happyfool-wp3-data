"""Am. 36 (run 42, MR32 M3/M4, pre-registered in prereg.md before any data): access audit of IFT-style training at a GAP
setting. Inputs: concept-arm 2AFC rows of inject_loc.py for one setting, from the untrained model (U), the trained adapter
(T) and the shuffled-label adapter (S), same concepts, sentences and seeds (rows paired by index), with hidden states at a
mid layer (default 30) and the final layer. The LoRA is attention-only, so the final norm and W_U are untouched and the
final-layer readout direction u = W_U["1"] - W_U["2"] is the same for all three arms (k3 checks it per arm).
Per arm and frame (s = +1 if the target is sentence 1, else -1; dh = h(inj) - h(none); m = mean s*dh):
  R      report accuracy: frames intro/neutral pick the target, frame ordinary picks the NON-target (scored as correct)
  P_l    leave-concept-out logistic probe refit on this arm at layer l (H34: the reader is refitted per model)
  X_l    transfer: the probe fitted on all U rows at layer l, applied to this arm's rows
  cos_u  cos(u, m) at the final layer; b = mean s*dh.u_hat
  mid    cos(m_U, m_arm) and a-ratio |m_arm|/|m_U| at the mid layer
Paired concept bootstrap (2000, seed 36). Slots (registered; intro is the trained frame):
  K3      per-arm agreement of s*dh.u > 0 with the logged corrected report >= .98, else nothing is read
  GATE    R_T - R_U (intro) >= .15 with CI lower > 0           -> PASS / FAIL (k2 counts a FAIL)
  CONTROL R_S - R_U (intro) CI upper < .15                      -> OK / MISLABELLED (k3: nothing below is read)
  Stories, read only if GATE PASS and CONTROL OK (each slot SHOWN / NOT SHOWN independently; "not separated" if 0 or >1):
    OPENING   mid cos(m_U, m_T) CI lower >= .80 and mid a-ratio CI within [.75, 1.33] and final cos_u(T) - cos_u(U) CI lower > 0
    REROUTE   mid cos(m_U, m_T) CI upper < .80 and X_mid(T) >= P_mid(U) - .05 (point) [same information, moved]
    DETECTOR  P_mid(T) - P_mid(U) CI lower > 0 and X_mid(T) <= P_mid(U) - .10 (point) [new readable component]
  FRAME     frame spread (H36): gains R_T - R_U in neutral and ordinary; FRAME-LOCAL iff both CIs include 0
Usage: python3 audit36.py <wu.json> <U.json> <T.json> <S.json> <out.json> [--mid 30] [--frames intro,neutral,ordinary]"""
import json, random, sys
import numpy as np
import torch
from inject_loc_analyse import probe_preds


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def fit_apply(Xtr, ytr, Xte, wd=1e-2, steps=300):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    W = torch.zeros(Xtr.shape[1], 2, requires_grad=True); b = torch.zeros(2, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(((Xtr - mu) / sd) @ W + b, ytr) + wd * (W ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        return (((Xte - mu) / sd) @ W + b).argmax(1).tolist()


def main():
    wu = json.load(open(sys.argv[1]))
    arms = {k: json.load(open(p))["rows"] for k, p in zip("UTS", sys.argv[2:5])}
    outf, mid = sys.argv[5], arg("--mid", "30")
    fin = str(max(int(x) for x in arms["U"][0]["h_intro"]))
    frames = [q for q in arg("--frames", "intro,neutral,ordinary").split(",") if f"h_{q}" in arms["U"][0]]
    U = arms["U"]
    for k, rows in arms.items():  # paired design check
        assert [(r["concept"], r["target"], r["sents"]) for r in rows] == [(r["concept"], r["target"], r["sents"]) for r in U], k
    u = np.array(wu["rows"][0]) - np.array(wu["rows"][1]); uh = u / np.linalg.norm(u)
    s = np.array([1.0 if r["target"] == 0 else -1.0 for r in U])
    y = torch.tensor([r["target"] for r in U]); cs = [r["concept"] for r in U]; concepts = sorted(set(cs))
    idx = {c: [i for i, g in enumerate(cs) if g == c] for c in concepts}
    rng = random.Random(36); boots = [sum((idx[c] for c in [rng.choice(concepts) for _ in concepts]), []) for _ in range(2000)]
    allr = list(range(len(U)))

    def ci(f):
        bs = sorted(f(b) for b in boots); return [f(allr), bs[49], bs[1950]]

    def sdh(rows, q, l):
        return s[:, None] * (np.array([r[f"h_{q}"][l] for r in rows]) - np.array([r[f"hnone_{q}"][l] for r in rows]))
    A, res = {}, {"L": U[0]["L"], "alpha": U[0]["alpha"], "n_rows": len(U), "mid": mid, "final": fin, "arms": {}}
    for q in frames:
        flip = q == "ordinary"
        Xu = {l: torch.tensor(sdh(U, q, l) * s[:, None], dtype=torch.float32) for l in (mid, fin)}  # dh (unsigned) for probes
        for k, rows in arms.items():
            corr = np.array([(r[f"logits_{q}"][0] - r[f"none_{q}"][0]) - (r[f"logits_{q}"][1] - r[f"none_{q}"][1]) for r in rows])
            pick0 = corr > 0
            rep = np.array([float((p == (r["target"] == 0)) != flip) for p, r in zip(pick0, rows)])
            sdf, sdm = sdh(rows, q, fin), sdh(rows, q, mid)
            k3 = float(((s * (sdf @ u) > 0) == pick0).mean())  # dh.u > 0 must reproduce the logged choice
            X = {l: torch.tensor(sdh(rows, q, l) * s[:, None], dtype=torch.float32) for l in (mid, fin)}
            P = {l: np.array([float(a == b) for a, b in zip(probe_preds(X[l], y, cs, 1e-2), y.tolist())]) for l in (mid, fin)}
            Xt = np.array([float(a == b) for a, b in zip(fit_apply(Xu[mid], y, X[mid]), y.tolist())])
            A[(k, q)] = {"rep": rep, "P_mid": P[mid], "P_fin": P[fin], "X_mid": Xt, "sdf": sdf, "sdm": sdm, "k3": k3}

    def cosv(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    for q in frames:
        for k in "UTS":
            F, Fu = A[(k, q)], A[("U", q)]
            res["arms"][f"{k}_{q}"] = {
                "k3": F["k3"], "R": ci(lambda ii: F["rep"][ii].mean()), "P_mid": ci(lambda ii: F["P_mid"][ii].mean()),
                "P_fin": ci(lambda ii: F["P_fin"][ii].mean()), "X_mid": ci(lambda ii: F["X_mid"][ii].mean()),
                "cos_u": ci(lambda ii: cosv(F["sdf"][ii].mean(0), u)), "b": ci(lambda ii: float((F["sdf"][ii] @ uh).mean())),
                "mid_cos_U": ci(lambda ii: cosv(F["sdm"][ii].mean(0), Fu["sdm"][ii].mean(0))),
                "mid_a_ratio": ci(lambda ii: float(np.linalg.norm(F["sdm"][ii].mean(0)) / np.linalg.norm(Fu["sdm"][ii].mean(0)))),
                "dR_vs_U": ci(lambda ii: F["rep"][ii].mean() - Fu["rep"][ii].mean()),
                "dP_mid_vs_U": ci(lambda ii: F["P_mid"][ii].mean() - Fu["P_mid"][ii].mean()),
                "dcos_u_vs_U": ci(lambda ii: cosv(F["sdf"][ii].mean(0), u) - cosv(Fu["sdf"][ii].mean(0), u))}
    g = res["arms"]
    sl = {"K3": "PASS" if all(g[f"{k}_{q}"]["k3"] >= .98 for k in "UTS" for q in frames) else "FAIL"}
    dT, dS = g["T_intro"]["dR_vs_U"], g["S_intro"]["dR_vs_U"]
    sl["GATE"] = "PASS" if dT[0] >= .15 and dT[1] > 0 else "FAIL"
    sl["CONTROL"] = "OK" if dS[2] < .15 else "MISLABELLED"
    if sl["K3"] == "PASS" and sl["GATE"] == "PASS" and sl["CONTROL"] == "OK":
        t, pu = g["T_intro"], g["U_intro"]["P_mid"][0]
        st = {"OPENING": t["mid_cos_U"][1] >= .80 and .75 <= t["mid_a_ratio"][1] and t["mid_a_ratio"][2] <= 1.33 and t["dcos_u_vs_U"][1] > 0,
              "REROUTE": t["mid_cos_U"][2] < .80 and t["X_mid"][0] >= pu - .05,
              "DETECTOR": t["dP_mid_vs_U"][1] > 0 and t["X_mid"][0] <= pu - .10}
        sl.update({k: "SHOWN" if v else "NOT SHOWN" for k, v in st.items()})
        sl["STORY"] = [k for k, v in st.items() if v][0] if sum(st.values()) == 1 else "NOT SEPARATED"
        oth = [q for q in frames if q != "intro"]
        sl["FRAME"] = "FRAME-LOCAL" if oth and all(g[f"T_{q}"]["dR_vs_U"][1] <= 0 <= g[f"T_{q}"]["dR_vs_U"][2] for q in oth) else "SPREAD"
    else:
        sl["STORY"] = "NOT READ"
    res["slots"] = sl
    json.dump(res, open(outf, "w"), indent=1)
    print(json.dumps(sl))
    for q in frames:
        for k in "UTS":
            a = g[f"{k}_{q}"]
            print(q, k, "R %.2f [%.2f,%.2f]" % tuple(a["R"]), "Pmid %.2f" % a["P_mid"][0], "Pfin %.2f" % a["P_fin"][0],
                  "Xmid %.2f" % a["X_mid"][0], "cos_u %.2f" % a["cos_u"][0], "midcos %.2f" % a["mid_cos_U"][0], "k3 %.2f" % a["k3"])


if __name__ == "__main__":
    main()
