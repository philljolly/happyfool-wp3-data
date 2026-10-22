"""Am. 35 (run 41, pre-registered in prereg.md): readout geometry of the untrained 2AFC localisation rows (MR34).
At the final layer, Qwen3's last hidden state is post-norm and the logits are exactly h . W_U (tied embeddings; checked run 41:
logits_intro reproduced to 4 digits from h_intro['36'] and the two index-token rows). So the report's bias-corrected
evidence for sentence 1 over 2 is exactly dh . u, with dh = h(inj) - h(none) and u = W_U["1"] - W_U["2"].
Per row: s = +1 if target is sentence 1 else -1. Per frame q:
  report hit  = s * (dh . u) > 0        (k3: must agree with the logged bias-corrected report on >= .98 of rows)
  d_loo       = leave-concept-out difference-of-means direction of s * dh (the 'content' direction a linear probe can use)
  probe hit   = s * (dh . d_loo) > 0
  a_q = mean s * dh . d_hat (content along the probe direction), b_q = mean s * dh . u_hat (usage along the readout)
  cos(u, d), cos(m_q, m_r) between frames, with m_q = mean s * dh
Concept-bootstrap 95% CIs (2000, seed 35). Slots (registered):
  GAP SHOWN       iff |cos(u, m)| CI (both bounds) < .20 and a_q CI lower > 0 and the report CI includes or is below .5
  SIGN FLIP SHOWN iff b_intro and b_neutral CIs exclude 0 with opposite signs, and a_intro, a_neutral CI lower > 0
  UPSTREAM CONTENT EFFECT iff (a_intro - a_neutral) CI excludes 0 and |diff| > .25 * mean(a_intro, a_neutral)
Usage: python3 geom35.py <wu.json> <concept_rows.json> <out.json> [--frames intro,neutral,word]"""
import json, random, sys
import numpy as np


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    wu, d, outf = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
    rows = d["rows"]; assert d["n"] == 2
    fin = str(max(d["layers"]))
    frames = [q for q in arg("--frames", "intro,neutral,word,ordinary,own,foreign").split(",") if f"h_{q}" in rows[0]]
    u = np.array(wu["rows"][0]) - np.array(wu["rows"][1]); uh = u / np.linalg.norm(u)
    s = np.array([1.0 if r["target"] == 0 else -1.0 for r in rows])
    cs = [r["concept"] for r in rows]; concepts = sorted(set(cs))
    idx = {c: [i for i, g in enumerate(cs) if g == c] for c in concepts}
    rng = random.Random(35); boots = [[rng.choice(concepts) for _ in concepts] for _ in range(2000)]
    S = {}
    for q in frames:
        dh = np.array([r[f"h_{q}"][fin] for r in rows]) - np.array([r[f"hnone_{q}"][fin] for r in rows])
        sd = s[:, None] * dh
        logged = np.array([int((r[f"logits_{q}"][0] - r[f"none_{q}"][0]) - (r[f"logits_{q}"][1] - r[f"none_{q}"][1]) > 0) == int(r["target"] == 0)
                           for r in rows])
        rep = (sd @ u > 0).astype(float)
        dl = np.zeros_like(sd)
        for c in concepts:
            rest = [i for i in range(len(rows)) if cs[i] != c]
            v = sd[rest].mean(0); dl[idx[c]] = v / np.linalg.norm(v)
        dfull = sd.mean(0) / np.linalg.norm(sd.mean(0))
        S[q] = {"sd": sd, "rep": rep, "probe": (np.einsum("ij,ij->i", sd, dl) > 0).astype(float),
                "a": np.einsum("ij,ij->i", sd, dl), "b": sd @ uh, "k3": float((rep == logged).mean()), "dfull": dfull}

    def ci(f):
        bs = sorted(f(sum((idx[c] for c in b), [])) for b in boots); return [f(sum((idx[c] for c in concepts), [])), bs[49], bs[1950]]

    def cosd(q, ii):
        v = S[q]["sd"][ii].mean(0); return float(v @ uh / np.linalg.norm(v))
    res = {"L": rows[0]["L"], "alpha": rows[0]["alpha"], "n_rows": len(rows), "frames": {}}
    for q in frames:
        F = S[q]
        res["frames"][q] = {"k3_agree": F["k3"], "report": ci(lambda ii: F["rep"][ii].mean()),
                            "probe_dm": ci(lambda ii: F["probe"][ii].mean()),
                            "a_content": ci(lambda ii: F["a"][ii].mean()), "b_usage": ci(lambda ii: F["b"][ii].mean()),
                            "cos_u_m": ci(lambda ii: cosd(q, ii)), "norm_m": float(np.linalg.norm(F["sd"].mean(0)))}
        r = res["frames"][q]
        r["GAP"] = "SHOWN" if (abs(r["cos_u_m"][1]) < .2 and abs(r["cos_u_m"][2]) < .2 and r["a_content"][1] > 0
                               and r["report"][1] <= .5) else "NOT SHOWN"
    pairs = {}
    for i, q in enumerate(frames):
        for p in frames[i + 1:]:
            pairs[f"{q}|{p}"] = {"cos_m": ci(lambda ii: float(S[q]["sd"][ii].mean(0) @ S[p]["sd"][ii].mean(0)
                                                         / np.linalg.norm(S[q]["sd"][ii].mean(0)) / np.linalg.norm(S[p]["sd"][ii].mean(0)))),
                                 "a_diff": ci(lambda ii: S[q]["a"][ii].mean() - S[p]["a"][ii].mean()),
                                 "b_diff": ci(lambda ii: S[q]["b"][ii].mean() - S[p]["b"][ii].mean())}
    res["pairs"] = pairs
    if "intro" in S and "neutral" in S:
        fi, fn = res["frames"]["intro"], res["frames"]["neutral"]
        opp = (fi["b_usage"][2] < 0 < fn["b_usage"][1]) or (fn["b_usage"][2] < 0 < fi["b_usage"][1])
        res["SIGN_FLIP"] = "SHOWN" if opp and fi["a_content"][1] > 0 and fn["a_content"][1] > 0 else "NOT SHOWN"
        ad = pairs["intro|neutral"]["a_diff"]; am = (fi["a_content"][0] + fn["a_content"][0]) / 2
        res["UPSTREAM_CONTENT"] = "SHOWN" if (ad[1] > 0 or ad[2] < 0) and abs(ad[0]) > .25 * abs(am) else "NOT SHOWN"
    json.dump(res, open(outf, "w"), indent=1)
    for q, r in res["frames"].items():
        print(res["L"], res["alpha"], q, "k3", round(r["k3_agree"], 3), "rep", [round(x, 2) for x in r["report"]],
              "probe", [round(x, 2) for x in r["probe_dm"]], "a", [round(x, 2) for x in r["a_content"]],
              "b", [round(x, 2) for x in r["b_usage"]], "cos(u,m)", [round(x, 2) for x in r["cos_u_m"]], r["GAP"])
    print({k: v for k, v in res.items() if k in ("SIGN_FLIP", "UPSTREAM_CONTENT")},
          {k: [round(x, 2) for x in v["cos_m"]] for k, v in pairs.items()})


if __name__ == "__main__":
    main()
