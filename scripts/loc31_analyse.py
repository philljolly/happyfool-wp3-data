"""Amendment 31 (run 38) estimands and three-slot verdicts for one setting of inject_loc.py at Qwen3-4B.
Inputs: the concept-arm file and the random-arm (matched-norm) file of the same setting.
Per framing q: report_cor (bias-corrected argmax) and the leave-concept-out probe on h(injected) - h(none) at the primary
(final) layer, as in inject_loc_analyse.py. Concept bootstrap (2000 draws, paired by concept across framings and arms).
  E_q   = acc(report_cor, q) - acc(probe_final, q)                    slots with delta_E = .25 (probe - chance)
  F     = acc(report_cor, intro) - acc(report_cor, neutral); F_probe   slots with delta_F = .25 (mean report - chance)
  dC_q  = acc(probe_final, concept) - acc(probe_final, random)        slots with delta_C = .25 (concept probe - chance)
Channel: concept probe (final, mean of framings) CI lower > .30 and report_cor < .90 in both framings.
Also writes a slim copy of each input (rows without hidden states) with --slim <dir>.
Am. 31b (run 38, pre-data for 2AFC): --probe-floor sets the channel's probe threshold (default .30 for N = 5; .60 for
N = 2), and delta_F uses |mean report - chance| (in Am. 31 it went negative when the report sat below chance).
Am. 32 (run 39, pre-data): frames are read from the rows (intro, neutral and, if present, word). With "word":
  F_word = acc(report_cor, word) - acc(report_cor, intro)  (minimal pair "injected word" vs "injected thought");
  slots with delta = .25 |mean of the two reports - chance|. Coherence (if rows carry mass_<q>): mean index-token
  probability mass with and without injection; coherent iff, in every frame, mass_inj >= .5 and >= .5 * mass_none.
  F's slot labels now name the wording difference (H35), not "self-frame"; recorded Am. 31/31b slots are not re-scored.
  --run sets the run label in NOT SHOWN slots (default "run 38").
Usage: python3 loc31_analyse.py <concept.json> <random.json> <out.json> [--slim <dir>] [--wd 1e-2] [--probe-floor .30]
       [--run "run 39"] [--channel-from <untrained analysis.json>]
Run 40 (MR32 M2, before any trained-cell data): --channel-from takes the headroom part of the channel gate (report < .90)
from the untrained cell's analysis; the probe floor is still checked on this cell. Without it, a cell whose report reaches
the probe (the readout-patch positive control, or a successful trained model) is labelled "channel checked: no"."""
import math
import json, os, random, sys
import torch
from inject_loc_analyse import arg, probe_preds


def hits(d, wd):
    rows, n = d["rows"], d["n"]
    final = str(max(d["layers"]))
    y = torch.tensor([r["target"] for r in rows]); groups = [r["concept"] for r in rows]
    h = {}
    for q in [q for q in ("intro", "neutral", "word", "ordinary") if f"logits_{q}" in rows[0]]:
        flip = q == "ordinary"  # run 40: demand-reversed frame, correct = the non-target (N = 2 only)
        assert not flip or n == 2
        h[f"rep_{q}"] = [int((max(range(n), key=lambda j: r[f"logits_{q}"][j] - r[f"none_{q}"][j]) == r["target"]) != flip) for r in rows]
        X = torch.tensor([r[f"h_{q}"][final] for r in rows]) - torch.tensor([r[f"hnone_{q}"][final] for r in rows])
        h[f"probe_{q}"] = [int(a == b) for a, b in zip(probe_preds(X, y, groups, wd), y.tolist())]
    return h, groups, 1 / n


def slot(ci, delta, pos, neg, excl):
    est, lo, hi = ci
    if lo > 0: return pos
    if hi < 0: return neg
    if -delta < lo and hi < delta: return excl
    return "NOT SHOWN AS OF " + arg("--run", "run 38")


def main():
    dc, dr, outf = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
    wd = float(arg("--wd", "1e-2"))
    hc, gc, chance = hits(dc, wd); hr, gr, _ = hits(dr, wd)
    assert gc == gr, "arms must share concepts and row order"
    concepts = sorted(set(gc)); by_c = {c: [i for i, g in enumerate(gc) if g == c] for c in concepts}
    rng = random.Random(38); boots = [[rng.choice(concepts) for _ in concepts] for _ in range(2000)]

    def acc(h, cs): idx = [i for c in cs for i in by_c[c]]; return sum(h[i] for i in idx) / len(idx)

    def ci(f):
        bs = sorted(f(b) for b in boots); return [f(concepts), bs[49], bs[1950]]
    A = {"concept": hc, "random": hr}
    res = {"L": dc["rows"][0]["L"], "alpha": dc["rows"][0]["alpha"], "chance": chance, "n_rows": len(gc), "acc": {}}
    for arm, h in A.items():
        for k, v in h.items(): res["acc"][f"{arm}_{k}"] = ci(lambda b, v=v: acc(v, b))
    pmean = ci(lambda b: (acc(hc["probe_intro"], b) + acc(hc["probe_neutral"], b)) / 2)
    res["channel"] = {"probe_mean_final": pmean,
                      "checked": pmean[1] > float(arg("--probe-floor", ".30")) and all(res["acc"][f"concept_rep_{q}"][0] < .90 for q in ("intro", "neutral"))}
    ch_from = arg("--channel-from", "")
    if ch_from:  # Run 40 (M2, pre-M4-data): a trained/patched cell inherits the untrained cell's headroom gate, else success blanks every slot
        res["channel"]["checked_own"] = res["channel"]["checked"]
        res["channel"]["checked"] = json.load(open(ch_from))["channel"]["checked"] and pmean[1] > float(arg("--probe-floor", ".30"))
        res["channel"]["from"] = ch_from
    v = {}
    for q in ("intro", "neutral"):
        dE = .25 * (res["acc"][f"concept_probe_{q}"][0] - chance)
        e = ci(lambda b, q=q: acc(hc[f"rep_{q}"], b) - acc(hc[f"probe_{q}"], b))
        v[f"E_{q}"] = {"ci": e, "delta": dE, "slot": slot(e, dE, "GAIN (probe underfit?)", "LOSS SHOWN", "LOSS EXCLUDED")}
        dC = .25 * (res["acc"][f"concept_probe_{q}"][0] - chance)
        c = ci(lambda b, q=q: acc(hc[f"probe_{q}"], b) - acc(hr[f"probe_{q}"], b))
        v[f"dC_{q}"] = {"ci": c, "delta": dC, "slot": slot(c, dC, "CONCEPT-SPECIFIC SHOWN", "RANDOM > CONCEPT SHOWN",
                                                          "SOMETHING-ADDED (content-free) SHOWN")}
    dF = .25 * abs((res["acc"]["concept_rep_intro"][0] + res["acc"]["concept_rep_neutral"][0]) / 2 - chance)
    f = ci(lambda b: acc(hc["rep_intro"], b) - acc(hc["rep_neutral"], b))
    fp = ci(lambda b: acc(hc["probe_intro"], b) - acc(hc["probe_neutral"], b))
    s = slot(f, dF, "INTRO > NEUTRAL SHOWN (mechanism-naming vs salience wording)",
             "INTRO < NEUTRAL SHOWN (mechanism-naming vs salience wording)", "EXCLUDED")
    if s.startswith("INTRO < NEUTRAL") and not (fp[1] <= 0 <= fp[2]): s += " (but F_probe CI excludes 0)"
    v["F"] = {"ci": f, "delta": dF, "slot": s}; v["F_probe"] = {"ci": fp}
    if "rep_word" in hc:
        dW = .25 * abs((res["acc"]["concept_rep_intro"][0] + res["acc"]["concept_rep_word"][0]) / 2 - chance)
        w = ci(lambda b: acc(hc["rep_word"], b) - acc(hc["rep_intro"], b))
        v["F_word"] = {"ci": w, "delta": dW, "slot": slot(w, dW, "WORD > THOUGHT SHOWN", "THOUGHT > WORD SHOWN", "EXCLUDED")}
    r0 = dc["rows"][0]
    if any(k.startswith("mass_") for k in r0):
        coh = {}
        for q in [k[5:] for k in r0 if k.startswith("mass_")]:
            mi = sum(math.exp(r[f"mass_{q}"]) for r in dc["rows"]) / len(dc["rows"])
            mn = sum(math.exp(r[f"massnone_{q}"]) for r in dc["rows"]) / len(dc["rows"])
            coh[q] = {"mass_inj": mi, "mass_none": mn, "ok": mi >= .5 and mi >= .5 * mn}
        res["coherence"] = {"frames": coh, "coherent": all(c["ok"] for c in coh.values())}
    v["B_trigger"] = all(res["acc"][f"concept_rep_{q}"][1] <= chance for q in ("intro", "neutral"))
    if not res["channel"]["checked"]:
        for k in v:
            if isinstance(v[k], dict) and "slot" in v[k]: v[k]["slot"] = "NOT SHOWN AS OF " + arg("--run", "run 38") + " (channel checked: no)"
    res["verdicts"] = v
    print(json.dumps(res, indent=1))
    json.dump(res, open(outf, "w"), indent=1)
    sd = arg("--slim", "")
    if sd:
        os.makedirs(sd, exist_ok=True)
        for p, d in ((sys.argv[1], dc), (sys.argv[2], dr)):
            slim = {k: x for k, x in d.items() if k != "rows"}
            slim["rows"] = [{k: x for k, x in r.items() if not k.startswith(("h_", "hnone_"))} for r in d["rows"]]
            json.dump(slim, open(os.path.join(sd, os.path.basename(p).replace(".json", "_slim.json")), "w"))


if __name__ == "__main__":
    main()
