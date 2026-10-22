"""Amendment 32 (run 39, pre-data) sweep summary. Runs loc31_analyse.py on every setting pair
<prefix>_concept_L<L>a<a>.json / <prefix>_random_L<L>a<a>.json found in <dir>, then applies the registered rules:
  selection: among coherent settings with channel checked and concept probe (final, mean of framings) in [.75, .95],
  the setting with the largest P - R, where R = max over framings of the concept report_cor (2AFC); ties -> lower alpha.
  k1 check: any coherent setting where E is LOSS EXCLUDED in some framing (report reaches probe).
Usage: python3 sweep_analyse.py <dir> <prefix> <out.json> [--run run39]"""
import glob, json, os, re, subprocess, sys
from inject_loc_analyse import arg


def main():
    d, prefix, outf = sys.argv[1], sys.argv[2], sys.argv[3]
    run = arg("--run", "run39")
    rows = []
    for fc in sorted(glob.glob(os.path.join(d, f"{prefix}_concept_L*a*.json"))):
        tag = re.search(r"_(L\d+a[\d.]+)\.json$", fc).group(1)
        fr = fc.replace("_concept_", "_random_")
        fa = os.path.join(d, f"analysis_{prefix}_{tag}.json")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "loc31_analyse.py"),
                        fc, fr, fa, "--probe-floor", "0.60", "--run", run, "--slim", os.path.join(d, "slim")],
                       check=True, stdout=subprocess.DEVNULL)
        a = json.load(open(fa)); acc, v = a["acc"], a["verdicts"]
        frames = [k[len("concept_rep_"):] for k in acc if k.startswith("concept_rep_")]
        P = a["channel"]["probe_mean_final"][0]; R = max(acc[f"concept_rep_{q}"][0] for q in frames)
        rows.append({"setting": tag, "L": a["L"], "alpha": a["alpha"], "P": P, "R": R, "P_minus_R": P - R,
                     "coherent": a.get("coherence", {}).get("coherent"), "channel": a["channel"]["checked"],
                     "rep": {q: acc[f"concept_rep_{q}"] for q in frames},
                     "slots": {k: x["slot"] for k, x in v.items() if isinstance(x, dict) and "slot" in x}})
    elig = [r for r in rows if r["coherent"] and r["channel"] and .75 <= r["P"] <= .95]
    sel = max(elig, key=lambda r: (round(r["P_minus_R"], 6), -r["alpha"])) if elig else None
    k1 = [r["setting"] for r in rows if r["coherent"] and any(s == "LOSS EXCLUDED" for k, s in r["slots"].items() if k.startswith("E_"))]
    res = {"settings": rows, "eligible": [r["setting"] for r in elig], "selected": sel and sel["setting"], "k1_settings": k1}
    json.dump(res, open(outf, "w"), indent=1)
    for r in rows:
        print(r["setting"], "P %.2f R %.2f coh %s ch %s" % (r["P"], r["R"], r["coherent"], r["channel"]),
              {q: round(x[0], 2) for q, x in r["rep"].items()})
    print("selected", res["selected"], "k1", k1)


if __name__ == "__main__":
    main()
