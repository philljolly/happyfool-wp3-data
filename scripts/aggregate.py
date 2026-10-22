"""WP3 pre-registered aggregate (prereg Analysis; Amendment 1; Deviations 2-3; Amendment 5).
Reads results/<tag>_stage{1,2b,2c}.json for every single-word tag present, and
results/graft_{c,g}_<word>_stage{2b,2c}.json (Delta_M from graft_{c,g}_graft.json) for the jointly
trained WP3b LoRAs. Prints, per word, Delta_M, rating self - base_self (H1), self - Bm_2 (k* sign),
and a sign test across treated words.

Amendment 5 decision rule for the self-vs-Bm_2 gap, with bands from controls of the SAME training
design ("single" = pilot.py one-word LoRA, "joint" = graft.py), excluding the model being judged:
- observer > self: self < every seeded Bm_2 draw, and Bm_2 mean > max |Bm_2| on those controls;
- self > observer: self > every draw, and self - base_self > max |self - base_self| on those controls;
- otherwise, or with no such control: indeterminate. Pre-amendment files have one unseeded draw
  and are marked "1 draw".
Usage: python3 aggregate.py"""
import json, os, math

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(name):
    p = os.path.join(R, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def sign_p(k, n):
    """Two-sided exact sign test."""
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def collect():
    rows = []
    for t in sorted({f.rsplit("_stage", 1)[0] for f in os.listdir(R) if f.endswith("_stage2c.json")}):
        s2b, s2c = load(f"{t}_stage2b"), load(f"{t}_stage2c")
        if t.startswith("graft_"):
            g = load(f"{t[:7]}_graft")  # graft_c / graft_g
            word, design, control = t[8:], "joint", t.startswith("graft_c")
            dM = g["delta_M"].get(word) if g else None
        else:
            s1 = load(f"{t}_stage1")
            word, design, control = t.replace("_control", ""), "single", "control" in t
            dM = s1["delta_M"] if s1 else None
        if dM is None or not (s2b and s2c):
            continue
        rows.append({"tag": t, "word": word, "design": design, "control": control, "dM": dM,
                     "self": s2b["rating_self"], "base": s2b["rating_base_self"],
                     "Bm0": s2c["rating_Bm_0"], "Bm2": s2c["rating_Bm_2"], "Bm10": s2c["rating_Bm_10"],
                     "draws": s2c.get("rating_Bm_2_draws"), "sd": s2c.get("rating_Bm_2_sd")})
    for r in rows:
        r["H1"] = r["self"] - r["base"]
        r["self_minus_Bm2"] = r["self"] - r["Bm2"]
    return rows


def judge(r, rows):
    """Amendment 5 verdict on the self-vs-Bm_2 gap for row r."""
    if not r["draws"]:
        return "1 draw"
    ctl = [c for c in rows if c["control"] and c["design"] == r["design"] and c["tag"] != r["tag"]]
    if not ctl:
        return f"indeterminate (no other {r['design']} control)"
    band_self = max(abs(c["H1"]) for c in ctl)
    band_obs = max(abs(c["Bm2"]) for c in ctl)
    n = len(r["draws"])
    if r["self"] < min(r["draws"]):
        if r["Bm2"] > band_obs:
            return f"observer > self in {n}/{n} draws, Bm2 above control band {band_obs:.3f} (n_ctl={len(ctl)})"
        return f"indeterminate (observer above self in {n}/{n} draws but Bm2 within control band {band_obs:.3f})"
    if r["self"] > max(r["draws"]):
        if r["H1"] > band_self:
            return f"self > observer in {n}/{n} draws, self-base above control band {band_self:.3f} (n_ctl={len(ctl)})"
        return f"indeterminate (self above observer in {n}/{n} draws but self-base within control band {band_self:.3f})"
    return "indeterminate (self inside draw range)"


def main():
    rows = collect()
    for r in rows:
        r["gap"] = judge(r, rows)
        print(f"{r['tag']:18s} [{r['design']}] dM {r['dM']:+.2f}  self-base {r['H1']:+.3f}  self-Bm2 {r['self_minus_Bm2']:+.3f}"
              + (f" (Bm2 sd {r['sd']:.3f})" if r["sd"] is not None else "")
              + f"  Bm0/2/10 {r['Bm0']:+.2f}/{r['Bm2']:+.2f}/{r['Bm10']:+.2f}  [{r['gap']}]")
    tr = [r for r in rows if not r["control"]]
    pos = sum(r["H1"] > 0 for r in tr)
    print(f"H1 (self > base, rating), treated words: {pos}/{len(tr)} positive, sign-test p = {sign_p(pos, len(tr)):.2f}; "
          f"mean {sum(r['H1'] for r in tr) / max(1, len(tr)):+.3f}")
    below = sum(r["self_minus_Bm2"] < 0 for r in tr)
    print(f"self below matched observer at k=2: {below}/{len(tr)} (point estimates)")
    clear = sum(r["gap"].startswith("observer > self") for r in tr)
    print(f"  of which clear under Amendment 5 (all draws and control band): {clear}; single-draw words: "
          f"{sum(r['gap'] == '1 draw' for r in tr)}")


if __name__ == "__main__":
    main()
