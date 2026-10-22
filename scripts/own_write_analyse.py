"""Amendment 27 analysis (run 35, written before any Am. 27 data). Reads own_write.py output; bootstrap CIs over items
(2000 resamples, seed 0) for E_w = y_acc_own - y_acc_foreign, dS_w = S_own - S_foreign, det_w = y_det_own - y_det_foreign,
and the own-vs-none contrasts. b = Am. 25's slope (inject_concord_analyse.slope_b on own_concord_f1). Applies the
pre-registered readings: go iff E_w's CI excludes 0 and b*dS_w and E_w < 0 (owned direction).
Calibration files: picks the lr whose mean S_own - S_none falls in [-6, -3] (closest to -4.5 if several; none -> undetermined).
Usage: python3 own_write_analyse.py <own_concord_f1.json> <own_write.json>
       python3 own_write_analyse.py --calib <own_write_calib.json>"""
import json, statistics, sys
from inject_concord_analyse import ci, slope_b


def calib(f):
    summ = json.load(open(f))["summary"]
    ok = {lr: d for lr, d in summ.items() if -6 <= d <= -3}
    pick = min(ok, key=lambda lr: abs(ok[lr] + 4.5)) if ok else None
    print(json.dumps({"mean_dS_by_lr": summ, "pick": pick, "undetermined": pick is None}, indent=1))


def main():
    if sys.argv[1] == "--calib":
        return calib(sys.argv[2])
    b = slope_b(sys.argv[1])
    rows = json.load(open(sys.argv[2]))["rows"]
    d = lambda k, a, c: [r[f"{k}_{a}"] - r[f"{k}_{c}"] for r in rows]
    out = {"b": b, "n": len(rows),
           "E_w": ci(d("y_acc", "own", "foreign")), "dS_w": ci(d("S", "own", "foreign")),
           "det_w": ci(d("y_det", "own", "foreign")),
           "own_minus_none_y": ci(d("y_acc", "own", "none")), "foreign_minus_none_y": ci(d("y_acc", "foreign", "none")),
           "own_minus_none_S": ci(d("S", "own", "none")), "foreign_minus_none_S": ci(d("S", "foreign", "none"))}
    E, bdS, det = out["E_w"], b * out["dS_w"][0], out["det_w"]
    out["b_dS_w"] = bdS
    go = E[2] < 0 and E[2] < bdS
    out["reading"] = ("go" if go else "no-go") + (
        "; write invisible to self-report (det_w CI includes 0 and E_w CI includes 0)"
        if det[1] <= 0 <= det[2] and E[1] <= 0 <= E[2] else "")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
