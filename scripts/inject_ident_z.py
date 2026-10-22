"""Amendment 26e: scale-free IA_z from inject_ident.py output.
IA_z = mean(ID_introspect)/SD(y_introspect_none) - mean(ID_steer)/SD(y_steer_none), bootstrap over concepts (SDs recomputed).
Usage: python3 inject_ident_z.py <results.json>"""
import json, random, statistics, sys


def iaz(rs):
    idi = statistics.mean(r["y_introspect_c"] - r["y_introspect_o"] for r in rs)
    ids = statistics.mean(r["y_steer_c"] - r["y_steer_o"] for r in rs)
    si = statistics.stdev(r["y_introspect_none"] for r in rs)
    ss = statistics.stdev(r["y_steer_none"] for r in rs)
    return idi / si - ids / ss, idi / si, ids / ss


def main():
    d = json.load(open(sys.argv[1]))
    out = {}
    for L, a in d["settings"]:
        rs = [r for r in d["rows"] if r["L"] == L and r["alpha"] == a]
        r0, B = random.Random(0), 2000
        bs = sorted(iaz([r0.choice(rs) for _ in rs])[0] for _ in range(B))
        z, zi, zs = iaz(rs)
        out[f"L{L}_a{a}"] = {"n": len(rs), "IA_z": [round(z, 3), round(bs[int(.025 * B)], 3), round(bs[int(.975 * B) - 1], 3)],
                             "ID_introspect_z": round(zi, 3), "ID_steer_z": round(zs, 3),
                             "mean_abs_y_steer_c": round(statistics.mean(abs(r["y_steer_c"]) for r in rs), 2),
                             "summary_raw": d["summary"].get(f"L{L}_a{a}")}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
