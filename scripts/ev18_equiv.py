"""Study 2 (ev18) self-vs-other label contrast with pairs as the unit (policy H32).

Paired difference in the lottery-minus-fixed effect on the self addressee, "you" vs
another label, wordings averaged within answer-word pair; t_3 95% interval; margin
delta = 25% of the other label's shared effect (a convention). Used in whose-state r2 (PR #10).
"""
import json, statistics as st, sys

T3 = 3.182
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "notes/wp3/results/ev18.json"))
p = d["p_risky"]

def eff(lab, add="self"):
    ws = [[l - f for l, f in zip(p[f"lottery|{lab}|{W}|{add}"], p[f"fixed|{lab}|{W}|{add}"])] for W in ("W1", "W2")]
    return [(a + b) / 2 for a, b in zip(*ws)]

you = eff("you")
for other in ("typ", "model"):
    o = eff(other)
    diff = [a - b for a, b in zip(you, o)]
    m, se = st.mean(diff), st.stdev(diff) / len(diff) ** 0.5
    lo, hi = m - T3 * se, m + T3 * se
    delta = 0.25 * st.mean(o)
    # one-sided: the claim is about a self *advantage*, so only the upper bound meets the margin
    slot = "SHOWN" if lo > 0 else ("SELF ADVANTAGE >= delta EXCLUDED" if hi < delta else "NOT SHOWN")
    print(f"you vs {other}: diff {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  delta {delta:.3f}  -> {slot}")
