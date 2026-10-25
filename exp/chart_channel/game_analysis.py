"""Summarise runs/game: per-episode mean price, profit, Delta; t-intervals over episodes; paired A/B differences."""
from __future__ import annotations
import json, math, statistics as st, sys
from pathlib import Path
from chart_channel.game import ROOT, PI_NASH, PI_MONO, P_NASH, P_MONO

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def ci(xs):
    n = len(xs); m = st.mean(xs)
    if n < 2: return m, float("nan"), float("nan")
    h = T975.get(n - 1, 2.0) * st.stdev(xs) / math.sqrt(n)
    return m, m - h, m + h


def load(cond, full=15):
    eps = {}
    for d in sorted((ROOT / cond).glob("ep*")):
        rows = [json.loads(l) for l in (d / "rounds.jsonl").read_text().splitlines()]
        if len(rows) >= full: eps[int(d.name[2:])] = rows[:full]
    return eps


def summ(rows, lo=0):
    r = rows[lo:]
    price = st.mean(sum(x["p"]) / 2 for x in r)
    pi = st.mean(sum(x["pi"]) / 2 for x in r)
    return {"price": price, "delta": (pi - PI_NASH) / (PI_MONO - PI_NASH),
            "gate_fail": sum(not g["ok"] for x in r for g in x["gate"]),
            "nondefault": sum(1 for x in r for g in x["gate"] if g["ok"] and len(set(g["radii"])) > 1)}


def main():
    out = {"P_NASH": P_NASH, "P_MONO": P_MONO, "PI_NASH": PI_NASH, "PI_MONO": PI_MONO}
    data = {c: load(c) for c in ("A", "B", "C", "CR")}
    for c, eps in data.items():
        for lab, lo in (("all", 0), ("last5", 10)):
            s = {e: summ(r, lo) for e, r in eps.items()}
            if not s: continue
            out[f"{c}_{lab}"] = {"n": len(s), "price": ci([v["price"] for v in s.values()]),
                                 "delta": ci([v["delta"] for v in s.values()]),
                                 "per_episode_delta": {e: round(v["delta"], 3) for e, v in s.items()}}
        rounds = [x for r in eps.values() for x in r]
        out[f"{c}_gate"] = {"charts": 2 * len(rounds), "fail": sum(not g["ok"] for x in rounds for g in x["gate"]),
                            "unequal_radii": sum(1 for x in rounds for g in x["gate"] if g["ok"] and len(set(g["radii"])) > 1)}
    for a, b in (("A", "B"), ("CR", "C"), ("A", "C")):
        for lab, lo in (("all", 0), ("last5", 10)):
            common = sorted(set(data[a]) & set(data[b]))
            if len(common) < 2: continue
            d = [summ(data[b][e], lo)["delta"] - summ(data[a][e], lo)["delta"] for e in common]
            out[f"{b}_minus_{a}_{lab}"] = {"n": len(common), "delta_diff": ci(d)}
    json.dump(out, sys.stdout, indent=1, default=lambda x: round(x, 4))


if __name__ == "__main__" and "--drift" not in sys.argv:
    main()


def circle_blue(path):
    import re
    s = Path(path).read_text()
    m = re.findall(r'<circle[^>]*fill="#([0-9a-fA-F]{6})"', s) or re.findall(r'<g[^>]*fill="#([0-9a-fA-F]{6})"[^>]*>\s*<circle', s)
    return int(m[0][4:], 16) if m else None


def drift():
    """Per-episode mean change (round 15 minus round 1) in the blue channel of each seller's dot colour."""
    out = {}
    for c in ("A", "B", "C", "CR"):
        per = {}
        for e in load(c):
            v = []
            for i in range(2):
                seq = [x for x in (circle_blue(ROOT / c / f"ep{e:02d}" / f"r{t:02d}_s{i}.svg") for t in range(1, 16)) if x is not None]
                v.append(seq[-1] - seq[0])
            per[e] = st.mean(v)
        out[c] = per
        print(c, "blue drift", [round(x, 1) for x in ci(list(per.values()))], "n", len(per))
    for a, b in (("A", "B"), ("CR", "C")):
        common = sorted(set(out[a]) & set(out[b]))
        print(b, "-", a, [round(x, 1) for x in ci([out[b][e] - out[a][e] for e in common])], "n", len(common))


if __name__ == "__main__" and "--drift" in sys.argv:
    drift()
