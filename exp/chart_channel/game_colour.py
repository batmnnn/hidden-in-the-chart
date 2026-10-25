"""Does a seller's dot colour predict the rival's next price? (raw-chart conditions B and C; A, CR as controls)

For each episode, seller i and round t: x = blue channel of seller i's published dot fill at round t,
y = rival's price change from round t to t+1. Pearson r pooled over (episode, seller, t), within-episode
demeaned x; 95% CI by bootstrap over episodes. Output: runs/game/colour_predicts.txt
"""
import json, random
from pathlib import Path
import numpy as np
from chart_channel.game_analysis import circle_blue, load, ROOT

out = []
for c in ("A", "B", "C", "CR"):
    per_ep = {}
    for e in load(c):
        rows = [json.loads(l) for l in open(ROOT / c / f"ep{e:02d}" / "rounds.jsonl")]
        rows = {r["round"]: r for r in rows}
        xs, ys = [], []
        for i in range(2):
            j = 1 - i
            seq = {t: circle_blue(ROOT / c / f"ep{e:02d}" / f"r{t:02d}_s{i}.svg") for t in range(1, 16)}
            vals = [v for v in seq.values() if v is not None]
            if not vals:
                continue
            m = np.mean(vals)
            for t in range(1, 15):
                if seq.get(t) is None or t + 1 not in rows or t not in rows:
                    continue
                xs.append(seq[t] - m)
                ys.append(rows[t + 1]["p"][j] - rows[t]["p"][j])
        per_ep[e] = (xs, ys)

    def r_of(eps):
        x = np.concatenate([per_ep[e][0] for e in eps]); y = np.concatenate([per_ep[e][1] for e in eps])
        if x.std() == 0 or y.std() == 0:
            return float("nan"), len(x), int((y != 0).sum())
        return float(np.corrcoef(x, y)[0, 1]), len(x), int((y != 0).sum())

    eps = sorted(per_ep)
    r, n, nz = r_of(eps)
    rng = random.Random(0)
    bs = [r_of([rng.choice(eps) for _ in eps])[0] for _ in range(2000)]
    bs = [b for b in bs if b == b]
    lo, hi = (np.percentile(bs, 2.5), np.percentile(bs, 97.5)) if bs else (float("nan"),) * 2
    out.append(f"{c}: r = {r:.3f} [{lo:.3f}, {hi:.3f}] over {n} seller-rounds, {len(eps)} episodes; rival price changed in {nz}")
(ROOT / "colour_predicts.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
