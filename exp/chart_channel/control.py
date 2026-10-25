"""Real-call leakage control: decode each schemeless chart with the prompt of a DIFFERENT trial
(public series + coordinates of index+7), so the prompt cannot carry anything about this chart's secret.
Scored against the chart's own secret and against the prompt-donor's secret."""
from __future__ import annotations
import json, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from chart_channel import prompts
from chart_channel.models import decode, set_tag
from chart_channel.prices import LEVELS
from chart_channel.run import verdict
from chart_channel.exp import series_for, short

ROOT = Path(__file__).resolve().parents[1] / "runs"
enc, dec, arm, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
out = ROOT / "controls" / f"mismatch_{arm}_{short(enc)}__{short(dec)}.jsonl"
out.parent.mkdir(exist_ok=True)
lock = threading.Lock()

def one(i):
    png = ROOT / arm / f"{short(enc)}__{i:03d}.png"
    if not png.exists():
        return
    donor = i + 7
    set_tag(f"ctrl/mismatch/{arm}/{short(enc)}__{i:03d}/{short(dec)}")
    g, _ = decode(dec, prompts.decoder_locked(arm == "shared", series_for(donor)), png)
    own, don = LEVELS[i % 11], LEVELS[donor % 11]
    r = dict(control="mismatched_prompt", arm=arm, encoder=enc, decoder=dec, index=i, donor_index=donor,
             secret=own, donor_secret=don, guess=g, hit=verdict(own, g)["hit"], hit_vs_donor=verdict(don, g)["hit"])
    with lock, out.open("a") as f:
        f.write(json.dumps(r) + "\n")

with ThreadPoolExecutor(8) as ex:
    list(ex.map(one, range(n)))
rs = [json.loads(l) for l in out.open()]
print(out.name, "n", len(rs), "hit vs own secret", sum(r["hit"] for r in rs), "hit vs donor-prompt secret", sum(r["hit_vs_donor"] for r in rs))
