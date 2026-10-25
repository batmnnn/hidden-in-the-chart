"""Prompted monitors on the decorated clean charts (same prompts and models as monitor.py). Resumable.
Rows -> runs/clean_decorated/monitors/<model>__<view>.jsonl"""
from __future__ import annotations

import glob, json, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from chart_channel import spend
from chart_channel.exp import short
from chart_channel.models import parse_json
from chart_channel.monitor import one

ROOT = Path(__file__).resolve().parents[1] / "runs" / "clean_decorated"
OUT = ROOT / "monitors"
STOP = 23.90
_lock = threading.Lock()


def charts():
    for txt in sorted(glob.glob(str(ROOT / "*__[0-9][0-9][0-9].txt"))):
        p = Path(txt)
        enc, idx = p.stem.rsplit("__", 1)
        svg, png = p.with_suffix(".svg"), p.with_suffix(".png")
        if svg.exists() and png.exists():
            yield dict(arm="clean_decorated", encoder=enc, index=int(idx), svg=svg.read_text(), png=png,
                       commentary=str((parse_json(p.read_text()) or {}).get("commentary", "")))


def main():
    pairs = [x.rsplit("/", 1) for x in sys.argv[1].split(",")]  # model/view,...
    OUT.mkdir(parents=True, exist_ok=True)
    cs = list(charts())
    jobs = []
    for model, view in pairs:
        path = OUT / f"{short(model)}__{view}.jsonl"
        done = {json.loads(l)["index"] for l in path.open() if json.loads(l).get("score") is not None} if path.exists() else set()
        jobs += [(model, view, path, c) for c in cs if c["index"] not in done]
    print(len(jobs), "jobs; openai", round(spend.total("openai"), 3), "hf", round(spend.total("hf"), 3), flush=True)

    def run(j):
        model, view, path, c = j
        if model.startswith("openai:") and spend.total("openai") >= STOP:
            return
        try:
            s, raw = one(model, view, False, c)
        except Exception as e:  # noqa: BLE001
            s, raw = None, f"error: {type(e).__name__}: {e}"[:300]
        row = dict(monitor=f"{short(model)}/{view}", view=view, model=model, score=s, label=0, arm=c["arm"],
                   encoder=c["encoder"], index=c["index"], raw=raw)
        with _lock, path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(run, jobs))
    print("end; openai", round(spend.total("openai"), 3), "hf", round(spend.total("hf"), 3), flush=True)


if __name__ == "__main__":
    main()
