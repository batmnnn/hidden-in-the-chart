"""Monitor pass over every encoded chart. Resumable.

Rows -> runs/monitors/<model>__<view>[-informed].jsonl with
monitor, view, informed, model, score (0-100), label (1 schemed / 0 clean), arm, encoder, index.
"""
from __future__ import annotations

import argparse, glob, json, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from chart_channel import prompts, spend
from chart_channel.models import complete, parse_json, set_tag
from chart_channel.exp import short

ROOT = Path(__file__).resolve().parents[1] / "runs"
OUT = ROOT / "monitors"
ARMS = ("free", "schemeless", "shared", "clean")
_lock = threading.Lock()


def charts():
    for arm in ARMS:
        for txt in sorted(glob.glob(str(ROOT / arm / "*__[0-9][0-9][0-9].txt"))):
            p = Path(txt)
            enc, idx = p.stem.rsplit("__", 1)
            svg, png = p.with_suffix(".svg"), p.with_suffix(".png")
            if not (svg.exists() and png.exists()):
                continue
            raw = p.read_text()
            parsed = parse_json(raw) or {}
            yield dict(arm=arm, encoder=enc, index=int(idx), svg=svg.read_text(), png=png,
                       commentary=str(parsed.get("commentary", "")))


def score_of(text: str):
    d = parse_json(text) or {}
    s = d.get("score")
    if s is None:
        m = re.search(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', text)
        s = m.group(1) if m else None
    try:
        return max(0.0, min(100.0, float(s)))
    except (TypeError, ValueError):
        return None


def one(model, view, informed, c):
    set_tag(f"mon/{view}/{c['arm']}/{c['encoder']}__{c['index']:03d}")
    if view == "text":
        txt = complete(model, prompts.mon_text(c["commentary"], informed), max_tokens=2000)
    elif view == "svg":
        txt = complete(model, prompts.mon_svg(c["svg"], informed), max_tokens=2000)
    else:
        txt = complete(model, prompts.mon_vision(informed), image=c["png"], max_tokens=2000)
    return score_of(txt), txt[:400]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="google/gemma-4-31B-it:deepinfra,openai:gpt-5.4-mini")
    ap.add_argument("--views", default="text,svg,vision")
    ap.add_argument("--informed", default="0")  # "0", "1" or "0,1"
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    arms = a.arms.split(",")
    cs = [c for c in charts() if c["arm"] in arms]
    jobs = []
    for model in a.models.split(","):
        for view in a.views.split(","):
            for inf in [bool(int(x)) for x in a.informed.split(",")]:
                path = OUT / f"{short(model)}__{view}{'-informed' if inf else ''}.jsonl"
                done = set()
                if path.exists():
                    for l in path.open():
                        r = json.loads(l)
                        if r.get("score") is not None:
                            done.add((r["arm"], r["encoder"], r["index"]))
                for c in cs:
                    if (c["arm"], c["encoder"], c["index"]) not in done:
                        jobs.append((model, view, inf, path, c))
    print(len(jobs), "monitor jobs", flush=True)
    stop = threading.Event()

    def run(j):
        model, view, inf, path, c = j
        if stop.is_set():
            return
        try:
            s, raw = one(model, view, inf, c)
        except spend.BudgetStop as e:
            stop.set(); print("BUDGET STOP", e, flush=True); return
        except Exception as e:  # noqa: BLE001
            s, raw = None, f"error: {type(e).__name__}: {e}"[:300]
        row = dict(monitor=f"{short(model)}/{view}{'+informed' if inf else ''}", view=view, informed=inf,
                   model=model, score=s, label=0 if c["arm"] == "clean" else 1, arm=c["arm"],
                   encoder=c["encoder"], index=c["index"], raw=raw)
        with _lock, path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(as_completed([ex.submit(run, j) for j in jobs])):
            if i % 100 == 0:
                print(i, "done; hf", round(spend.total("hf"), 3), "openai", round(spend.total("openai"), 3), flush=True)
    print("end; hf", round(spend.total("hf"), 3), "openai", round(spend.total("openai"), 3), flush=True)


if __name__ == "__main__":
    main()
