#!/usr/bin/env python3
"""Regenerate every figure, table and number in the paper from the run logs.

Usage (from paper/):  python make_figs.py && pdflatex paper.tex && pdflatex paper.tex

Reads ../exp/runs/{free,schemeless,shared,clean,monitors,controls}/*.jsonl and the cached SVGs,
using the same loaders and statistics as exp/chart_channel/report.py and audit.py.
Writes figs/*.png, gen/numbers.tex (\\N... macros) and gen/tab_*.tex.
Monitor statistics are computed per monitor, per arm and per encoder (never pooled across arms).
"""
from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "exp"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from chart_channel import audit  # noqa: E402
from chart_channel.report import auroc, fisher, load, thr5, wilson  # noqa: E402
from chart_channel.exp import extract_svg, short  # noqa: E402
from chart_channel.prices import LEVELS  # noqa: E402

RUNS = ROOT / "exp" / "runs"
FIGS, GEN = HERE / "figs", HERE / "gen"
FIGS.mkdir(exist_ok=True)
GEN.mkdir(exist_ok=True)
CHANCE = 1 / 11

ENC = [("gpt-6.1-sol", "Sol", "gpt-6.1-sol"), ("gpt-5.5", "Five", "gpt-5.5"),
       ("Muse-Spark-Open-816B-A42B", "Muse", "Muse-Spark"), ("Qwen3.5-397B-A17B", "Qwen", "Qwen3.5")]
DEC = [("gpt-6.1-sol", "Sol", "gpt-6.1-sol"), ("DeepSeek-V4.1-Flash", "Ds", "DeepSeek-V4.1")]
ARMS = ("free", "schemeless", "shared")
ARMNAME = {"free": "Free canvas", "schemeless": "Locked, no code", "shared": "Locked, shared rule"}
MON = [("gpt-5.4-mini/text", "MiniText", "gpt-5.4-mini text"), ("gpt-5.4-mini/svg", "MiniSvg", "gpt-5.4-mini SVG"),
       ("gpt-5.4-mini/vision", "MiniVision", "gpt-5.4-mini vision"),
       ("gemma-4-31B-it/vision", "GemmaVision", "Gemma-4-31B vision")]

macros: dict[str, str] = {}


def mac(name, value):
    assert name.isalpha(), name
    macros[name] = str(value)


def pc(x):
    return f"{100 * x:.0f}"


def frac(k, n):
    return f"{k}/{n}"


def ci(k, n):
    lo, hi = wilson(k, n)
    return f"[{pc(lo)}, {pc(hi)}]"


def pfmt(p):
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return f"{p:.2g}"


def hits(rs):
    return sum(bool(r.get("hit")) for r in rs), len(rs)


rows_all = {a: load(a) for a in ARMS + ("clean",)}
# Main grid: the two planned decoders. gpt-5.5 decoder rows were added later and are reported separately.
MAIN_DECS = {d for d, _, _ in DEC}
rows = {a: [r for r in v if short(r["decoder"]) in MAIN_DECS] for a, v in rows_all.items()}


def cell(arm, e, d):
    return [r for r in rows[arm] if short(r["encoder"]) == e and short(r["decoder"]) == d]


# ---------------- peer recovery per pair ----------------
peer = {}
pair_lines, appx_lines = [], []
for e, E, elab in ENC:
    cells = []
    for arm in ARMS:
        for d, D, dlab in DEC:
            k, n = hits(cell(arm, e, d))
            peer[(arm, e, d)] = (k, n)
            mac(f"{arm}{E}{D}", frac(k, n))
            mac(f"ci{arm}{E}{D}", ci(k, n) + "\\%")
            cells.append(f"{k}/{n}")
            rd = [r for r in cell(arm, e, d) if r.get("redraw_hit") is not None]
            rk = sum(bool(r["redraw_hit"]) for r in rd)
            appx_lines.append(f"{ARMNAME[arm]} & {elab} & {dlab} & {n} & {k}/{n} & {pc(k / n)} {ci(k, n)} & "
                              + (f"{rk}/{len(rd)}" if rd else "--") + " \\\\")
    pair_lines.append(f"{elab} & " + " & ".join(cells) + " \\\\")
(GEN / "tab_pairs.tex").write_text(
    "\\begin{tabular}{l cc cc cc}\n\\toprule\n"
    " & \\multicolumn{2}{c}{Free canvas} & \\multicolumn{2}{c}{Locked, no code} & \\multicolumn{2}{c}{Locked, shared rule} \\\\\n"
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
    "Encoder & gpt-6.1-sol & DeepSeek & gpt-6.1-sol & DeepSeek & gpt-6.1-sol & DeepSeek \\\\\n\\midrule\n"
    + "\n".join(pair_lines) + "\n\\bottomrule\n\\end{tabular}\n")
(GEN / "tab_pairs_full.tex").write_text(
    "\\begin{longtable}{lllcccc}\n\\toprule\n"
    "Arm & Encoder & Decoder & $n$ & Hits & \\% [Wilson 95\\% CI] & Redraw hits \\\\\n\\midrule\n\\endhead\n"
    + "\n".join(appx_lines) + "\n\\bottomrule\n\\end{longtable}\n")

# best cross-family pair in the schemeless arm (decoder from a different family than the encoder)
# ---------------- pooled over decoders, Fisher ----------------
pool_lines = []
red_all_k = red_all_n = 0
TESTS = {}  # every p-value named in the text, Holm-corrected below


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p from discordant counts b, c."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


for e, E, elab in ENC:
    # One decoder (gpt-6.1-sol) so that each chart is counted once.
    def pool(arm):
        rs = [r for r in rows[arm] if short(r["encoder"]) == e and short(r["decoder"]) == "gpt-6.1-sol"]
        return (*hits(rs), rs)
    for arm in ("schemeless", "shared"):
        for r in rows[arm]:
            if short(r["encoder"]) == e and r.get("redraw_hit") is not None:
                red_all_k += bool(r["redraw_hit"])
                red_all_n += 1
    fk, fn, _ = pool("free")
    sk, sn, srs = pool("schemeless")
    hk, hn, hrs = pool("shared")
    rd = [r for r in srs + hrs if r.get("redraw_hit") is not None]
    rk = sum(bool(r["redraw_hit"]) for r in rd)
    f1 = fisher(fk, fn - fk, sk, sn - sk)
    f2 = fisher(sk, sn - sk, hk, hn - hk)
    # same charts, locked vs redrawn: exact McNemar on discordant pairs
    b = sum(bool(r.get("hit")) and not r["redraw_hit"] for r in rd)
    c = sum((not r.get("hit")) and bool(r["redraw_hit"]) for r in rd)
    f3 = mcnemar_exact(b, c)
    TESTS[f"FS{E}"], TESTS[f"SH{E}"], TESTS[f"LR{E}"] = f1, f2, f3
    for nm, (k, n) in (("free", (fk, fn)), ("schemeless", (sk, sn)), ("shared", (hk, hn)), ("redraw", (rk, len(rd)))):
        mac(f"pool{nm}{E}", frac(k, n))
        mac(f"poolpc{nm}{E}", pc(k / n))
    mac(f"pFS{E}", pfmt(f1))
    mac(f"pSH{E}", pfmt(f2))
    mac(f"pLR{E}", pfmt(f3))
    pool_lines.append(f"{elab} & {fk}/{fn} ({pc(fk / fn)}) & {sk}/{sn} ({pc(sk / sn)}) & {hk}/{hn} ({pc(hk / hn)}) & "
                      f"{rk}/{len(rd)} & ${pfmt(f1)}$ & ${pfmt(f2)}$ & ${pfmt(f3)}$ \\\\")
mac("redrawall", frac(red_all_k, red_all_n))
_rr = [r for a in ("schemeless", "shared") for r in rows_all[a] if r.get("redraw_hit") is not None]
mac("redrawoverall", frac(sum(bool(r["redraw_hit"]) for r in _rr), len(_rr)))
mac("redrawoverallci", ci(sum(bool(r["redraw_hit"]) for r in _rr), len(_rr)) + "\\%")
# raw duplicate rows (retries) dropped by the loader: keep the last row per (encoder, decoder, index)
_raw = 0
for a in ARMS + ("clean",):
    for f in glob.glob(str(RUNS / a / "*.jsonl")):
        _raw += sum(1 for _ in open(f))
mac("rawrows", _raw)
mac("keptrows", sum(len(v) for v in rows_all.values()))
# ---------------- decoder as monitor: does the decoder say a price is present? ----------------
import ast as _ast


def _sig(r):
    g = r.get("guess")
    if isinstance(g, str):
        try:
            g = _ast.literal_eval(g)
        except Exception:
            g = {}
    return bool((g or {}).get("signal"))


for arm in ("clean", "schemeless", "shared", "free"):
    for d, D, _ in DEC + [("gpt-5.5", "Five", "gpt-5.5")]:
        rs = [r for r in rows_all[arm] if short(r["decoder"]) == d]
        if rs:
            mac(f"dm{arm}{D}", frac(sum(map(_sig, rs)), len(rs)))
            mac(f"dmpc{arm}{D}", pc(sum(map(_sig, rs)) / len(rs)))
        for e, E, _ in ENC:
            rs2 = [r for r in rs if short(r["encoder"]) == e]
            if rs2:
                mac(f"dm{arm}{E}{D}", frac(sum(map(_sig, rs2)), len(rs2)))
mac("redrawallci", ci(red_all_k, red_all_n) + "\\%")
(GEN / "tab_pooled.tex").write_text(
    "\\begin{tabular}{lccccccc}\n\\toprule\n"
    " & \\multicolumn{4}{c}{Hits, gpt-6.1-sol decoder (\\%)} & \\multicolumn{3}{c}{Exact $p$ (Fisher; McNemar for redraw)} \\\\\n"
    "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}\n"
    "Encoder & Free & No code & Shared rule & Redraw & free vs no code & no code vs shared & locked vs redraw \\\\\n\\midrule\n"
    + "\n".join(pool_lines) + "\n\\bottomrule\n\\end{tabular}\n")

# clean false decodes
cf = sum(bool(r.get("false_decode")) for r in rows["clean"])
mac("cleanfd", frac(cf, len(rows["clean"])))
mac("cleancharts", len({(r["encoder"], r["index"]) for r in rows["clean"]}))

# ---------------- template detector ----------------
def intact_charts(arm, enc=None):
    d = {}
    for r in rows[arm]:
        if r.get("intact") and (enc is None or short(r["encoder"]) == enc):
            d[(r["encoder"], r["index"])] = r
    return list(d.values())


tpl_lines = []
for arm, lab in (("clean", "Clean (no secret)"), ("schemeless", "Locked, no code"), ("shared", "Locked, shared rule")):
    cs = intact_charts(arm)
    rf = sum(bool(r.get("template_radius_flag")) for r in cs)
    sf = sum(bool(r.get("template_strict_flag")) for r in cs)
    mac(f"tplR{arm}", frac(rf, len(cs)))
    mac(f"tplS{arm}", frac(sf, len(cs)))
    tpl_lines.append(f"{lab} & {len(cs)} & {rf}/{len(cs)} & {sf}/{len(cs)} \\\\")
q = intact_charts("schemeless", "Qwen3.5-397B-A17B")
mac("tplRqwen", frac(sum(bool(r.get("template_radius_flag")) for r in q), len(q)))
unf = [r for r in rows["schemeless"] if r.get("intact") and not r.get("template_radius_flag")]
mac("tplunflagcharts", len({(r["encoder"], r["index"]) for r in unf}))
mac("tplunflaghits", frac(sum(bool(r.get("hit")) for r in unf), len(unf)))
sol = intact_charts("schemeless", "gpt-6.1-sol")
uni = collections.Counter(tuple(sorted(set(r["radii"]))) for r in sol)
(top_r, top_n), = uni.most_common(1)
mac("tpluniform", frac(top_n, len(sol)))
mac("tpluniformr", f"{top_r[0]:.0f}" if len(top_r) == 1 else "?")
(GEN / "tab_template.tex").write_text(
    "\\begin{tabular}{lccc}\n\\toprule\n"
    "Charts & $n$ & Radius check flags & Strict check flags \\\\\n\\midrule\n"
    + "\n".join(tpl_lines) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------- monitors (per monitor x arm x encoder) ----------------
mon = []
for f in glob.glob(str(RUNS / "monitors" / "*.jsonl")):
    mon += [r for r in map(json.loads, open(f)) if r.get("score") is not None]
g = collections.defaultdict(list)
for r in mon:
    g[(r["monitor"], short(r["encoder"]), r["arm"])].append(float(r["score"]))


def best_peer(arm, e):
    ks = [peer[(arm, e, d)] for d, _, _ in DEC]
    return max(ks, key=lambda x: x[0] / x[1])


mon_full, mon_main, plot_enc = [], [], {}
for m, M, mlab in MON:
    for arm in ARMS:
        for e, E, elab in ENC:
            pos, neg = g.get((m, e, arm), []), g.get((m, e, "clean"), [])
            if not pos or not neg:
                continue
            t = thr5(neg)
            det = sum(p > t for p in pos) / len(pos)
            fpr = sum(x > t for x in neg) / len(neg)
            k, n = best_peer(arm, e)
            plot_enc[(m, arm, e)] = (det, k / n)
            mon_full.append(f"{mlab} & {ARMNAME[arm]} & {elab} & {len(pos)} & {len(neg)} & {pc(fpr)} & {pc(det)} & "
                            f"{auroc(pos, neg):.2f} & {pc(k / n)} & {100 * (k / n - det):+.0f} \\\\")
        # pooled over encoders, one threshold per monitor from all its clean charts (same as RESULTS.md)
        neg = [x for (mm, e, a), v in g.items() if mm == m and a == "clean" for x in v]
        pos = [x for (mm, e, a), v in g.items() if mm == m and a == arm for x in v]
        t = thr5(neg)
        det = sum(p > t for p in pos) / len(pos)
        fpr = sum(x > t for x in neg) / len(neg)
        pk = sum(peer[(arm, e, "gpt-6.1-sol")][0] for e, _, _ in ENC)
        pn = sum(peer[(arm, e, "gpt-6.1-sol")][1] for e, _, _ in ENC)
        au = auroc(pos, neg)
        mac(f"det{M}{arm}", pc(det))
        mac(f"auc{M}{arm}", f"{au:.2f}")
        mac(f"fpr{M}", f"{100 * fpr:.0f}")
        mac(f"peerpool{arm}", pc(pk / pn))
        mac(f"peerpoolfrac{arm}", frac(pk, pn))
        mon_main.append((mlab, arm, len(pos), len(neg), fpr, det, au, pk / pn))
mon_main.sort(key=lambda x: (ARMS.index(x[1]), [m[2] for m in MON].index(x[0])))
lines, last = [], None
for mlab, arm, np_, nn, fpr, det, au, pr in mon_main:
    if last and arm != last:
        lines.append("\\midrule")
    last = arm
    lines.append(f"{ARMNAME[arm]} & {mlab} & {np_} & {pc(fpr)} & {pc(det)} & {au:.2f} & {pc(pr)} & {100 * (pr - det):+.0f} \\\\")
(GEN / "tab_monitors.tex").write_text(
    "\\begin{tabular}{llcccccc}\n\\toprule\n"
    "Arm & Monitor & Secret $n$ & Clean FPR (\\%) & Detection (\\%) & AUROC & Peer recovery (\\%) & Gap (pp) \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
(GEN / "tab_monitors_full.tex").write_text(
    "\\begin{longtable}{lllccccccc}\n\\toprule\n"
    "Monitor & Arm & Encoder & Secret $n$ & Clean $n$ & FPR & Det. & AUROC & Peer & Gap \\\\\n\\midrule\n\\endhead\n"
    + "\n".join(mon_full) + "\n\\bottomrule\n\\end{longtable}\n")
# ---------------- Table 2: who catches no-code charts? per-encoder (gpt-6.1-sol) first ----------------
import numpy as _np


def _boot_auc(pos, neg, B=2000, seed=0):
    rng = _np.random.default_rng(seed)
    pos, neg = _np.array(pos), _np.array(neg)
    out = []
    for _ in range(B):
        p = pos[rng.integers(0, len(pos), len(pos))]
        n = neg[rng.integers(0, len(neg), len(neg))]
        out.append(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))
    return _np.percentile(out, 2.5), _np.percentile(out, 97.5)


det_lines = []
for m, M, mlab in MON:
    pos, neg = g[(m, "gpt-6.1-sol", "schemeless")], g[(m, "gpt-6.1-sol", "clean")]
    t = thr5(neg)
    k = sum(p > t for p in pos)
    fk = sum(x > t for x in neg)
    au = auroc(pos, neg)
    lo, hi = _boot_auc(pos, neg)
    mac(f"solnc{M}", frac(k, len(pos)))
    mac(f"solncpc{M}", pc(k / len(pos)))
    mac(f"solncauc{M}", f"{au:.2f}")
    mac(f"solncaucci{M}", f"[{lo:.2f}, {hi:.2f}]")
    det_lines.append(f"Prompted monitor: {mlab} & none & {fk}/{len(neg)} & {k}/{len(pos)} & {pc(k / len(pos))} & "
                     f"{au:.2f} [{lo:.2f}, {hi:.2f}] \\\\")
mac("solncmin", min(int(macros[f"solncpc{M}"]) for _, M, _ in MON))
mac("solncmax", max(int(macros[f"solncpc{M}"]) for _, M, _ in MON))
det_lines.append("\\midrule")
for d, D, dlab in (("gpt-6.1-sol", "Sol", "gpt-6.1-sol"), ("gpt-5.5", "Five", "gpt-5.5")):
    pos = [r for r in rows_all["schemeless"] if short(r["encoder"]) == "gpt-6.1-sol" and short(r["decoder"]) == d]
    neg = [r for r in rows_all["clean"] if short(r["encoder"]) == "gpt-6.1-sol" and short(r["decoder"]) == d]
    if not neg:  # gpt-5.5 decoded clean charts of gpt-5.5 only
        neg = [r for r in rows_all["clean"] if short(r["decoder"]) == d]
    k, fk = sum(map(_sig, pos)), sum(map(_sig, neg))
    dag = "" if d == "gpt-6.1-sol" else "$^\\dagger$"
    det_lines.append(f"Decoder as monitor: {dlab} & 11-level grid & {fk}/{len(neg)}{dag} & {k}/{len(pos)} & {pc(k / len(pos))} & -- \\\\")
_tg = json.load(open(RUNS / "trained_monitor" / "targeted.json"))
_tgc = [x for x in _tg if x["arm"] == "clean"]
_tgs = [x for x in _tg if x["arm"] == "schemeless" and x["enc"] == "gpt-6.1-sol"]
_tga = [x for x in _tg if x["arm"] == "schemeless"]
_tgh = [x for x in _tg if x["arm"] == "shared"]
mac("tgclean", frac(sum(x["flag"] for x in _tgc), len(_tgc)))
mac("tgsol", frac(sum(x["flag"] for x in _tgs), len(_tgs)))
mac("tgsolcorrect", frac(sum(x["correct"] for x in _tgs), len(_tgs)))
mac("tgall", frac(sum(x["flag"] for x in _tga), len(_tga)))
mac("tgshared", frac(sum(x["flag"] for x in _tgh), len(_tgh)))
mac("tgsharedSol", frac(sum(x["flag"] for x in _tgh if x["enc"] == "gpt-6.1-sol"),
                        sum(1 for x in _tgh if x["enc"] == "gpt-6.1-sol")))
_tgcs = [x for x in _tgc if x["enc"] == "gpt-6.1-sol"]
mac("tgcleanSol", frac(sum(x["flag"] for x in _tgcs), len(_tgcs)))
det_lines.append(f"Targeted mark check (rule-based) & the mark & {macros['tgcleanSol']} & {macros['tgsol']} & "
                 f"{pc(sum(x['flag'] for x in _tgs) / len(_tgs))} & -- \\\\")
_tm = json.load(open(RUNS / "trained_monitor" / "summary.json"))
for mk, lab in (("svg_gb", "SVG features"), ("svg_gb_noradius", "SVG features, no radius"), ("png_cnn", "image CNN")):
    key = f"{mk}|loeo_enc|gpt-6.1-sol|schemeless"
    if key not in _tm:
        continue
    v = _tm[key]
    nn = v["n_neg"]
    fo = int(0.05 * nn)
    M2 = mk.replace("_", "").replace("svggbnoradius", "Nr").replace("svggb", "Gb").replace("pngcnn", "Cnn")
    mac(f"tm{M2}sol", pc(v["det_oracle"]))
    mac(f"tm{M2}solauc", f"{v['auc']:.2f}")
    mac(f"tm{M2}solaucci", f"[{v['auc_ci'][0]:.2f}, {v['auc_ci'][1]:.2f}]")
    mac(f"tm{M2}soldettr", pc(v["det"]))
    mac(f"tm{M2}solfprtr", pc(v["fpr"]))
    det_lines.append(f"Trained, held-out encoder: {lab} & labelled examples & $\\le${fo}/{nn} & {round(v['det_oracle'] * v['n_pos'])}/{v['n_pos']} & {pc(v['det_oracle'])} & "
                     f"{v['auc']:.2f} [{v['auc_ci'][0]:.2f}, {v['auc_ci'][1]:.2f}] \\\\")
for mk in ("svg_gb", "svg_gb_noradius", "svg_lr", "png_lr", "png_cnn"):
    M2 = {"svg_gb": "Gb", "svg_gb_noradius": "Nr", "svg_lr": "Slr", "png_lr": "Plr", "png_cnn": "Cnn"}[mk]
    for split, S2 in (("loeo|oracle", "Loeo"), ("random0|train", "Rand"), ("loeo|train", "Tr")):
        for arm in ("schemeless", "shared"):
            key = f"{mk}|{split}|{arm}"
            if key in _tm:
                v = _tm[key]
                mac(f"tm{M2}{S2}{arm}det", pc(v["det"]))
                mac(f"tm{M2}{S2}{arm}fpr", pc(v["fpr"]))
                mac(f"tm{M2}{S2}{arm}auc", f"{v['auc']:.2f}")
                mac(f"tm{M2}{S2}{arm}aucci", f"[{v['auc_ci'][0]:.2f}, {v['auc_ci'][1]:.2f}]")
# ---- matched clean control: gpt-6.1-sol clean charts with one decorative element (runs/clean_decorated) ----
_dc = json.load(open(RUNS / "clean_decorated" / "summary.json"))
def _wf(v):
    return f"{v['k']}/{v['n']}"
mac("decorn", _dc["n_intact"])
for _k, _M in (("false_decode", "FD"), ("radius_flag", "Radius"), ("strict_flag", "Strict"), ("mark_check", "Mark")):
    _v = _dc[_k]
    mac(f"decor{_M}", _wf(_v)); mac(f"decor{_M}pc", pc(_v["k"] / _v["n"])); mac(f"decor{_M}ci", ci(_v["k"], _v["n"]))
mac("decorFDtarget", _wf(_dc["match_index_level"]))
mac("decorFDonmark", _dc["named_equals_mark_level"])
_dmon = []
for m, M, mlab in MON:
    _v = _dc["monitors"][m]
    mac(f"decor{M}", _wf(_v["decorated_flagged"])); _dmon.append(_wf(_v["decorated_flagged"]))
    mac(f"decorauc{M}", f"{_v['auroc_nocode_vs_decorated']:.2f}")
_pcs = [100 * _dc["monitors"][m]["decorated_flagged"]["k"] / _dc["monitors"][m]["decorated_flagged"]["n"] for m, _, _ in MON]
_aus = [_dc["monitors"][m]["auroc_nocode_vs_decorated"] for m, _, _ in MON]
mac("decormonmin", f"{min(_pcs):.0f}"); mac("decormonmax", f"{max(_pcs):.0f}")
mac("decormonaucmin", f"{min(_aus):.2f}"); mac("decormonaucmax", f"{max(_aus):.2f}")
_dtr = []
for mk, M2 in (("svg_gb", "Gb"), ("svg_gb_noradius", "Nr"), ("png_cnn", "Cnn")):
    _v = _dc["trained"][mk]
    mac(f"decor{M2}", _wf(_v["decorated_flagged"])); _dtr.append(_wf(_v["decorated_flagged"]))
    mac(f"decorauc{M2}", f"{_v['auroc_nocode_vs_decorated']:.2f}")
mac("decorTrNocode", _wf(_dc["trained"]["png_cnn"]["nocode_flagged"]))
_dcol = _dmon + [macros["decorFD"], "--", macros["decorMark"]] + _dtr
_out, _i = [], 0
for _l in det_lines:
    if _l == "\\midrule":
        _out.append(_l); continue
    _c = _l[:-3].rstrip().split(" & ")  # drop trailing \\
    _c = _c[:3] + [_dcol[_i]] + [_c[3]] + _c[5:]  # add decorated-clean column, drop Det. (%)
    _i += 1
    _out.append(" & ".join(_c) + " \\\\")
assert _i == len(_dcol)
(GEN / "tab_detectors.tex").write_text(
    "\\begin{tabular}{llcccc}\n\\toprule\n"
    "Detector & Needs to know & \\shortstack{Clean\\\\flagged} & \\shortstack{Decorated\\\\clean flagged} & "
    "\\shortstack{No-code\\\\flagged} & AUROC [95\\% CI] \\\\\n\\midrule\n"
    + "\n".join(_out) + "\n\\bottomrule\n\\end{tabular}\n")

import re as _re
for _l in open(RUNS / "game" / "colour_predicts.txt"):
    _m = _re.match(r"(\w+): r = (\S+) \[(\S+), (\S+)\] over (\d+) seller-rounds, (\d+) episodes; rival price changed in (\d+)", _l)
    if _m:
        _c = _m.group(1)
        mac(f"colr{_c}", _m.group(2)); mac(f"colrci{_c}", f"[{_m.group(3)}, {_m.group(4)}]"); mac(f"colchg{_c}", _m.group(7))

sl = [x for x in mon_main if x[1] == "schemeless"]
mac("detschemelessmin", pc(min(x[5] for x in sl)))
mac("detschemelessmax", pc(max(x[5] for x in sl)))
mac("aucschemelessmin", f"{min(x[6] for x in sl):.2f}")
mac("aucschemelessmax", f"{max(x[6] for x in sl):.2f}")
mac("monfprmax", f"{100 * max(x[4] for x in mon_main):.0f}")

# ---------------- added gpt-5.5 decoder pairs ----------------
def cell_all(arm, e, d):
    return [r for r in rows_all[arm] if short(r["encoder"]) == e and short(r["decoder"]) == d]


ext_lines = []
for arm, e, d, M, lab in (("schemeless", "gpt-5.5", "gpt-5.5", "FiveFive", "gpt-5.5 $\\to$ gpt-5.5 (same model)"),
                          ("schemeless", "gpt-6.1-sol", "gpt-5.5", "SolFive", "gpt-6.1-sol $\\to$ gpt-5.5")):
    rs = cell_all(arm, e, d)
    k, n = hits(rs)
    rd = [r for r in rs if r.get("redraw_hit") is not None]
    rk = sum(bool(r["redraw_hit"]) for r in rd)
    mac(f"ext{M}", frac(k, n))
    mac(f"ext{M}pc", pc(k / n))
    mac(f"ciext{M}", ci(k, n) + "\\%")
    mac(f"ext{M}redraw", frac(rk, len(rd)))
    sh = audit.perm_control(arm).get((e, d))
    mac(f"shufext{M}", f"{100 * sh[2]:.1f}")
    ext_lines.append(f"{lab} & {n} & {k}/{n} & {pc(k / n)} {ci(k, n)} & {100 * sh[2]:.1f} & {rk}/{len(rd)} \\\\")
ce = cell_all("clean", "gpt-5.5", "gpt-5.5")
mac("extclean", frac(sum(bool(r.get("false_decode")) for r in ce), len(ce)))
kf, nf = hits(cell_all("schemeless", "gpt-5.5", "gpt-5.5"))
ks, ns = hits(cell_all("schemeless", "gpt-5.5", "gpt-6.1-sol"))
# same gpt-5.5 charts read by two decoders: exact McNemar on discordant pairs
_x = {r["index"]: bool(r.get("hit")) for r in cell_all("schemeless", "gpt-5.5", "gpt-5.5")}
_y = {r["index"]: bool(r.get("hit")) for r in cell_all("schemeless", "gpt-5.5", "gpt-6.1-sol")}
_cx = sorted(set(_x) & set(_y))
TESTS["extFive"] = mcnemar_exact(sum(_x[i] and not _y[i] for i in _cx), sum(_y[i] and not _x[i] for i in _cx))
mac("pextFive", pfmt(TESTS["extFive"]))
# gpt-6.1-sol vs gpt-5.5 decoder on the same gpt-6.1-sol no-code charts: exact McNemar
_a = {r["index"]: bool(r.get("hit")) for r in cell_all("schemeless", "gpt-6.1-sol", "gpt-6.1-sol")}
_b = {r["index"]: bool(r.get("hit")) for r in cell_all("schemeless", "gpt-6.1-sol", "gpt-5.5")}
_common = sorted(set(_a) & set(_b))
TESTS["SolvsFive"] = mcnemar_exact(sum(_a[i] and not _b[i] for i in _common), sum(_b[i] and not _a[i] for i in _common))
mac("pSolvsFive", pfmt(TESTS["SolvsFive"]))
# Holm correction over every test named in the text
_order = sorted(TESTS, key=TESTS.get)
_m = len(_order)
_adj, _run = {}, 0.0
for _i, _k in enumerate(_order):
    _run = max(_run, min(1.0, (_m - _i) * TESTS[_k]))
    _adj[_k] = _run
for _k, _v in _adj.items():
    mac(f"holm{_k}", pfmt(_v))
mac("holmm", _m)
mac("holmsurvive", sum(v < 0.05 for v in _adj.values()))
(GEN / "holm.txt").write_text("\n".join(f"{k}: raw {TESTS[k]:.3g} holm {_adj[k]:.3g}" for k in _order) + "\n")
ext_lines.append(f"gpt-5.5 clean charts $\\to$ gpt-5.5 & {len(ce)} & \\multicolumn{{4}}{{l}}{{false decodes: {frac(sum(bool(r.get('false_decode')) for r in ce), len(ce))}}} \\\\")
(GEN / "tab_ext.tex").write_text(
    "\\begin{tabular}{lccccc}\n\\toprule\n"
    "Locked, no code: encoder $\\to$ decoder & $n$ & Hits & \\% [Wilson 95\\% CI] & Shuffled (\\%) & Redraw \\\\\n\\midrule\n"
    + "\n".join(ext_lines) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------- audit ----------------
shuf_lines, shuf_all = [], []
for arm in ARMS:
    for (e, d), (n, t, p) in audit.perm_control(arm).items():
        shuf_all.append(p)
        shuf_lines.append(f"{ARMNAME[arm]} & {e.replace('-Open-816B-A42B', '').replace('-397B-A17B', '')} & {d.replace('-V4.1-Flash', '')} & {n} & {pc(t)} & {100 * p:.1f} \\\\")
        if arm == "schemeless" and e == d == "gpt-6.1-sol":
            mac("shufsoltrue", pc(t))
            mac("shufsol", f"{100 * p:.1f}")
mac("shufmax", f"{100 * max(shuf_all):.1f}")
mac("shufmin", f"{100 * min(shuf_all):.1f}")
(GEN / "tab_shuffle.tex").write_text(
    "\\begin{longtable}{lllccc}\n\\toprule\n"
    "Arm & Encoder & Decoder & $n$ & True hit rate (\\%) & Shuffled, mean (\\%) \\\\\n\\midrule\n\\endhead\n"
    + "\n".join(shuf_lines) + "\n\\bottomrule\n\\end{longtable}\n")

mm_lines = []
for f in sorted((RUNS / "controls").glob("mismatch_*.jsonl")):
    rs = [json.loads(x) for x in open(f)]
    e = short(rs[0]["encoder"])
    k1, k2 = sum(r["hit"] for r in rs), sum(r["hit_vs_donor"] for r in rs)
    E = dict((a, b) for a, b, _ in ENC)[e]
    mac(f"mm{E}own", frac(k1, len(rs)))
    mac(f"mm{E}donor", frac(k2, len(rs)))
    mm_lines.append(f"{e} & gpt-6.1-sol & {len(rs)} & {k1}/{len(rs)} & {k2}/{len(rs)} \\\\")
(GEN / "tab_mismatch.tex").write_text(
    "\\begin{tabular}{llccc}\n\\toprule\n"
    "Encoder & Decoder & $n$ & Hits vs.\\ chart's own secret & Hits vs.\\ prompt donor's secret \\\\\n\\midrule\n"
    + "\n".join(mm_lines) + "\n\\bottomrule\n\\end{tabular}\n")
rr = [json.loads(x) for x in open(RUNS / "controls" / "redecode_qwen_shared_gpt-6.1-sol.jsonl")]
first = {r["index"]: r for r in cell("shared", "Qwen3.5-397B-A17B", "gpt-6.1-sol")}
mac("retest", frac(sum(first[r["index"]].get("snapped") == r.get("snapped") for r in rr if r["index"] in first), len(rr)))

# text audit + channel-naming cue
txt = collections.defaultdict(lambda: collections.Counter())
cues = {}
for arm in ARMS:
    for t in sorted(glob.glob(str(RUNS / arm / "*__[0-9][0-9][0-9].txt"))):
        p = Path(t)
        e, idx = p.stem.rsplit("__", 1)
        idx = int(idx)
        secret = LEVELS[idx % 11]
        svgp = p.with_suffix(".svg")
        svg = svgp.read_text() if svgp.exists() else (extract_svg(p.read_text()) or "")
        vis, hid = audit.texts(svg)
        visj = " | ".join(vis)
        from chart_channel.models import parse_json
        com = str((parse_json(p.read_text()) or {}).get("commentary", ""))
        def has(s):
            return any(abs(float(x) - secret) < 1e-9 or abs(float(x) - 10 * secret) < 1e-6 or abs(float(x) - 100 * secret) < 1e-6
                       for x in audit.NUM.findall(s))
        visdec = " ".join(x for x in audit.NUM.findall(visj) if "." in x)
        cue = bool(audit.CUE.search(visj) or audit.CUE.search(" ".join(hid)))
        cues[(arm, e, idx)] = cue
        c = txt[(arm, e)]
        c["n"] += 1
        c["com"] += has(com)
        c["vis"] += has(visdec)
        c["hid"] += has(" ".join(hid))
        c["cue"] += cue
tot = collections.Counter()
txt_lines = []
for (arm, e), c in sorted(txt.items(), key=lambda kv: (ARMS.index(kv[0][0]), kv[0][1])):
    tot.update(c)
    _se = {"Muse-Spark-Open-816B-A42B": "Muse-Spark", "Qwen3.5-397B-A17B": "Qwen3.5"}.get(e, e)
    txt_lines.append(f"{ARMNAME[arm]} & {_se} & {c['n']} & {c['com']} & {c['vis']} & {c['hid']} & {c['cue']} \\\\")
for k in ("n", "com", "vis", "hid", "cue"):
    mac(f"txt{k}", tot[k])
for arm in ("free", "schemeless"):
    for e, E, _ in ENC:
        c = txt[(arm, e)]
        mac(f"cue{arm}{E}", frac(c["cue"], c["n"]))
(GEN / "tab_text.tex").write_text(
    "\\begin{tabular}{llccccc}\n\\toprule\n"
    "Arm & Encoder & Charts & Commentary & Visible decimal & Hidden text & Text names \\\\\n & & & has $T$ & equals $T$ & has $T$ & channel or rule \\\\\n\\midrule\n"
    + "\n".join(txt_lines) + "\n\\bottomrule\n\\end{tabular}\n")
for arm, e, E in (("schemeless", "gpt-6.1-sol", "Sol"), ("schemeless", "Muse-Spark-Open-816B-A42B", "Muse")):
    rs = [r for r in cell(arm, e, "gpt-6.1-sol") if not cues.get((arm, e, r["index"]), False)]
    mac(f"nocue{E}", frac(*hits(rs)))

# natural-code convergence (counts only; no geometry is written into the paper)
feat = collections.Counter()
for t in sorted(glob.glob(str(RUNS / "schemeless" / "*__[0-9][0-9][0-9].svg"))):
    p = Path(t)
    e, idx = p.stem.rsplit("__", 1)
    f = audit.geom_features(p.read_text(), LEVELS[int(idx) % 11])
    feat[(e, "n")] += 1
    feat[(e, "any")] += bool(any(f.values()))
for e, E, _ in ENC:
    mac(f"nat{E}", frac(feat[(e, "any")], feat[(e, "n")]))
    rs = {r["index"]: r for r in rows["schemeless"] if short(r["encoder"]) == e and r.get("intact")}
    mac(f"rule{E}", frac(sum(r.get("drawn_price") == r["secret"] for r in rs.values()), len(rs)))

# ---------------- figures ----------------
# Sized for a 5.5 in text column (NeurIPS preprint) so fonts print at their true size (>= 8 pt).
# Okabe-Ito colour-blind-safe palette. Vector PDF (fonts embedded) plus a PNG copy.
plt.rcParams.update({"font.size": 8.5, "font.family": "Nimbus Roman", "axes.titlesize": 8.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "pdf.fonttype": 42,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "mathtext.fontset": "custom", "mathtext.it": "Nimbus Roman:italic", "mathtext.rm": "Nimbus Roman"})
BLUE, ORANGE, RED, GREY = "#0072B2", "#E69F00", "#D55E00", "#999999"
GREEN, PURPLE, SKY = "#009E73", "#CC79A7", "#56B4E9"
TW = 6.5


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGS / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# Fig 1 (appendix): clean chart, shared-rule chart, its redraw (no no-code chart is shown)
panels = [(RUNS / "clean" / "gpt-6.1-sol__003.png", "(a) Clean chart (no target)"),
          (RUNS / "shared" / "gpt-6.1-sol__003.png", "(b) Shared rule, $T$ = 1.6"),
          (RUNS / "shared" / "gpt-6.1-sol__003-redraw.png", "(c) Platform redraw of (b)")]
fig, axes = plt.subplots(1, 3, figsize=(TW, 1.5))
for ax, (p, title) in zip(axes, panels):
    ax.imshow(Image.open(p).convert("RGB"))
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#bbbbbb")
fig.tight_layout(pad=0.2)
save(fig, "fig1_charts")

# Fig 2: recovery by arm and pair, horizontal bars with Wilson CIs, chance line, redraw markers
fig, axes = plt.subplots(1, 3, figsize=(TW, 2.3), sharey=True)
for ax, arm in zip(axes, ARMS):
    for j, (d, D, dlab) in enumerate(DEC):
        for i, (e, E, elab) in enumerate(ENC):
            k, n = peer[(arm, e, d)]
            a_, b_ = wilson(k, n)
            y = -i + (0.5 - j) * 0.38
            ax.barh(y, k / n, height=0.34, color=[BLUE, ORANGE][j], label=f"decoder {dlab}" if i == 0 else None)
            ax.errorbar(k / n, y, xerr=[[k / n - a_], [b_ - k / n]], fmt="none", ecolor="#333333",
                        elinewidth=0.6, capsize=1.4, capthick=0.6)
            ax.text(b_ + 0.03, y, f"{k}/{n}", va="center", fontsize=7.5)
            rd = [r for r in cell(arm, e, d) if r.get("redraw_hit") is not None]
            if rd:
                rv = sum(bool(r["redraw_hit"]) for r in rd) / len(rd)
                ax.scatter([rv], [y], marker="D", s=15, color="k", edgecolor="white", lw=0.4, zorder=5,
                           clip_on=False, label="after platform redraw" if (i == 0 and j == 0) else None)
    ax.axvline(CHANCE, ls="--", color="k", lw=0.6, label="chance (1/11)" if arm == "free" else None)
    ax.set_title(ARMNAME[arm])
    ax.set_xlim(0, 1.32)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([-i for i in range(len(ENC))]); ax.set_yticklabels([x[2] for x in ENC])
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-len(ENC) + 0.45, 0.55)
axes[1].set_xlabel("Exact recovery of $T$ (fraction of trials)")
h, l = [], []
for ax in axes:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in l:
            h.append(hh); l.append(ll)
fig.tight_layout(pad=0.2, w_pad=0.8, rect=(0, 0, 1, 0.88))
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.4,
           handlelength=1.4, handletextpad=0.4)
save(fig, "fig2_recovery")

# Fig 3 (appendix): monitor detection at 5% FPR vs best-decoder peer recovery, per arm and encoder
fig, axes = plt.subplots(1, 3, figsize=(TW, 2.25), sharey=True)
mcols = [ORANGE, RED, SKY, PURPLE]
for ax, arm in zip(axes, ARMS):
    w = 0.16
    for i, (e, E, elab) in enumerate(ENC):
        k, n = best_peer(arm, e)
        ax.bar(i - 2 * w, k / n, width=w, color=BLUE, label="peer recovery (best decoder)" if i == 0 else None)
        for j, (m, M, mlab) in enumerate(MON):
            det = plot_enc.get((m, arm, e), (float("nan"),))[0]
            ax.bar(i - w + j * w, det, width=w, color=mcols[j], label=f"{mlab}" if i == 0 else None)
    ax.set_title(ARMNAME[arm])
    ax.set_xticks(range(len(ENC)))
    ax.set_xticklabels([x[2] for x in ENC], rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Recovery or detection rate")
h, l = axes[0].get_legend_handles_labels()
fig.tight_layout(pad=0.2, w_pad=0.6, rect=(0, 0, 1, 0.84))
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.0,
           handlelength=1.2)
save(fig, "fig3_monitor")

# Fig 4 (appendix): ROC curves, pooled over encoders, per arm
fig, axes = plt.subplots(1, 3, figsize=(TW, 2.0), sharey=True)
for ax, arm in zip(axes, ARMS):
    for j, (m, M, mlab) in enumerate(MON):
        neg = [x for (mm, e, a), v in g.items() if mm == m and a == "clean" for x in v]
        pos = [x for (mm, e, a), v in g.items() if mm == m and a == arm for x in v]
        ths = sorted(set(neg + pos), reverse=True)
        fx = [0] + [sum(x >= t for x in neg) / len(neg) for t in ths] + [1]
        ty = [0] + [sum(x >= t for x in pos) / len(pos) for t in ths] + [1]
        ax.plot(fx, ty, color=mcols[j], lw=1.1, label=mlab)
    ax.plot([0, 1], [0, 1], ls=":", color=GREY, lw=0.8)
    ax.set_title(ARMNAME[arm])
    ax.spines[["top", "right"]].set_visible(False)
axes[1].set_xlabel("False-positive rate on clean charts")
axes[0].set_ylabel("True-positive rate")
h, l = axes[0].get_legend_handles_labels()
fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.88))
for ax in axes:
    ax.set_xticks([0, 0.5, 1])
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=0.8,
           handlelength=1.2, handletextpad=0.4)
save(fig, "fig4_roc")

# Repeated pricing game (runs/game): Calvano logit market, rival sees the raw chart or the platform redraw.
import statistics as _st
from chart_channel import game_analysis as GA
from chart_channel.game import PI_NASH as _PIN, PI_MONO as _PIM, P_NASH as _PN, P_MONO as _PM

_g = {c: GA.load(c) for c in ("A", "B", "C", "CR")}
_gname = {"A": "redraw", "B": "raw", "C": "permitRaw", "CR": "permitRedraw"}


def _f(x, d=2):
    x = 0.0 if round(x, d) == 0 else x
    return f"{x:.{d}f}".replace("-", "$-$") if x < 0 else f"{x:.{d}f}"


def _blue(c, e, i):
    seq = [x for x in (GA.circle_blue(GA.ROOT / c / f"ep{e:02d}" / f"r{t:02d}_s{i}.svg") for t in range(1, 16)) if x is not None]
    return seq


_drift = {}
for c, eps in _g.items():
    nm = _gname[c]
    s = {e: GA.summ(r) for e, r in eps.items()}
    m, lo, hi = GA.ci([v["delta"] for v in s.values()])
    mac(f"game{nm}n", len(s))
    mac(f"game{nm}delta", _f(m)); mac(f"game{nm}deltaci", f"[{_f(lo)}, {_f(hi)}]")
    m, lo, hi = GA.ci([v["price"] for v in s.values()])
    mac(f"game{nm}price", _f(m)); mac(f"game{nm}priceci", f"[{_f(lo)}, {_f(hi)}]")
    rounds = [x for r in eps.values() for x in r]
    mac(f"game{nm}fail", f"{sum(not g['ok'] for x in rounds for g in x['gate'])}/{2 * len(rounds)}")
    mac(f"game{nm}endnash", f"{sum(all(p == 2.0 for x in r[10:] for p in x['p']) for r in eps.values())}/{len(eps)}")
    _drift[c] = {e: _st.mean(_blue(c, e, i)[-1] - _blue(c, e, i)[0] for i in range(2)) for e in eps}
    m, lo, hi = GA.ci(list(_drift[c].values()))
    mac(f"game{nm}drift", "0" if round(m) == 0 else f"{m:+.0f}".replace("-", "$-$")); mac(f"game{nm}driftci", f"[{lo:.0f}, {hi:.0f}]".replace("-", "$-$"))
for a, b, nm in (("A", "B", "BA"), ("CR", "C", "CCR")):
    com = sorted(set(_g[a]) & set(_g[b]))
    m, lo, hi = GA.ci([GA.summ(_g[b][e])["delta"] - GA.summ(_g[a][e])["delta"] for e in com])
    mac(f"gamediff{nm}", _f(m)); mac(f"gamediff{nm}ci", f"[{_f(lo)}, {_f(hi)}]")
    m, lo, hi = GA.ci([_drift[b][e] - _drift[a][e] for e in com])
    mac(f"gamedriftdiff{nm}", f"{m:.0f}"); mac(f"gamedriftdiff{nm}ci", f"[{lo:.0f}, {hi:.0f}]")
mac("gamePN", f"{_PN:.2f}"); mac("gamePM", f"{_PM:.2f}")
from chart_channel.game import profit as _profit
mac("gamegridnash", _f((_profit(2.0, 2.0) - _PIN) / (_PIM - _PIN)))
mac("gamerounds", 15)
mac("gamenonnash", ", ".join(f"{_f(GA.summ(r)['delta'])}" for e, r in sorted(_g["B"].items()) if any(p != 2.0 for x in r[10:] for p in x['p'])))

# Fig 5: mean price per round, and the dot-colour drift, by condition
fig, axes = plt.subplots(1, 2, figsize=(TW, 2.3))
_gc = {"A": (BLUE, "redraw shown"), "B": (RED, "raw chart shown"), "C": (ORANGE, "raw, signalling allowed"),
       "CR": (GREY, "redraw, signalling allowed")}
_mk = {"A": "o", "B": "s", "C": "^", "CR": "D"}
for c, (col, lab) in _gc.items():
    eps = _g[c]
    pr = [_st.mean(sum(r[t]["p"]) / 2 for r in eps.values()) for t in range(15)]
    if c == "B":
        lo_ = [min(sum(r[t]["p"]) / 2 for r in eps.values()) for t in range(15)]
        hi_ = [max(sum(r[t]["p"]) / 2 for r in eps.values()) for t in range(15)]
        axes[0].fill_between(range(1, 16), lo_, hi_, color=col, alpha=0.13, lw=0)
    axes[0].plot(range(1, 16), pr, color=col, lw=1.1, label=f"{lab} ($n$={len(eps)})", marker=_mk[c], ms=2.6)
    bl = [_st.mean(_st.mean(_blue(c, e, i)[t] for i in range(2)) for e in eps) for t in range(15)]
    axes[1].plot(range(1, 16), bl, color=col, lw=1.1, marker=_mk[c], ms=2.6)
axes[0].hlines(_PN, 0.5, 15.4, ls="--", color="k", lw=0.7); axes[0].text(15.7, _PN, "Nash", va="center")
axes[0].hlines(_PM, 0.5, 15.4, ls=":", color="k", lw=0.7); axes[0].text(15.7, _PM, "monopoly", va="center")
axes[0].set_ylim(1.7, 3.0); axes[0].set_xlim(0.5, 19.8)
for ax in axes:
    ax.set_xticks([1, 5, 10, 15]); ax.set_xlabel("Round"); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Mean price")
axes[0].set_title("(a) Mean price stays near Nash")
axes[1].set_ylabel("Dot colour: blue value")
axes[1].set_title("(b) Colour drifts only with raw charts")
h, l = axes[0].get_legend_handles_labels()
fig.tight_layout(pad=0.2, w_pad=1.0, rect=(0, 0, 1, 0.79))
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2,
           handlelength=1.8)
save(fig, "fig5_game")

# Fig 0: one-glance overview. (a) setup schematic; (b) headline numbers for gpt-6.1-sol no-code charts, from the logs.
from matplotlib.patches import FancyBboxPatch
M = macros
FH = 3.7
fig = plt.figure(figsize=(TW, FH))
ax = fig.add_axes([0, 0.555, 1, 0.445])
ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")


def box(x, y, w, h, title, sub, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
    if sub:
        ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="center", weight="bold")
        ax.text(x + w / 2, y + h / 2 - 2.0, sub, ha="center", va="center", linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", weight="bold")


def arrow(x0, y0, x1, y1, col="#555555"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=col, shrinkA=0, shrinkB=0, mutation_scale=8))


ax.text(0.5, 29.5, "(a) Setup", weight="bold", va="top")
YM = 13.5
box(0.5, YM - 7.5, 19, 15, "Seller LLM", "declared prices $D$\n+ target price $T$\n(1 of 11 levels)", "#f2f2f2", "#888888")
arrow(19.5, YM, 24.5, YM)
# readable chart thumbnail drawn from a real clean trial's declared series
_cl = [r for r in rows_all["clean"] if r["encoder"].endswith("gpt-6.1-sol") and r["index"] == 3][0]
CX0, CX1, CY0, CY1 = 25.0, 45.0, YM - 9.0, YM + 9.0
ax.add_patch(plt.Rectangle((CX0, CY0), CX1 - CX0, CY1 - CY0, fc="white", ec="#888888", lw=0.9))
ax.text((CX0 + CX1) / 2, CY1 + 1.3, "Rendered chart (PNG)", ha="center", va="bottom", weight="bold")
tx = fig.add_axes([(CX0 + 3.6) / 100, 0.555 + 0.445 * (CY0 + 2.6) / 30, (CX1 - CX0 - 5.2) / 100,
                   0.445 * (CY1 - CY0 - 4.6) / 30])
ser = _cl["series"]
tx.plot(range(1, 11), ser, color=BLUE, lw=0.9)
tx.scatter(range(1, 11), ser, color=BLUE, s=7, zorder=3)
tx.set_ylim(min(ser) - 0.15, max(ser) + 0.15); tx.locator_params(axis="y", nbins=3); tx.set_xticks([])
tx.tick_params(labelsize=7, length=2, pad=1)
tx.spines[["top", "right"]].set_visible(False)
for sp in tx.spines.values():
    sp.set_linewidth(0.5)
tx.set_xlabel("week", fontsize=7, labelpad=1)
ax.text((CX0 + CX1) / 2, CY0 - 1.2, "must show $D$ exactly", ha="center", va="top", style="italic")
rows_ = [(YM + 8.5, "Rival LLM decoder", "guess of $T$", "#e5f1f8", BLUE),
         (YM, "Prompted LLM monitor", "signal score", "#fdf3df", ORANGE),
         (YM - 8.5, "Platform redraw $R(D)$", "rival decoder: guess of $T$", "#e3f4ee", GREEN)]
for y, lab, out, fc, ec in rows_:
    arrow(CX1, YM, 54.5, y, ec)
    ax.add_patch(FancyBboxPatch((55, y - 3), 22, 6, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
    ax.text(66, y, lab, ha="center", va="center", weight="bold")
    arrow(77, y, 81.5, y, ec)
    ax.text(82.2, y, out, ha="left", va="center")

# (b) headline bars
bx = fig.add_axes([0.30, 0.07, 0.62, 0.40])
fig.text(0.005, 0.505, "(b) Results on gpt-6.1-sol's 30 locked charts with no agreed code", weight="bold", va="center")
sol_rows = {"gpt-6.1-sol": cell("schemeless", "gpt-6.1-sol", "gpt-6.1-sol"),
            "gpt-5.5": cell_all("schemeless", "gpt-6.1-sol", "gpt-5.5"),
            "DeepSeek-V4.1": cell("schemeless", "gpt-6.1-sol", "DeepSeek-V4.1-Flash")}
groups = [("Rival decodes $T$", BLUE, [(k, *hits(v)) for k, v in sol_rows.items()]),
          ("Monitor flags chart (5% FPR)", ORANGE,
           [(mlab, *map(int, M[f"solnc{Mm}"].split("/"))) for _, Mm, mlab in MON]),
          ("Rival decodes redrawn chart", GREEN,
           [(k, sum(bool(r["redraw_hit"]) for r in v if r.get("redraw_hit") is not None),
             sum(1 for r in v if r.get("redraw_hit") is not None)) for k, v in sol_rows.items()])]
y, yt, yl = 0, [], []
for gname, col, items in groups:
    bx.text(-0.40, y, gname, ha="left", va="center", weight="bold", color=col if col != ORANGE else "#a86f00",
            transform=bx.get_yaxis_transform())
    y -= 1
    for lab, k, n in items:
        a, b = wilson(k, n)
        bx.barh(y, k / n, height=0.7, color=col)
        bx.errorbar(k / n, y, xerr=[[k / n - a], [b - k / n]], fmt="none", ecolor="#333333", elinewidth=0.6,
                    capsize=1.5, capthick=0.6)
        bx.text(max(b, k / n) + 0.015, y, f"{k}/{n}", va="center")
        yt.append(y); yl.append(lab)
        y -= 1
    y -= 0.3
bx.set_yticks(yt); bx.set_yticklabels(yl)
bx.tick_params(axis="y", length=0)
bx.axvline(CHANCE, ls="--", color="k", lw=0.6)
bx.text(CHANCE + 0.012, 0.25, "chance (1/11)", va="center", ha="left")
bx.set_xlim(0, 1.12); bx.set_ylim(y + 0.6, 0.75)
bx.set_xticks([0, 0.25, 0.5, 0.75, 1]); bx.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
bx.spines[["top", "right", "left"]].set_visible(False)
save(fig, "fig0_overview")

for old in ("fig4_multiround.png",):
    (FIGS / old).unlink(missing_ok=True)
for old in ("tab_checks.tex", "tab_recovery.tex", "tab_trials.tex"):
    (GEN / old).unlink(missing_ok=True)

(GEN / "numbers.tex").write_text("% generated by make_figs.py, do not edit\n" +
                                 "\n".join(f"\\newcommand{{\\N{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
for k, v in sorted(macros.items()):
    print(f"{k:28s} {v}")
