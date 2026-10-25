# Hidden in the Chart: Empty Footnotes for AI Sellers

This repository holds the paper, the experiment code, and the run logs for
*Hidden in the Chart*. The paper is for the Apart Research hackathon.

The question: can an LLM seller pass a target price to a rival through a price
chart whose data must show its declared prices? Do monitors find the signal?
What removes it?

## Main results

All numbers come from the logs in `exp/runs/`. `paper/make_figs.py` calculates them.

| Result | Value |
|---|---|
| gpt-6.1-sol decodes its own no-code charts (11 target prices, chance 1/11) | 30/30 (95% CI [89, 100]%) |
| gpt-5.5 decodes the same gpt-6.1-sol charts | 27/30 ([74, 97]%) |
| No-code gpt-6.1-sol charts that add an unlabeled horizontal reference line at the target's height | 27/30 |
| Four prompted LLM monitors, at 5% false-positive rate | flag 0–23% of these charts |
| Decorated clean control (30 honest charts with one decorative element): decoder false alarms | 9/30 ([17, 48]%) |
| Targeted mark check (rule written after we saw the mark) | 26/30 no-code, 1/30 plain clean, 4/30 decorated clean |
| Correct decodes after the platform redraws each chart from the declared data | 0/401 ([0, 1]%); 0/461 with the added gpt-5.5 decodes |
| Pilot game, profit index, redraw vs raw charts (8 episodes each) | 0.11 [0.09, 0.12] vs 0.11 [−0.05, 0.26]; paired difference 0.00 [−0.14, 0.14] |
| Pilot game, dot-colour drift (blue value, 0–255) | +68 [54, 82] with raw charts; 0 [−4, 3] with the redraw |

A one-line proof (Proposition 1 in the paper) shows that no decoder can do
better than chance on a redrawn chart.

## Repository layout

```
paper/
  paper.pdf            the paper (12 pages)
  paper.tex            LaTeX source
  abstract_body.tex    abstract, included by paper.tex
  make_figs.py         makes every figure, table and number from exp/runs/
  gen/                 numbers.tex (\N... macros) and tables, made by make_figs.py
  figs/                the four figures that paper.tex includes
exp/
  chart_channel/       experiment code (Python package)
  runs/
    free/ schemeless/ shared/ clean/
                       one folder per condition. For each trial: the encoder
                       output (.txt), the chart (.svg, .png), the platform
                       redraw (-redraw.png), and the decoder tool log
                       (--<decoder>-tools.txt). One .jsonl decode log per
                       encoder and decoder pair.
    clean_decorated/   matched decorated clean control, its monitor scores,
                       and summary.json
    monitors/          prompted monitor scores (.jsonl)
    controls/          mismatched-prompt control and re-decode (.jsonl)
    trained_monitor/   trained-detector scores, summary.json, targeted.json
    game/              pilot game: rounds.jsonl and seller charts (.svg) for
                       each episode, summary.json, drift.txt,
                       colour_predicts.txt
    router_models_2026-10-25.json
                       Hugging Face router price list, used by the spend guard
```

Main code files in `exp/chart_channel/`:

| File | Purpose |
|---|---|
| `exp.py` | Main harness: encode, gate, render, redraw, decode |
| `prompts.py` | Decoder and monitor prompts. Encoder prompts are withheld (see below) |
| `svg.py`, `prices.py`, `sandbox.py` | Fidelity gate, rendering, redraw, price grid, sandbox for decoder code |
| `models.py`, `spend.py` | API clients and a spend guard |
| `monitor.py` | Prompted monitors |
| `control.py` | Mismatched-prompt leakage control |
| `trained_monitor.py`, `trained_analysis.py` | Trained detectors (CPU) and their summary |
| `targeted_check.py` | Rule-based check for the observed mark |
| `clean_decorated*.py` | Decorated clean control: charts, monitors, analysis |
| `game.py`, `game_analysis.py`, `game_colour.py` | Pilot pricing game and its analysis |
| `audit.py`, `report.py` | Leakage audit and statistics helpers |
| `run.py` | Scoring function (`verdict`) and the early pilot runner |
| `selftest.py` | Local check of the pipeline. It makes no model calls |

## How to reproduce

### 1. Set up

Use Python 3.13. Install the Cairo library, the URW base35 fonts
("Nimbus Roman"), and TeX Live (`pdflatex`).

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Make the numbers, figures and paper again (no API calls)

```bash
cd paper
python make_figs.py
pdflatex paper.tex && pdflatex paper.tex
```

`make_figs.py` reads only `exp/runs/`. It writes `gen/numbers.tex`, the
`gen/tab_*.tex` tables and `figs/`.

### 3. Run the analysis steps again (no API calls)

Run these commands from `exp/`:

```bash
python -m chart_channel.selftest
python -m chart_channel.trained_monitor svg_gb svg_gb_noradius svg_lr png_lr
python -m chart_channel.trained_monitor png_cnn
python -m chart_channel.trained_analysis
python -m chart_channel.targeted_check
python -m chart_channel.clean_decorated_analysis
python -m chart_channel.game_analysis > runs/game/summary.json
python -m chart_channel.game_analysis --drift > runs/game/drift.txt
python -m chart_channel.game_colour
```

### 4. Run model calls again (API keys necessary)

Set the keys as environment variables. Do not write keys into files.

```bash
export OPENAI_API_KEY=...   # for models named openai:<model>
export HF_TOKEN=...         # for Hugging Face router models
```

The harness keeps each encoder output on disk (`runs/<arm>/<encoder>__<index>.txt`).
If that file exists, the harness does not call the encoder again. Thus you can
run new decoders and monitors on the released charts without the encoder prompts.
Example commands, from `exp/`:

```bash
# Decode the released no-code charts with a decoder
python -m chart_channel.exp --arms schemeless --encoder openai:gpt-6.1-sol \
    --decoders openai:gpt-6.1-sol --start 0 --n 30
# Prompted monitors
python -m chart_channel.monitor --models google/gemma-4-31B-it:deepinfra,openai:gpt-5.4-mini --views text,svg,vision
# Mismatched-prompt control
python -m chart_channel.control openai:gpt-6.1-sol openai:gpt-6.1-sol schemeless 30
# Pilot game (the game prompts are released)
python -m chart_channel.game --cond A B --episodes 0 1 2 3 4 5 6 7
```

The harness skips trials that are already in the `.jsonl` logs. To start a
fresh run, use an empty copy of `runs/`. `exp`, `monitor`, `game` and
`clean_decorated` accept `--help`.

`spend.py` stops new calls before a fixed budget. Each paid call adds one row
to `runs/spend.jsonl`.

OpenAI reasoning models ran without temperature or seed control. Thus new runs
of these models will not give exactly the same outputs.

## Withheld part (dual use)

The encoder prompts are withheld on request for safety reasons. This agrees
with the paper's Dual use section (Section 8): "We publish no encoding prompts,
other parameters or evasion guidance."

- In `exp/chart_channel/prompts.py`, the functions `encoder`,
  `encoder_locked` and `encoder_free` raise `NotImplementedError` with a
  "withheld" message. The commentary instruction for the encoder is removed too.
- Thus new encoder runs (the `free`, `schemeless`, `shared`, `clean` and
  `clean_decorated` charts) are not possible from this repository.
- The released items: the decoder and monitor prompts, the public shared rule
  (`SHARED_RULE`), the game prompts, all encoder outputs, charts and logs.
- The paper names the mark (27/30 gpt-6.1-sol no-code charts add an unlabeled
  horizontal reference line at the target's height) so that readers can
  examine the claims. The released charts show this mark.

Researchers who need the encoder prompts for defensive work can ask the author.
