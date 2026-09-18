# Handoff

For any agent (or human) picking this up. Read this first, then
`docs/FINDINGS.md` for what this implementation discovered, `docs/RESEARCH.md` for
the science, and `docs/ARCHITECTURE.md` for the code.

## What this is

A complete, from-scratch implementation of **Growing Neural Cellular Automata**
(Mordvintsev et al., Distill 2020) in PyTorch. A single 8,320-parameter CNN is
applied repeatedly to every cell of a grid using only a 3×3 neighbourhood; trained
by backpropagation through ~100 recurrent steps, it learns to grow a target pattern
from one pixel and to regenerate itself when damaged.

Repo: `https://github.com/Asterisk-Hunter/NN.git` (branch `main`)

## Status

### Done, verified, and committed

- **Model** (`nca/model.py`): fixed Sobel perception, 1×1-conv update rule with a
  zeroed final layer, stochastic per-cell firing, alive masking, and a `theta`
  rotation parameter for Experiment 4.
- **Training** (`nca/train.py`): the paper's full regime — sample pool, reseed the
  highest-loss state, damage the lowest-loss states, random step count in [64, 96],
  optional gradient L2 normalisation, CSV logging, checkpoint save/resume, and
  tunable lesion size.
- **Targets** (`nca/targets.py`): 10 procedural premultiplied RGBA shapes, no
  downloaded assets, including a multi-part `face`.
- **Damage** (`nca/damage.py`), **pool** (`nca/pool.py`), **metrics**
  (`nca/metrics.py`), **rendering** (`nca/render.py`).
- **41 tests pass** (`python -m tests.run_tests`).
- **Two trained models rendered**, both with regeneration from damage types never
  seen in training. Media committed in `docs/media/`.
- **Three findings written up** in `docs/FINDINGS.md`, all reproducible.

### Verified numbers

heart, 1,500 iterations: **alpha IoU 0.972**, RGB MAE 0.0140, persistence IoU 0.998.
face, 1,800 iterations (`--damage-radius 0.14`, no `--grad-l2`): **alpha IoU 0.956**,
persistence IoU 0.968.

Regeneration, trained only on circular lesions:

| Damage | heart | face | Seen in training? |
|--------|-------|------|-------------------|
| circle | 0.999 | 0.953 | yes |
| half-left | 0.998 | 0.937 | no |
| half-top | 0.999 | 0.938 | no |
| square | 0.998 | 0.924 | no |

### The important finding

**The paper's per-variable gradient L2 normalisation causes silent catastrophic
collapse on targets with internal structure.** Two of two `face` runs collapsed to
an all-empty output (IoU 0.892 → 0.000 between evaluations); zero of four runs
collapsed without it. It is **off by default** here, with `--grad-l2` to reproduce
the paper's regime. The collapse is **invisible in the training loss** — the loss
*improved* (0.0434 → 0.0329) over the window in which the model destroyed itself.
Only `persist_mass_ratio`, falling to 0.00, revealed it. Full evidence in
`docs/FINDINGS.md`.

### Unresolved

- **The collapse mechanism is a hypothesis, not established.** The proposed
  explanation is that unit-norm gradients keep step sizes constant even when true
  gradients are tiny, so the optimiser thrashes near saturated optima. Confirming it
  means logging parameter and gradient norms per evaluation across grad-l2 on/off
  and several seeds. See `docs/FINDINGS.md` open question 1.
- **Every finding rests on one or two runs per configuration.** The direction is
  consistent and the effect size is large, but nothing has been seed-swept. Treat
  the numbers as strong evidence of a real effect, not as a characterisation.
- **No size generalisation.** A model trained at 40×40 does not transfer to 80×80.
  Untested here; it is the field's main open problem.
- **The hidden channels are unexamined.** Twelve channels with no assigned meaning.
  Nothing in this repo probes what they encode.

## Environment (as of this commit)

| | |
|---|---|
| OS | Windows, bash (Git Bash) |
| Python | 3.13.5 |
| torch | 2.8.0+cu128, CUDA available |
| GPU | NVIDIA RTX 4050 Laptop, 6 GB |
| Already installed | numpy 2.2.6, pillow 12.1.1, matplotlib 3.11.1, tqdm 4.67.1, imageio 2.37.4 |

Nothing needs installing. Throughput ~5–7 iterations/s at 40×40/batch 8, noisy on a
laptop GPU. The bottleneck is the 64–96 sequential steps, not model size.

## How to run

```bash
python -m tests.run_tests                      # 41 tests, seconds

# Simple targets: defaults are fine
python -m nca.train --shape heart --iterations 1500 --out-dir runs/heart

# Structured targets: smaller lesion, no --grad-l2 (see FINDINGS.md)
python -m nca.train --shape face --iterations 1800 --damage-radius 0.14 --out-dir runs/face

python -m nca.train --shape heart --iterations 3000 --out-dir runs/heart \
    --resume runs/heart/checkpoint.pt           # continue an interrupted run

python -m nca.render grow       --checkpoint runs/heart/checkpoint.pt
python -m nca.render regenerate --checkpoint runs/heart/checkpoint.pt
python -m nca.render rotate     --checkpoint runs/heart/checkpoint.pt
python -m nca.render plot       --run-dir runs/heart
```

## Repo map

```
nca/model.py     NCA, Perception, rgba, nca_loss       <- the automaton
nca/targets.py   procedural RGBA targets + seed
nca/damage.py    circles/rects (train), halves/squares (eval)
nca/pool.py      SamplePool
nca/metrics.py   alpha_iou, rgb_mae, per_sample_mse, evaluate
nca/train.py     TrainConfig, training loop, checkpoints, CLI
nca/render.py    GIFs and plots, CLI
tests/           41 invariant tests + a pytest-free runner
docs/FINDINGS.md      what the paper does not report, with evidence
docs/RESEARCH.md      the science and why each design choice exists
docs/ARCHITECTURE.md  code map, invariants, decision rationale, gotchas
docs/EXPERIMENTS.md   how to run each experiment + ablation table
docs/media/           committed demo GIFs and plots
handoff.md            this file
```

## Invariants — do not break these

Each has a test; if a refactor breaks one, fix the refactor, not the test.

1. **Receptive field is exactly 3×3.** No dilated or larger kernels without
   rethinking the premise.
2. **The untrained model is a no-op.** The final layer is zero-initialised.
3. **No ReLU on the output layer.** Increments must be able to be negative.
4. **Alpha is the life signal.** `alive_mask` thresholds it at 0.1 over a 3×3
   max-pool. Damage must zero *all* channels, not just RGB.
5. **The rule is shared across all cells.** No positional inputs, ever.
6. **8,320 parameters** at the default 48→128→16. If this changes,
   `docs/RESEARCH.md` and `docs/ARCHITECTURE.md` both need updating.

## Gotchas that will waste your time

- **`--grad-l2` will silently destroy a structured target.** If `alpha_iou` is good
  at one evaluation and 0.000 at the next, this is why. Check the flag before
  debugging anything else. `docs/FINDINGS.md`.
- **Training loss cannot detect that failure.** `--eval-every` small enough to see
  the collapse, and read `persist_mass_ratio`, not `loss`.
- **CUDA vs CPU generators.** Tensor-producing ops need a generator on the tensor's
  device; scalar sampling needs a CPU one. Mixing them raises *"Expected a 'cpu'
  device type for generator but found 'cuda'"*. See the two generators in `train()`.
- **`F.mse_loss` warns** because the target broadcasts from batch 1. `nca_loss` uses
  explicit arithmetic deliberately — don't "clean it up" back to `mse_loss`.
- **Windows file locks.** A killed run leaves an orphan `python.exe` holding
  `runs/*/log.csv`, so `rm -rf runs/...` fails with *Device or resource busy*. Check
  `tasklist //FI "IMAGENAME eq python.exe"`.
- **stdout is block-buffered when redirected.** A backgrounded run looks like it
  printed nothing but warnings. Tail `runs/<name>/log.csv` (flushed every eval).
- **Background processes do not survive here.** No job queue; chunk long runs with
  `--resume` in the foreground.
- **Don't commit `runs/`.** Gitignored on purpose.

## Suggested next steps, in priority order

1. **Confirm Finding 1's mechanism.** Log parameter and gradient norms per
   evaluation across `--grad-l2` on/off and 3–5 seeds, on both `heart` and `face`.
   This is the highest-value open item in the repo: it turns the strongest result
   here from an observation into an explanation.
2. **Seed-sweep the shipped results.** Everything currently rests on one or two runs
   per configuration. A small sweep would let the README quote a mean and spread.
3. **Run the remaining ablations** in `docs/EXPERIMENTS.md` — `--pool-size 1` and
   `--damage-n 0` are the two that substantiate the paper's central claims.
4. **Probe the hidden channels.** A linear probe for distance-to-target or local
   curvature would say whether the model grows a coordinate system. Most genuinely
   open question available here.
5. **The `--size` generalisation test.** Train at 24, evaluate at 48. It should
   fail; documenting *how* is the interesting part.
6. **Multi-shape conditioning.** One model that grows whichever shape is requested.
   Requires conditioning the update rule on a per-sample latent — a real
   architectural change, not a flag.

## Conventions

- **Commits:** imperative subject, then a body explaining *why* not *what*. No
  tooling attribution or generated-by footers — this repo represents the author's
  own work. Commit in logical units as work completes; do not batch everything at
  the end.
- **Code:** typed signatures, `from __future__ import annotations`, docstrings that
  explain *why* a choice was made rather than restating the code.
- **Docs:** every non-obvious decision gets its reason written down. Where this
  implementation deviates from the paper, say so explicitly and say why — see the
  Sobel /8 normalisation in `docs/RESEARCH.md` §3.2 and the gradient normalisation
  default in `docs/FINDINGS.md`.
- **Honesty over polish.** Label results from short verification runs as such. Where
  a mechanism is unproven, say it is a hypothesis. Where a number rests on one run,
  say so. `docs/FINDINGS.md` is written to that standard deliberately.
