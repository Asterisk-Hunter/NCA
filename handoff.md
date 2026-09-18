# Handoff

For any agent (or human) picking this up. Read this first, then
`docs/RESEARCH.md` for the science and `docs/ARCHITECTURE.md` for the code.

## What this is

A complete, from-scratch implementation of **Growing Neural Cellular Automata**
(Mordvintsev et al., Distill 2020) in PyTorch. A single 8,320-parameter CNN is
applied repeatedly to every cell of a grid using only a 3×3 neighbourhood; trained
by backpropagation through ~100 recurrent steps, it learns to grow a target pattern
from a single pixel and to regenerate itself when damaged.

Repo: `https://github.com/Asterisk-Hunter/NN.git` (branch `main`)

## Status

### Done and verified

- **Core model** (`nca/model.py`): fixed Sobel perception, 1×1-conv update rule with
  a zeroed final layer, stochastic per-cell firing, alive masking, and a `theta`
  rotation parameter for Experiment 4.
- **Training** (`nca/train.py`): the paper's full regime — sample pool, reseed the
  highest-loss state, damage the lowest-loss states, random step count in [64, 96],
  per-variable gradient L2 normalisation, CSV logging, checkpoint save/resume.
- **Targets** (`nca/targets.py`): 10 procedural shapes, premultiplied RGBA,
  antialiased, no downloaded assets. Includes a multi-part `face`.
- **Damage** (`nca/damage.py`), **pool** (`nca/pool.py`), **metrics**
  (`nca/metrics.py`), **rendering** (`nca/render.py`).
- **41 tests pass** (`python -m tests.run_tests`), covering exactly-3×3 locality,
  translation equivariance, the zeroed-update fixed point, alive-mask semantics,
  damage erasing all channels, and pool bookkeeping.
- **End-to-end training verified** on a heart target: alpha IoU 0.917 at 300
  iterations, stable (mass ratio ~1.0).
- **Regeneration verified and sanity-checked**: half-damage drops IoU to 0.479 and
  alpha mass 1372 → 681, then recovers to IoU 0.904 with mass 1383. An untrained
  control sits at IoU 0.001, so the metric is not trivially high.

### Verified numbers (heart, 300 iterations, 40×40, batch 8)

| Stage | alpha IoU | Alpha mass |
|-------|-----------|------------|
| Grown from seed | 0.911 | 1372 |
| After half removed | 0.479 | 681 |
| After recovery | 0.904 | 1383 |

### In progress / unresolved

- **The `face` target collapses to a degenerate minimum.** At iteration 500 the eval
  reported `alpha_iou 0.0`, `persist_mass_ratio 0.0`, `rgb_mae 0.609` — the model
  learned to output *nothing* rather than grow the shape. The heart does not do this.
  Leading hypothesis: the face disc covers ~63 % of the canvas (heart is much
  smaller), so the "predict empty canvas" solution is a much better local optimum
  relative to the target, and the larger radius needs more diffusion steps than 96.
  **Next step:** re-run with either a smaller face radius (`_face_layers` in
  `nca/targets.py`), a higher `--steps-max`, or a warm-up phase with `--damage-n 0`.
  Also worth trying `--batch-size 16`.
- **No long runs are committed.** `runs/` is gitignored by design. Curated demo GIFs
  belong in `docs/media/` — that directory does not exist yet.
- **The 1000/3000/4000-iteration runs in `docs/EXPERIMENTS.md` have not all been
  executed.** The commands are correct; the numbers quoted are from short
  verification runs, and are labelled as such.

## Environment (as of this commit)

| | |
|---|---|
| OS | Windows, bash (Git Bash) |
| Python | 3.13.5 |
| torch | 2.8.0+cu128, CUDA available |
| GPU | NVIDIA RTX 4050 Laptop, 6 GB |
| Already installed | numpy 2.2.6, pillow 12.1.1, matplotlib 3.11.1, tqdm 4.67.1, imageio 2.37.4 |

Nothing needs installing. Throughput: **~4.5 iterations/s** at 40×40/batch 8. The
bottleneck is the 64–96 sequential steps, not model size or batch size.

## How to run

```bash
python -m tests.run_tests                      # 41 tests, seconds

python -m nca.train --shape heart --iterations 3000 --out-dir runs/heart
python -m nca.train --shape face  --iterations 3000 --out-dir runs/face \
    --resume runs/face/checkpoint.pt           # continue an interrupted run

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
docs/RESEARCH.md      the science, and why each design choice exists
docs/ARCHITECTURE.md  code map, invariants, design decisions, gotchas
docs/EXPERIMENTS.md   how to run each experiment + ablation table
handoff.md            this file
```

## Invariants — do not break these

These are the properties that make it an NCA. Each has a test; if a refactor breaks
one, fix the refactor, not the test.

1. **Receptive field is exactly 3×3.** No dilated or larger kernels without
   rethinking the whole premise.
2. **The untrained model is a no-op.** The final layer is zero-initialised. Random
   init here destroys training.
3. **No ReLU on the output layer.** Increments must be able to be negative.
4. **Alpha is the life signal.** `alive_mask` thresholds it at 0.1 over a 3×3
   max-pool. Damage must zero *all* channels, not just RGB — clearing alpha is what
   marks cells dead.
5. **The rule is shared across all cells.** No positional inputs, ever.
6. **8,320 parameters** at the default 48→128→16. If this changes, `docs/RESEARCH.md`
   and `ARCHITECTURE.md` both need updating.

## Gotchas that will waste your time

- **CUDA vs CPU generators.** Tensor-producing ops need a generator on the tensor's
  device; scalar sampling needs a CPU one. Mixing them raises *"Expected a 'cpu'
  device type for generator but found 'cuda'"*. See the two generators in `train()`.
- **`F.mse_loss` warns** because the target broadcasts from batch 1. `nca_loss` uses
  explicit arithmetic deliberately — don't "clean it up" back to `mse_loss`.
- **Windows file locks.** A killed run leaves an orphan `python.exe` holding
  `runs/*/log.csv`, so `rm -rf runs/...` fails with *Device or resource busy*. Check
  `tasklist //FI "IMAGENAME eq python.exe"` and kill it.
- **stdout is block-buffered when redirected.** A backgrounded run looks like it
  printed nothing but warnings. Tail `runs/<name>/log.csv` (flushed every eval)
  instead of the log.
- **Background processes do not survive here.** There is no job queue; long runs
  must be chunked with `--resume` in the foreground.
- **Don't commit `runs/`.** Gitignored on purpose; checkpoints and pool states are
  large.

## Suggested next steps, in priority order

1. **Fix the `face` collapse** (see above). Until then, `heart` is the reliable
   showcase target.
2. **Produce and commit the demo media** into `docs/media/`: `growth.gif`,
   `regen_half-left.gif`, `rotation.png`, `curves.png`. Update the README results
   table with the real numbers.
3. **Run the ablations** in `docs/EXPERIMENTS.md`. The `--pool-size 1` and
   `--damage-n 0` ablations are the two that substantiate the paper's central
   claims, and each is under 15 minutes.
4. **Linear probes on the hidden channels** — this is the most genuinely open
   question in the repo (`docs/EXPERIMENTS.md`, question 3). Twelve channels with no
   assigned meaning; probing for distance-to-target would say whether the model
   grows a coordinate system.
5. **The `--size` generalisation test.** Train at 24, evaluate at 48. It should
   fail; documenting *how* it fails is the interesting part, and it is the field's
   main open problem.
6. **Multi-shape conditioning** if there's appetite for more: one model that grows
   whichever shape is requested. Would need the update rule conditioned on a
   per-sample latent, which is a real architectural change.

## Conventions

- **Commits:** imperative subject, then a body explaining *why* not *what*. No
  tooling attribution or generated-by footers -- this repo represents the author's
  own work. Commit in logical units as work completes; do not batch everything at
  the end.
- **Code:** typed signatures, `from __future__ import annotations`, docstrings that
  explain *why* a choice was made rather than restating the code. Comments should
  carry the reasoning that isn't recoverable from reading the line.
- **Docs:** every non-obvious decision gets its reason written down. Where this
  implementation deviates from the paper, say so explicitly and say why — see the
  Sobel /8 normalisation note in `docs/RESEARCH.md` §3.2.
- **Honesty over polish.** Where a result is from a short verification run, label it
  as such. Where something doesn't work (the `face` collapse), document it in this
  file rather than quietly dropping the target.
