# Experiments

How to run each of the paper's four experiments, what to look for, and the
ablations and open questions this repo makes cheap to answer.

> **Read [FINDINGS.md](FINDINGS.md) first if you are training a target with internal
> structure.** The `face` target requires `--damage-radius 0.14` and no `--grad-l2`;
> the default configuration collapses it into an all-empty output, and the training
> loss will not tell you.

## Setup

```bash
pip install -r requirements.txt
python -m tests.run_tests          # 41 tests, should pass in seconds
```

Everything below writes to `runs/<name>/` (gitignored) and takes a GPU if one is
available. Measured on an RTX 4050 Laptop: ~4.5 iterations/s at 40×40, batch 8.

## Experiment 1 + 2: grow, and stay grown

Training always includes the sample pool, so Experiments 1 and 2 are the same
command — the difference is *how long you look at the result*. To see Experiment 1's
instability, train briefly and then roll out far past the training horizon.

```bash
# Experiment 1: a short run. It will reach the target and then drift.
python -m nca.train --shape heart --iterations 300 --eval-every 100 --out-dir runs/e1

# Experiment 2: a long run. The target becomes an attractor.
python -m nca.train --shape heart --iterations 3000 --eval-every 500 --out-dir runs/e2
```

Look at `persist_alpha_iou` and `persist_mass_ratio` in the eval lines. This is the
whole point of the experiment:

- `persist_alpha_iou` — IoU after running *twice* as long as training steps. A short
  run will drop; a pooled run should hold.
- `persist_mass_ratio` — total alpha after the second rollout divided by after the
  first. **Near 1.0 is stable.** Well above 1 means runaway growth (the pattern
  spreads), well below 1 means decay. A good training loss with a mass ratio of 5 is
  a failed model, and loss alone would never have told you.

Render it:

```bash
python -m nca.render grow --checkpoint runs/e2/checkpoint.pt --steps 400 --every 8
python -m nca.render plot --run-dir runs/e2
```

`runs/e2/growth.gif` should show a single pixel expanding into the shape.
`runs/e2/curves.png` shows loss and IoU against iteration.

## Experiment 3: regenerate

The interesting result, and the one worth showing people. Train with damage, then
apply damage types the model has **never seen**.

Note the two non-default flags. With the defaults this run collapses (Finding 1).

```bash
python -m nca.train --shape face --iterations 1800 --damage-kind circle --damage-n 3 \
    --damage-radius 0.14 --out-dir runs/e3 --eval-every 600
python -m nca.render regenerate --checkpoint runs/e3/checkpoint.pt --steps 200 --recovery 300
```

Measured on the shipped model: `circle` 0.953, `half-left` 0.937, `half-top` 0.938,
`square` 0.924 — so recovery from unseen damage within 3 % of the trained type.

This writes four GIFs:

| File | Damage | Seen in training? |
|------|--------|-------------------|
| `regen_circle.gif` | circular lesion | yes |
| `regen_half-left.gif` | left half erased | **no** |
| `regen_half-top.gif` | top half erased | **no** |
| `regen_square.gif` | central square erased | **no** |

The claim to verify: recovery from the three unseen damage types is nearly as good
as from the one it trained on. That is the difference between "learned to repair
circles" and "learned dynamics whose attractor is the target shape".

Measured on a 300-iteration heart run:

| Stage | alpha IoU | Alpha mass |
|-------|-----------|------------|
| Grown from seed | 0.911 | 1372 |
| After half removed | 0.479 | 681 |
| After recovery | 0.904 | 1383 |
| *Untrained control* | *0.001* | — |

The mass going 681 → 1383 against an original of 1372 is the key number: it
genuinely regrew the missing half rather than just spreading what was left. The
untrained control at 0.001 confirms the metric is not trivially high.

## Experiment 4: rotate the perceptive field

```bash
python -m nca.render rotate --checkpoint runs/e2/checkpoint.pt --angles 0,15,30,45,60,90 --steps 400
```

Writes `rotation.png`: one panel per angle, all from a **single unretrained model**.
The pattern should rotate with θ.

Worth being precise about why this works, because it is easy to overclaim. The
update rule never sees an absolute direction — only gradients along its own sensing
axes. Rotating those axes is locally a change of reference frame, so the pattern
rotates. The pixel lattice means this is *not* true rotational equivariance, and it
degrades at large angles; a genuinely equivariant model would need group averaging.

## Ablations

Every design decision in `docs/RESEARCH.md` §3–4 is a flag, so each claim can be
tested rather than believed.

| Claim | How to test |
|-------|-------------|
| The pool is what creates persistence | Train with `--pool-size 1`, compare `persist_mass_ratio`. Expect runaway growth or decay. |
| Alive masking keeps one organism | `--no-alive-masking`, then look at the growth GIF for scattered debris. |
| Damage is what creates regeneration | Same run with `--damage-n 0`, then `render regenerate`. Expect much weaker healing. |
| **Gradient L2 norm *causes* collapse on structured targets** | `--grad-l2` on a `face` run, `--eval-every 300`. Watch `alpha_iou` drop from ~0.90 to 0.00. See `docs/FINDINGS.md` Finding 1. |
| Gradient L2 norm helps simple targets | Same flag on `heart`: IoU 0.972 with it, 0.942 without. |
| Circles are a gentler lesion than rectangles | `--damage-kind rect` and compare `regen_half-left` on the two models. |
| Lesion size interacts with target structure | `--damage-radius 0.14` vs `0.23` on `face` with `--grad-l2`. Smaller delays the collapse; it does not prevent it. |
| Asynchrony matters | `--fire-rate 1.0` (fully synchronous) vs `0.5`. |
| Capacity is not the bottleneck | `--hidden 32` — 2,560 parameters. How much quality is actually lost? |
| Grid size is a hard limit | Train at `--size 24` and evaluate at `--size 48`. Expect failure. |

The `--size` ablation is the most interesting one, because it is the field's main
open problem (`docs/RESEARCH.md` §6) and it costs minutes to check.

## Open questions this repo makes cheap to answer

Framed as things you could actually get a result on, not idle speculation. Each is
a few GPU-hours at most.

1. **What is the minimum parameter count that still regenerates?** `--hidden` down
   to 8 is 1,000 parameters. There should be a sharp capacity cliff; finding where
   it is would be a real result.
2. **Is regeneration predictability a function of the training damage budget?**
   Sweep `--damage-n` 0→6 and plot `regen_half-left` IoU against it. Does it
   saturate — is there a point past which more damage hurts growth?
3. **What do the hidden channels encode?** Twelve channels with no assigned meaning.
   A linear probe on hidden channels, predicting distance-to-target or local
   curvature, would say whether the model builds something like a coordinate system
   or a distance field. This is the most genuinely open question here.
4. **Does the alive threshold matter?** 0.1 is inherited by convention. Sweeping it
   changes how tolerant the organism is of thin structures.
5. **Is there a rotation/size trade-off?** Does training with rotated perception
   (`theta` varying per step) buy genuine robustness, or just blur the pattern?
6. **How many steps does regeneration actually need?** Measure recovery IoU as a
   function of recovery steps and damage severity — a healing curve.

Question 3 is the one worth doing, and the one closest to the paper's own framing.

## Reproducing everything from scratch

These are the exact commands behind the numbers in the README.

```bash
python -m tests.run_tests

# Experiment 1: short run, unstable long-term
python -m nca.train --shape heart --iterations 300 --out-dir runs/e1

# Experiment 2 + 3: the shipped heart model
python -m nca.train --shape heart --iterations 1500 --out-dir runs/e2 --eval-every 500

# Experiment 2 + 3 on a target with internal structure.
# Smaller lesion and no --grad-l2, both required. See docs/FINDINGS.md.
python -m nca.train --shape face --iterations 1800 --damage-radius 0.14 \
    --out-dir runs/e3 --eval-every 600

for d in e1 e2 e3; do python -m nca.render plot --run-dir runs/$d; done
python -m nca.render grow       --checkpoint runs/e2/checkpoint.pt
python -m nca.render regenerate --checkpoint runs/e2/checkpoint.pt
python -m nca.render rotate     --checkpoint runs/e2/checkpoint.pt
python -m nca.render grow       --checkpoint runs/e3/checkpoint.pt
python -m nca.render regenerate --checkpoint runs/e3/checkpoint.pt
```

Runs are seeded (`--seed`, default 0) and the resolved config is stored in every
checkpoint, so a run is self-describing. Note that stochastic firing and damage use
device RNG, so results are reproducible per device but may differ between a GPU run
and a CPU run.
