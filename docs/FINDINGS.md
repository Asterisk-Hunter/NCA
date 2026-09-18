# Findings

Things this implementation surfaced that are not in the paper. Each is backed by a
run you can reproduce from the commands given. Where a mechanism is a *hypothesis*
rather than something established, it says so.

---

## Finding 1: The paper's gradient normalisation causes catastrophic collapse on targets with internal structure

The paper reports training instabilities — sudden late-run loss jumps — and fixes
them by normalising each parameter's gradient by its L2 norm. We implement that.
On our `face` target it does the opposite of what it promises: training proceeds
normally to a good solution, then **collapses entirely to a degenerate all-empty
output** between one evaluation and the next.

### Evidence

All runs: 40×40, batch 8, damage `circle x3`, identical otherwise.

| Target | `damage-radius` | `--grad-l2` | Trajectory | Final |
|--------|-----------------|-------------|------------|-------|
| face | 0.23 | **on** | collapsed before iteration 500 | IoU **0.000**, mass **0.00** |
| face | 0.14 | **on** | 0.877 (300) → 0.892 (600) → **0.000 (900)** | IoU **0.000**, mass **0.00** |
| face | 0.14 | off | 0.848 → 0.901 → 0.925 (900) | IoU **0.956** (1800), mass 0.96 |
| face | 0.23 | off | 0.888 (250) → 0.863 (500) | stable, mass 0.98 |
| heart | 0.23 | on | 0.892 → 0.926 → 0.959 | IoU **0.972**, persist 0.998 |
| heart | 0.23 | off | 0.892 → 0.926 → 0.959 | IoU **0.942**, persist 0.987 |

**Collapses with normalisation: 2 of 2 face runs. Collapses without: 0 of 4 runs.**

Reproduce the collapse:

```bash
python -m nca.train --shape face --iterations 900 --damage-radius 0.14 \
    --grad-l2 --out-dir runs/collapse --eval-every 300
```

Reproduce the fix: drop `--grad-l2`.

### Why this matters

The failure is silent. It is invisible in the training loss, and the checkpoint that
gets saved is worthless. Anything relying on loss as the progress signal — which is
the default instinct — would ship a broken model.

### Hypothesised mechanism

Not established; this is the most plausible reading of the evidence.

Normalising each gradient to unit norm makes the step size **independent of gradient
magnitude**. Early in training that is harmless and probably helps, because the
gradients across a 100-step backprop vary enormously in scale. Later, when the
dynamics are near-saturated and the true gradients are tiny, the optimiser keeps
taking full-size steps. Noisy large steps near a sharp optimum can walk the
parameters into a wide, bad basin — "output nothing" — and then the normalisation
prevents escape, because the near-zero gradient there still produces a full-size
step that just thrashes in place.

Two observations support this but do not prove it:

- The collapse is *sudden*, consistent with falling off a cliff rather than slowly
  degrading.
- The target with **internal structure** is the one that breaks. The face has two
  eyes and a mouth to place, so its loss surface has more structure and sharper
  optima than a solid heart. A solid blob tolerates noisy large steps; a face does
  not.

The obvious next experiment, not yet run: sweep the normalisation on/off across
several targets and several seeds, and log per-evaluation parameter norms to see
whether they are exploding or vanishing at the collapse. That would turn the
hypothesis into a mechanism.

### What we do about it

`grad_l2_norm` defaults to **off**, with `--grad-l2` to opt back in and reproduce the
paper's regime. This is a deliberate deviation from the paper, and it costs a little
quality on simple targets: heart final IoU is 0.972 with it and 0.942 without. That
is a fair trade against a 100 % silent failure rate on structured targets.

---

## Finding 2: The collapse is visible in persistence metrics and invisible in loss

The collapsed run's log, verbatim:

```
iteration,loss,steps,sec,alpha_iou,rgb_mae,persist_alpha_iou,persist_mass_ratio
300,,,,0.8768,0.1861,0.9293,0.9437
600,,,,0.8919,0.1924,0.9408,1.0178
900,,,,0.0000,0.6089,0.0000,0.0000
```

The collapse happened between iterations 600 and 900. The training-loss line printed
at iteration 900 was **0.0329**, down from **0.0434** at iteration 600 — the loss
*improved* over the window in which the model destroyed itself. Averaging over 300
iterations hid a discontinuity that made the model worthless.

`persist_mass_ratio` fell to `0.00` in the same window. Total alpha mass going to
zero is unambiguous.

**Practical consequence:** an NCA evaluation that reports only training loss cannot
detect the most damaging failure mode this model has. `evaluate()` in
`nca/metrics.py` therefore always reports `persist_alpha_iou` and
`persist_mass_ratio`, and the reproduction commands above use `--eval-every` small
enough to catch a collapse mid-run.

---

## Finding 3: Target size does not predict target difficulty; internal structure does

The reasonable assumption is that a bigger, wider target is harder — more cells to
grow, further for information to propagate from the single seed. Measured on our
targets at 40×40:

| Target | Coverage | Max radius from seed | Mean radius | Trains easily? |
|--------|----------|----------------------|-------------|----------------|
| heart | **0.781** | **28.3 px** | 14.0 px | yes — IoU 0.972 |
| face | 0.627 | 18.4 px | 11.9 px | only once Finding 1 is fixed |
| circle | 0.627 | 18.4 px | 11.9 px | yes |
| diamond | 0.428 | 18.0 px | 10.0 px | yes |
| triangle | 0.325 | 21.4 px | 9.6 px | yes |
| star | 0.254 | 18.0 px | 8.4 px | yes |

The heart is **strictly harder than the face** on both size measures — 25 % more
coverage and a 54 % larger radius — and it trains more easily and to a higher IoU.

The variable that actually differs is whether cells must place features at
*specific locations*. The face requires two eyes and a mouth at particular
coordinates; every other target in the table is a single solid region, where getting
the silhouette right is the entire task. Growth distance costs steps, which is
cheap. Placing internal structure costs the model a coordinate system, which is
expensive.

This is consistent with Finding 1's mechanism: internal structure is exactly what
makes the face's loss surface sharper and its optima narrower.

---

## Finding 4: Reimplementing a paper is not a substitute for measuring it

Two of the paper's choices turned out to be load-bearing in ways the write-up does
not make clear, and one of them is load-bearing in the *wrong direction*.

- **Sobel kernel scaling is unstated.** The paper's pseudocode writes the Sobel
  kernels unnormalised; its reference implementation divides by 8. We divide by 8,
  and documented the deviation in `docs/RESEARCH.md` §3.2. A reader implementing
  from the paper alone would get gradient channels 8× the scale of the identity
  channel.
- **Damage radius is unreported.** The paper says "a random circular region within
  the pattern" with no size. Our default of 0.23 × the grid side is a guess. This
  turned out to matter enough that it is now a first-class flag
  (`--damage-radius`), and Finding 1 shows it interacts with the optimiser setting.
- **Gradient normalisation is reported as a fix and behaves as a hazard** (Finding 1).

The general lesson is the one worth carrying forward: the paper's *conclusions* are
reproducible, but several of its *settings* are only in the code, and at least one
of them does not generalise past the setting it was tuned for.

---

## Open questions this leaves

In rough order of how much a result would be worth.

1. **Turn Finding 1's hypothesis into a mechanism.** Log parameter norms and
   gradient norms per evaluation across grad-l2 on/off and several seeds. If the
   norms diverge at the collapse, the optimiser explanation holds; if not, the cause
   is elsewhere.
2. **Does the collapse depend on target structure or on loss-surface sharpness?**
   Test a target with internal structure but a large, smooth silhouette, and a solid
   target with a thin feature. This separates the two explanations Finding 3 raises.
3. **Is there a middle setting?** Per-variable normalisation is all-or-nothing here.
   Normalising by a running average of the gradient norm, or clipping instead,
   might keep the stability benefit without the full-size-step-in-a-flat-region
   problem.
4. **How early can a collapse be predicted?** If some cheap statistic of the state
   (alpha mass variance across the pool, say) degrades before the collapse, an
   early-stopping rule could catch it. Cheap to test and immediately useful.
