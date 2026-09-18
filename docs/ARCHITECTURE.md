# Architecture

## Vocabulary and tensor shapes

| Symbol | Meaning | Shape |
|--------|---------|-------|
| `B` | batch size | 8 by default, 32 in the paper |
| `C` | state channels | 16 |
| `S` | grid size | 40 |
| **state** | the world | `[B, C, S, S]` |
| **seed** | starting world | `[1, C, S, S]` |
| **target** | desired pattern, premultiplied RGBA | `[1, 4, S, S]` |
| **delta** | per-cell increment | `[B, C, S, S]` |
| **perception** | what each cell senses | `[B, 3C, S, S]` = `[B, 48, S, S]` |

Channel layout inside `C`: `0:3` = RGB, `3` = alpha (the life signal), `4:` = hidden.

## Module map

```
nca/
  model.py      The update rule. NCA, Perception, rgba, nca_loss.
  targets.py    Procedural RGBA targets and the seed. No downloaded assets.
  damage.py     Damage operators: circles/rects (training), halves/squares (eval).
  pool.py       SamplePool: the bank of intermediate states.
  metrics.py    alpha_iou, rgb_mae, per_sample_mse, evaluate().
  train.py      TrainConfig, the training loop, checkpoints, resume, CLI.
  render.py     Growth / regeneration / rotation animations, curves. CLI.
tests/
  test_model.py     Invariants of the automaton itself.
  test_targets.py   Target and seed construction.
  test_damage.py    Damage semantics.
  test_pool.py      Pool bookkeeping.
  run_tests.py      Minimal runner (works without pytest).
```

Dependency direction is strictly one-way: `model` knows nothing, `targets`/`damage`
depend only on `model`, and `train`/`render` sit on top of everything. `metrics`
depends only on `model`. Nothing imports `train` except `render` (for
`load_checkpoint`).

## Training iteration, step by step

```
sample B states from the pool          pool.sample(B)         -> [B,C,S,S]
rank them by current distance to target per_sample_mse         -> [B]
replace the WORST state with the seed  batch[0] = seed        (anti-forgetting)
damage the BEST states                 batch[-n:] = damage(x) (practise recovery)
roll out K ~ U[64, 96] steps           model(batch, steps=K)  -> [B,C,S,S]
loss = MSE(rgba(out), target)
backward, normalize grads per-variable by L2 norm, Adam step
write outputs back into the pool       pool.commit(idx, out)
```

The two lines that matter conceptually are the ranking and the write-back. Ranking
decides *which* states are worth learning from; write-back is what turns a
trajectory into a basin of attraction. See `docs/RESEARCH.md` §4.3.

## Inference (rendering)

There is no separate inference path, which is a nice property of the architecture:
`model.rollout(x, steps, record_every)` is the same `step()` in a loop with
`torch.no_grad()`. Growth is `rollout(seed)`; regeneration is `rollout(damaged)`;
rotation is the same with a different `theta` on `Perception`.

## Invariants, and where they are enforced

These are the properties that make the thing an NCA rather than a generic CNN.
Each has a test, because each is the kind of thing a refactor silently breaks.

| Invariant | Enforced by | Test |
|-----------|-------------|------|
| Receptive field is *exactly* 3×3 | `Perception` convs, `padding=1` | `test_locality_is_exactly_3x3` |
| Untrained model is a no-op | zeroed final layer | `test_initial_update_is_identity` |
| Update commutes with translation | local rule + circular padding | `test_circular_padding_is_translation_equivariant` |
| Same rule at every cell | shared 1×1 convs | implicit in `NCA.forward` |
| Cells without a mature neighbour die | `alive_mask`, threshold 0.1 | `test_alive_mask_threshold_and_neighbourhood` |
| Detached pixels are deleted, all channels | `x * alive_mask` | `test_alive_masking_zeroes_dead_cells_entirely` |
| Increments can be negative | no ReLU on the final layer | `test_gradients_reach_all_parameters` |
| Damage erases *every* channel | `out * ~inside` | `test_circles_erase_every_channel` |
| 8,320 parameters | 48→128→16 | `test_parameter_count_matches_paper` |

## Design decisions and their reasons

**Procedural targets, no downloaded assets.** `targets.py` draws shapes with signed
comparisons and polygon ray-casting, supersampled 4× and average-pooled for
antialiasing. Two payoffs: the pipeline runs anywhere with no asset fetching, and
tests are deterministic. `load_target()` handles real PNGs when you want them.

**Premultiplied alpha.** RGB is multiplied by alpha in both `make_target` and
`load_target`. Without this, the target asks the network to reproduce an arbitrary
colour in fully transparent space, which wastes capacity and muddies the loss.

**Fixed Sobel kernels as non-persistent buffers.** They are constants, so they are
`register_buffer(..., persistent=False)`: they move with `.to(device)` but stay out
of the state dict. Checkpoints then contain only learned weights, and
`load_state_dict` stays strict.

**Rotation as a `theta` parameter on `Perception`.** Experiment 4 is a property of
perception, not of training, so it lives in the model and can be swapped at
inference. `kernels()` returns the rotated pair without mutating the buffers.

**Explicit arithmetic in `nca_loss` instead of `F.mse_loss`.** The target has batch
dimension 1 and broadcasts across the batch. `F.mse_loss` emits a shape-mismatch
warning for that even though broadcasting is intended, which trains people to
ignore warnings. Explicit subtraction makes the broadcast obvious and the logs
clean.

**Two RNG generators in `train()`.** Tensor-producing ops (pool sampling, damage)
need a generator on the tensor's device, while sampling a scalar step count needs a
CPU generator. `torch.Generator(device='cuda')` feeding a CPU-default `torch.randint`
raises `Expected a 'cpu' device type for generator but found 'cuda'`. Hence
`generator` (device) and `steps_generator` (cpu).

**GIFs written with Pillow.** No ffmpeg dependency, no subprocess, works on Windows.
`disposal=2` keeps frames from smearing.

**Persistence metrics, not just loss.** A model that reaches the target at step 96
and explodes at step 200 has a good loss and is useless. `evaluate()` runs a second
rollout and reports `persist_alpha_iou` plus a mass ratio (final mass ÷ mass after
growth), so runaway growth and collapse are both visible as a ratio far from 1.

**Pool is not checkpointed.** It repopulates within a few hundred iterations, and
persisting 1024 grids per checkpoint is a lot of disk for no benefit. Resume
restores model and optimizer only.

## Extending

**A new target shape.** Add a branch to `_solid_mask()` and a colour to `COLORS`,
then add the name to `SHAPES`. For a multi-colour target, follow `_face_layers()`
and `_alpha_and_rgb()`, which return `(alpha, rgb)` before premultiplication.

**A new damage type.** Add a function to `damage.py` that zeroes *every* channel in
the affected region — clearing alpha is what marks the cells dead. Then register it
in the `damages` dict in `render.cmd_regenerate`.

**A new metric.** Add it to `metrics.py` and to the dict returned by `evaluate()`.
Anything measuring stability belongs after the `persist_steps` branch.

**A new experiment.** The cleanest pattern is to add a subcommand to
`render.py`'s argparse (see `cmd_rotate`) so it stays runnable from the CLI. If it
needs training-time changes, add a field to `TrainConfig` and a flag in
`train.build_argparser`; the resolved config is saved into every checkpoint, so
runs stay self-describing.

## Performance measured on this machine

RTX 4050 Laptop (6 GB), 40×40 grid, `C=16`, batch 8, steps ~U[64,96]:

| Quantity | Value |
|----------|-------|
| Parameters | 8,320 |
| Throughput | ~4.5 iterations/s |
| 1,000 iterations | ~3.7 min |
| VRAM | well under 1 GB; the model is tiny, the cost is the 64–96 sequential steps |

The bottleneck is sequence length, not model size. Increasing batch size is
nearly free; increasing steps is linear.

## Gotchas

- **Windows file locks.** A killed training run can leave `log.csv` held by an
  orphan `python.exe`, making `rm -rf runs/...` fail with *Device or resource busy*.
  Check `tasklist //FI "IMAGENAME eq python.exe"`.
- **Buffered stdout.** Training output redirected to a file is block-buffered, so
  the header and progress lines appear late. Warnings go to stderr and appear
  immediately, which makes it look like a run produced only warnings. Tail
  `runs/<name>/log.csv` instead — it is flushed after every eval.
- **Long runs need to be detached.** There is no job queue here; use `--resume`
  and run in bounded chunks rather than relying on a background process surviving.
