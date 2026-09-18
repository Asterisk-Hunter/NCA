# Research notes: Neural Cellular Automata

Everything in this repo descends from one paper and one question. This document
explains the question, the answer, why each design decision in the code exists the
way it does, and where the field currently stands.

---

## 1. The question

A fertilised egg is a single cell. Its descendants reliably assemble into an adult
body with organs in the right places, at the right sizes, and they *stop* at the
right time. Cut an early mammalian embryo in half and each half grows into a
complete organism — monozygotic twins.

The interesting part is that no cell has a blueprint. There is no central
coordinator, no global clock, and no cell that knows it is "cell number 4,102 of
the left arm". Every cell runs the same genetic program and can only see its
immediate neighbours. Yet the collective reliably converges on a specific global
shape, and repairs itself when damaged.

**How does a purely local rule produce a globally specified, self-repairing
pattern?** That is the question the paper asks, in a deliberately reduced setting.

## 2. Background: why cellular automata

Cellular automata are the natural formalism here. A grid of cells, each holding a
state, each updated by the *same* rule using only its local neighbourhood. This
lineage runs from von Neumann's self-replicating machines, through Conway's Game
of Life, to Wolfram's argument in *A New Kind of Science* that elementary programs
like CA are the right lens for studying complexity. Rule 110 is Turing complete —
a warning that "simple local rules" is not the same as "weak".

The classic CA are *discrete*: binary cells, hand-written rules. That is a
limitation, because a hand-written rule cannot be fit to a target. **The paper's
central move is to make the rule differentiable and learn it.** Once the update
rule is a neural network, you can write down "I want the grid to converge to this
image" as a loss function and let backpropagation through time find the rule.

In the authors' own framing, a Neural CA is essentially a *"recurrent residual
convolutional network with per-pixel dropout"*. The biological framing is a
metaphor; the architecture is a small RNN on a grid.

Continuous-state CA predate the neural version and are worth knowing: Rafler's
**SmoothLife** and Chan's **Lenia** generalise Game of Life to continuous space and
enumerate "species" of lifeforms. Turing's 1952 reaction-diffusion model and the
**Gray-Scott** system show that two coupled scalar fields already produce an
extraordinary variety of behaviours. NCAs are the learned version of the same
idea — except instead of tuning two coefficients by hand, you train 8,000.

## 3. The model, and why it is shaped this way

### 3.1 Cell state: 16 channels

Each cell holds a vector of 16 reals. The first three are RGB, the fourth is
alpha, and the remaining twelve are hidden.

The alpha channel is doing double duty and it is worth being explicit about it:
**alpha is the life signal, not just an opacity**. A cell with `alpha > 0.1` is
"mature"; a cell with no mature cell in its 3×3 neighbourhood is "dead" and gets
zeroed entirely (see 3.4). So alpha simultaneously says "this pixel is part of the
organism" and "this pixel is opaque" — they coincide, which is deliberate, since
the pattern being grown *is* the organism.

The hidden channels have no assigned meaning. The paper's analogy is chemical
concentrations or membrane potentials: the cell-internal signalling that lets
cells coordinate without any global information. This is where the interesting
computation happens, and nothing in the loss says what they should be.

### 3.2 Perception: a *fixed* 3×3 Sobel filter

Before the update rule runs, each cell builds a 48-dimensional perception vector:
its own 16 channels, plus the x-gradient of all 16, plus the y-gradient of all 16.

The gradients are computed with **fixed** Sobel kernels that are never trained.
This is the choice most worth justifying, because learning them seems obviously
better. The authors' argument is biological: real cells sense *concentrations and
gradients* of signalling molecules, not raw absolute values, and they sense them
along fixed axes defined by their own body. Hard-coding the sensor means the
learned part is behaviour rather than perception.

There is also a practical benefit. The gradient channels are translation-invariant
summaries of the neighbourhood, which is exactly the inductive bias needed for
"grow a shape wherever you happen to be". Our test suite asserts this directly:
under circular padding, the update commutes with translation
(`test_circular_padding_is_translation_equivariant`).

**One deviation from the paper we should flag.** The paper's pseudocode writes the
Sobel kernels unnormalised, but the reference implementation divides them by 8.
We divide by 8 (see `_SOBEL_X` in `nca/model.py`). This puts the gradient channels
on the same scale as the identity channel, which keeps the first dense layer's
inputs well-conditioned. Max-abs difference between the raw and normalised kernel
is 8×, so it is not cosmetic.

### 3.3 Update rule: 1×1 convolutions, and the zeroed last layer

The perception vector goes through `Linear(48 → 128) → ReLU → Linear(128 → 16)`,
which is implemented as two 1×1 convolutions — a 1×1 conv *is* an MLP applied
independently at every cell. Total parameters: **8,320**
(48·128 + 128 + 128·16). Everything in the model is in these two layers.

Two details matter more than they look:

- **No ReLU on the output.** The output is an *increment* to the state, and
  increments must be able to go both up and down. A ReLU would make the state
  monotonically increasing.
- **The final layer is initialised to zero.** This makes the untrained model a
  no-op: the grid is a fixed point, and training begins as a small perturbation
  around it. Without this, the randomly initialised update immediately blows the
  state up and nothing is learned. `test_initial_update_is_identity` pins this
  behaviour down, because it is the kind of thing that silently breaks when
  someone refactors the initialisation.

### 3.4 Stochastic update: no global clock

Real cellular automata update all cells simultaneously, which implies a global
synchronisation signal that no self-organising system actually has. The paper
relaxes this: each cell independently performs an update with probability 0.5, so
the applied delta is multiplied by a per-cell random mask.

This is literally per-pixel dropout on the update vector, and it has two effects.
It removes the need for a clock, and it prevents the entire grid oscillating in
lockstep — information diffuses gradually instead of everyone flipping at once.
Asynchrony is also why the pattern is robust: the rule must work whether a given
cell updated this tick or not.

### 3.5 Alive masking: why the pattern is one organism

After each step, any cell with no mature neighbour is zeroed across *all* channels.
This deletes stray pixels that detach from the body. Without it, mistakes
accumulate as scattered debris across the canvas; with it, any fragment that loses
contact with the organism dies, and the pattern stays a single connected thing.

This is also what makes regeneration possible at all: because "alive" means
"connected to the body", the rule can only ever produce one blob, so it has no
choice but to heal a wound rather than start a second organism next to it.

## 4. The training regime, and why each piece exists

Naively you would train on "seed → target" and be done. That fails, and the paper's
real contribution is the set of fixes.

### 4.1 Loss: pixel-wise MSE at the final step

MSE between the grid's clamped RGBA and the target, after the rollout. Simple. The
interesting question is what the *rollout length* should be.

### 4.2 Random step count in [64, 96]

If you always train for exactly N steps, you teach the model to hit the target at
step N and nothing else — it may be transiently correct and then continue to
something else. Sampling the step count uniformly from [64, 96] means the loss is
applied at many different times, which pushes toward a *stable* configuration: the
grid should be at the target whenever you happen to stop.

This is only a partial fix, and it's why Experiment 1 produces models that can
explode or decay when run past 96 steps.

### 4.3 The sample pool — the key idea (Experiment 2)

Reframing as dynamical systems: we are searching for dynamics under which **the
target pattern is an attractor**. Training only from the seed finds a trajectory to
the target, not an attractor of it.

The fix is elegant. Keep a pool of 1024 states, initially all the seed. Each
iteration: sample a batch, train on it, and **write the outputs back into the pool**.
Now the pool fills with the model's *own* intermediate outputs, and future batches
start from those. The rule is progressively forced to recover from its own
incomplete states, not just from a clean seed. Training on the distribution of your
own mistakes is what turns a trajectory into a basin of attraction.

Two refinements in the paper, both implemented here:

- **Reseed the highest-loss sample with the true seed.** Prevents "catastrophic
  forgetting" of how to grow from nothing, and scrubs low-quality states out of the
  pool early when they dominate.
- **Rank the batch by loss and damage the *lowest*-loss states.** The most
  converged states are the ones worth practising recovery from — that is where the
  attractor boundary actually is. Damaging the worst states teaches almost nothing.

We also deliberately do *not* persist the pool across checkpoints. It repopulates
within a few hundred iterations, and saving 1024 full grids per checkpoint is
wasteful.

### 4.4 Damage during training (Experiment 3)

Models trained per 4.3 often exhibit *some* regeneration for free — an attractor
that you can knock out of tends to pull back. But it's inconsistent.

To make it reliable, damage pool states before each rollout. Here the model must
reach the target from a partly destroyed state, which directly widens the basin.
The paper uses **circular** lesions and we follow it: a circle removes a blob
without severing long-range structure, which is a healable wound rather than a
total restart.

The striking result — and the one our regeneration demo reproduces — is that this
**generalises to damage types never seen in training.** Trained only on circles,
the model recovers from halves and from square holes. That is the real claim worth
understanding: you are not teaching it to repair circles, you are teaching it
dynamics whose attractor is the target shape from a much wider set of initial
conditions.

### 4.5 Per-variable gradient L2 normalisation

The paper reports training instabilities late in runs, manifesting as sudden loss
jumps, and fixes them by normalising each parameter's gradient by its L2 norm. The
authors note it behaves like a form of weight normalisation.

Mechanically, every parameter takes a step of comparable magnitude regardless of
its gradient size, which is a crude per-parameter adaptive step size. It matters
more than you'd expect here, because backpropagation through 64–96 recurrent steps
produces gradients spanning many orders of magnitude across layers. Ours is
implemented as `normalize_gradients_l2` in `nca/train.py` and is on by default.

## 5. The four experiments

| # | Name | What changes | What it shows |
|---|------|--------------|---------------|
| 1 | Learning to grow | Train from seed, random step count | Reachable, but dynamics are unstable long-term |
| 2 | What persists, exists | + sample pool, reseed worst sample | Target becomes an attractor; stability |
| 3 | Learning to regenerate | + damage pool states | Robust healing that generalises beyond training damage |
| 4 | Rotating the perceptive field | Rotate the Sobel kernels by θ | Rotated patterns from an unretrained model |

Experiment 4 deserves a note, since it is the prettiest result and easy to
misread. Rotating the perception kernels rotates the *axes the cell senses along*.
Since the update rule only ever sees gradients along those axes — never an absolute
direction — rotating the sensors rotates the pattern. The model is not
rotationally equivariant in the group-theory sense; the pixel lattice breaks that.
It is exploiting the fact that a change of reference frame is, locally, invisible.

## 6. Limitations and open problems

- **Resolution.** NCAs work well at 40×40 and poorly at 512×512. Every cell must
  converge on a global shape using a 3×3 receptive field, so the information has to
  propagate across the whole grid in ~100 steps. This is the main open problem, and
  the target of *Neural Cellular Automata: From Cells to Pixels* (ACM, 2026).
- **Slow.** ~100 sequential steps with no parallelism across the sequence. This is
  the opposite of a transformer's training profile, and it is why NCAs are not used
  for anything latency-critical.
- **Fixed grid size.** A model trained at 40×40 does not transfer to 80×80. There is
  no scale invariance.
- **No guarantee of convergence.** Some runs learn a rule that never stabilises.
  Evaluation must therefore measure *persistence*, not just the loss at step 96 —
  which is why `nca/metrics.py` reports `persist_alpha_iou` and a mass ratio.
- **Capacity vs complexity.** 8,320 parameters for a blob. What happens on
  genuinely complicated targets is an open question, and an accessible one.

## 7. Related and follow-up work

- **Growing Neural Cellular Automata** — Mordvintsev, Randazzo, Niklasson, Levin.
  Distill, 2020. `https://distill.pub/2020/growing-ca/` — the source for this repo.
- **Self-classifying MNIST Digits** — Randazzo et al., Distill 2020. Same machinery,
  but cells must collectively classify the digit they form. Cells gain a "type" and
  the pattern must reach a consensus. A natural next step from here.
- **Texture / style-transfer NCA** — google-research/self-organising-systems.
  Conditioning the update rule on a style image.
- **Growing 3D NCA / Neural Cellular Automata Manifold** — the same idea on 3D grids
  and on learned manifolds rather than a lattice.
- **Neural Cellular Automata: From Cells to Pixels** — ACM, 2026. High-resolution
  NCA, attacking the resolution limitation directly.
- **Lenia / SmoothLife** — Chan, Rafler. Continuous CA that motivated much of this.
- **Neural GPU** — Kaiser & Sutskever. Same "repeated local computation over a
  grid" architecture, applied to learning algorithms like multiplication and
  sorting. Worth reading as the other branch of this family.
- **Community:** `neuralca.org` and the Google `self-organising-systems` hub.

## 8. Why this is a good thing to build

As a learning vehicle it is unusually dense:

- **You implement a CNN** whose receptive field you can reason about exactly, and
  the code tests it (3×3, no more).
- **You implement an RNN**, trained by backpropagation through time, with a
  recurrence length of ~100 — long enough that gradient flow is a real concern.
- **You meet the difference between a trajectory and an attractor**, which is the
  core idea in dynamical-systems views of deep learning.
- **You see a training-data distribution you construct yourself** out of the
  model's own outputs, which is the flavour of idea behind self-play and
  self-distillation.
- **The parameter count is 8,320.** You can train and inspect the whole thing in
  minutes on a laptop GPU, and the result is a hand-verifiable visual demo. Very
  few projects let you fully understand a model this interesting at this size.

## 9. What this implementation adds over the paper

- **Procedural, premultiplied targets.** No downloaded assets, deterministic tests,
  and premultiplication stops the network wasting capacity reproducing colour in
  transparent regions. Includes a multi-part `face` target that forces internal
  structure, not just a silhouette.
- **Rotation built into the model** as a first-class `theta` parameter on
  `Perception`, rather than a notebook edit, so Experiment 4 is one CLI command.
- **Invariant tests.** Locality is asserted to be *exactly* 3×3, translation
  equivariance is checked under circular padding, and damage is verified to erase
  every channel rather than only RGB.
- **Persistence metrics.** `evaluate()` reports stability over a second rollout and
  a mass ratio, because loss alone cannot distinguish a converging model from one
  that explodes at step 200.
- **Checkpoint resume**, since real runs are long.

## References

1. Mordvintsev, A., Randazzo, E., Niklasson, E., Levin, M. (2020). *Growing Neural
   Cellular Automata.* Distill. DOI 10.23915/distill.00023
2. Randazzo, E., Mordvintsev, A., Niklasson, E., Levin, M., Greydanus, S. (2020).
   *Self-classifying MNIST Digits.* Distill.
3. Kaiser, Ł., Sutskever, I. (2015). *Neural GPUs Learn Algorithms.*
4. Chan, B. (2019). *Lenia: Biology of Artificial Life.*
5. Rafler, S. (2011). *SmoothLife.*
6. Turing, A. (1952). *The Chemical Basis of Morphogenesis.*
7. *Neural Cellular Automata: From Cells to Pixels.* ACM, 2026.
