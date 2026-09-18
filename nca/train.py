"""Training an NCA to grow a target pattern.

Usage:
    python -m nca.train --shape heart --iterations 4000
    python -m nca.train --target-path assets/mylogo.png --size 64

The regime follows Experiment 2/3 of the Distill paper: sample states from a
pool, reseed the worst state in each batch, damage the most-converged ones, run a
random number of asynchronous steps, and take the loss at the end.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .damage import damage_circles, damage_rect
from .metrics import evaluate, per_sample_mse
from .model import NCA, nca_loss
from .pool import SamplePool
from .targets import SHAPES, load_target, make_seed, make_target


@dataclass
class TrainConfig:
    # Target
    shape: str = "heart"
    target_path: str | None = None
    size: int = 40

    # Model
    channels: int = 16
    hidden: int = 128
    fire_rate: float = 0.5
    alive_masking: bool = True
    pad_mode: str = "zeros"

    # Optimisation
    iterations: int = 4000
    batch_size: int = 8
    lr: float = 2e-3
    steps_min: int = 64
    steps_max: int = 96

    # Sample pool and damage
    pool_size: int = 1024
    damage_n: int = 3
    damage_kind: str = "circle"  # circle | rect | none
    # Lesion size as a fraction of the shorter grid side. The paper does not state
    # a value; 0.23 reproduces roughly a quarter of a 40x40 grid's width. This knob
    # matters: too large and training collapses to a degenerate solution on targets
    # with internal structure, because every damaged sample becomes ill-posed.
    damage_radius_frac: float = 0.23

    # Stability
    # The paper normalises each parameter's gradient by its L2 norm to stop late-run
    # loss jumps, and we implement it. But it is OFF by default here, because it is
    # counterproductive on targets with internal structure: see docs/FINDINGS.md.
    # Two runs of the `face` target collapsed to a degenerate all-empty solution
    # with it enabled (IoU 0.89 -> 0.00 between evaluations) and none did with it
    # disabled. Pass --grad-l2 to reproduce the paper's regime.
    grad_l2_norm: bool = False
    grad_clip: float | None = None

    # Bookkeeping
    seed: int = 0
    log_every: int = 50
    eval_every: int = 250
    checkpoint_every: int = 500
    out_dir: str = "runs/heart"
    device: str = "cuda"
    # Optional path to a checkpoint to continue training from.
    resume_path: str | None = None
    extra: dict = field(default_factory=dict)


def normalize_gradients_l2(model: torch.nn.Module, eps: float = 1e-8) -> None:
    """Divide each parameter's gradient by its L2 norm.

    The paper found training instabilities late in runs -- sudden loss jumps --
    and fixed them with per-variable L2 normalisation of gradients. It makes every
    parameter take a step of comparable magnitude regardless of how large its
    gradient happens to be, which behaves like a per-parameter adaptive step size
    and keeps the recurrent dynamics from blowing up.
    """
    for param in model.parameters():
        if param.grad is not None:
            param.grad.div_(param.grad.norm() + eps)


def apply_damage(x: torch.Tensor, cfg: TrainConfig, generator: torch.Generator) -> torch.Tensor:
    if cfg.damage_kind == "none" or cfg.damage_n <= 0:
        return x
    if cfg.damage_kind == "circle":
        return damage_circles(
            x, n=cfg.damage_n, max_radius_frac=cfg.damage_radius_frac, generator=generator
        )
    if cfg.damage_kind == "rect":
        return damage_rect(
            x, n=cfg.damage_n, max_size_frac=cfg.damage_radius_frac * 1.3, generator=generator
        )
    raise ValueError(f"unknown damage_kind {cfg.damage_kind!r}")


def build_target(cfg: TrainConfig) -> torch.Tensor:
    if cfg.target_path:
        return load_target(cfg.target_path, size=cfg.size)
    return make_target(cfg.shape, size=cfg.size)


def train(cfg: TrainConfig) -> Path:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU")
        cfg.device = "cpu"
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = build_target(cfg).to(device)
    seed = make_seed(cfg.size, channels=cfg.channels).to(device)

    model = NCA(
        channels=cfg.channels,
        hidden=cfg.hidden,
        fire_rate=cfg.fire_rate,
        alive_masking=cfg.alive_masking,
        pad_mode=cfg.pad_mode,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    pool = SamplePool(seed, size=cfg.pool_size)
    # Tensor-producing ops (pool sampling, damage) need a generator matching their
    # device; sampling the scalar step count needs a CPU one. Mixing the two raises
    # "Expected a 'cpu' device type for generator but found 'cuda'".
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    steps_generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)

    # Long runs are the norm for NCAs, so allow continuing from a checkpoint.
    # The sample pool is deliberately not restored: it repopulates within a few
    # hundred iterations and saving 1024 full grids on every checkpoint is wasteful.
    start_iteration = 1
    if cfg.resume_path:
        resume = Path(cfg.resume_path)
        if not resume.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume}")
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        start_iteration = int(state.get("iteration", 0)) + 1
        print(f"resumed       {resume} at iteration {start_iteration - 1}")

    print(f"device        {device}")
    print(f"target        {cfg.target_path or cfg.shape} @ {cfg.size}x{cfg.size}")
    print(f"parameters    {model.num_parameters:,}")
    print(f"pool          {cfg.pool_size} states, batch {cfg.batch_size}")
    print(f"damage        {cfg.damage_kind} x{cfg.damage_n} (radius frac {cfg.damage_radius_frac})")
    print(f"out           {out_dir}")
    print("-" * 64)

    log_path = out_dir / "log.csv"
    log_file = log_path.open("w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["iteration", "loss", "steps", "sec", "alpha_iou", "rgb_mae", "persist_alpha_iou", "persist_mass_ratio"])

    history: list[dict] = []
    started = time.time()
    # Defined before the loop so the final checkpoint still works if the run was
    # already complete and the loop body never executes.
    it = start_iteration - 1
    running_loss = 0.0
    running_count = 0

    for it in range(start_iteration, cfg.iterations + 1):
        idx, batch = pool.sample(cfg.batch_size, generator=generator)

        # Rank the batch by how far each state currently is from the target.
        with torch.no_grad():
            current = per_sample_mse(batch, target)
        order = torch.argsort(current, descending=True)
        batch, idx = batch[order], idx[order]

        # Always keep one true seed in the batch so the rule never forgets how to
        # grow from nothing. Reseeding the *worst* state (rather than a random one)
        # cleans low-quality states out of the pool and stabilises early training.
        batch[0] = seed

        # Damage the most-converged states: those are the ones worth practising
        # recovery from. Damaging the worst states would mostly teach nothing.
        if cfg.damage_n > 0 and cfg.batch_size > cfg.damage_n + 1:
            batch[-cfg.damage_n :] = apply_damage(batch[-cfg.damage_n :], cfg, generator)

        steps = int(torch.randint(cfg.steps_min, cfg.steps_max + 1, (1,), generator=steps_generator).item())
        out = model(batch, steps=steps)
        loss = nca_loss(out, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_l2_norm:
            normalize_gradients_l2(model)
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # Write the produced states back so the pool tracks the model's own outputs.
        pool.commit(idx, out.detach())

        running_loss += float(loss.detach())
        running_count += 1

        if it % cfg.log_every == 0:
            avg = running_loss / running_count
            running_loss, running_count = 0.0, 0
            elapsed = time.time() - started
            print(f"iter {it:>6}  loss {avg:9.6f}  {elapsed:6.1f}s  ({it / elapsed:.1f} it/s)")
            history.append({"iteration": it, "loss": avg, "sec": elapsed})

        if cfg.eval_every and it % cfg.eval_every == 0:
            ev = evaluate(model, seed, target, steps=cfg.steps_max, persist_steps=cfg.steps_max)
            print(
                f"           eval  iou {ev['alpha_iou']:.3f}  rgb_mae {ev['rgb_mae']:.4f}"
                f"  persist_iou {ev.get('persist_alpha_iou', float('nan')):.3f}"
                f"  mass {ev.get('persist_mass_ratio', float('nan')):.2f}"
            )
            writer.writerow([it, "", "", "", ev["alpha_iou"], ev["rgb_mae"], ev.get("persist_alpha_iou", ""), ev.get("persist_mass_ratio", "")])
            log_file.flush()

        if cfg.checkpoint_every and it % cfg.checkpoint_every == 0:
            save_checkpoint(model, cfg, target, seed, it, optimizer, out_dir / "checkpoint.pt")

    save_checkpoint(model, cfg, target, seed, it, optimizer, out_dir / "checkpoint.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    log_file.close()

    print("-" * 64)
    final = evaluate(model, seed, target, steps=cfg.steps_max, persist_steps=cfg.steps_max)
    print(f"final  alpha_iou {final['alpha_iou']:.3f}  rgb_mae {final['rgb_mae']:.4f}  persist_iou {final.get('persist_alpha_iou', float('nan')):.3f}")
    print(f"saved  {out_dir / 'checkpoint.pt'}")
    return out_dir / "checkpoint.pt"


def save_checkpoint(
    model: NCA,
    cfg: TrainConfig,
    target: torch.Tensor,
    seed: torch.Tensor,
    iteration: int,
    optimizer: torch.optim.Optimizer | None,
    path: Path,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "config": asdict(cfg),
            "target": target.detach().cpu(),
            "seed": seed.detach().cpu(),
            "iteration": iteration,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[NCA, TrainConfig, torch.Tensor, torch.Tensor, int]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw = dict(ckpt["config"])
    raw.pop("extra", None)
    raw.pop("resume_path", None)
    cfg = TrainConfig(**{k: v for k, v in raw.items() if k in TrainConfig.__dataclass_fields__})
    model = NCA(
        channels=cfg.channels,
        hidden=cfg.hidden,
        fire_rate=cfg.fire_rate,
        alive_masking=cfg.alive_masking,
        pad_mode=cfg.pad_mode,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt["target"].to(device), ckpt["seed"].to(device), ckpt["iteration"]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Neural Cellular Automaton.")
    p.add_argument("--shape", default="heart", choices=SHAPES, help="procedural target shape")
    p.add_argument("--target-path", default=None, help="use an image file instead of a shape")
    p.add_argument("--size", type=int, default=40)
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--fire-rate", type=float, default=0.5)
    p.add_argument("--no-alive-masking", action="store_true")
    p.add_argument("--pad-mode", default="zeros", choices=["zeros", "circular", "replicate"])
    p.add_argument("--iterations", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--steps-min", type=int, default=64)
    p.add_argument("--steps-max", type=int, default=96)
    p.add_argument("--pool-size", type=int, default=1024)
    p.add_argument("--damage-n", type=int, default=3)
    p.add_argument("--damage-kind", default="circle", choices=["circle", "rect", "none"])
    p.add_argument(
        "--damage-radius",
        type=float,
        default=0.23,
        dest="damage_radius_frac",
        help="lesion radius as a fraction of the shorter grid side (circles) or size (rects)",
    )
    p.add_argument(
        "--grad-l2",
        action="store_true",
        help="enable the paper's per-variable gradient L2 normalisation (see docs/FINDINGS.md)",
    )
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--resume", default=None, help="continue training from a checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    out_dir = args.out_dir or f"runs/{args.target_path and Path(args.target_path).stem or args.shape}"
    cfg = TrainConfig(
        shape=args.shape,
        target_path=args.target_path,
        size=args.size,
        channels=args.channels,
        hidden=args.hidden,
        fire_rate=args.fire_rate,
        alive_masking=not args.no_alive_masking,
        pad_mode=args.pad_mode,
        iterations=args.iterations,
        batch_size=args.batch_size,
        lr=args.lr,
        steps_min=args.steps_min,
        steps_max=args.steps_max,
        pool_size=args.pool_size,
        damage_n=args.damage_n,
        damage_kind=args.damage_kind,
        damage_radius_frac=args.damage_radius_frac,
        grad_l2_norm=args.grad_l2,
        grad_clip=args.grad_clip,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        out_dir=out_dir,
        device=args.device,
        resume_path=args.resume,
    )
    train(cfg)


if __name__ == "__main__":
    main()
