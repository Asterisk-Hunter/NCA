"""Render trained NCAs: growth, regeneration, and rotated growth.

Produces GIFs (written with Pillow, so no ffmpeg needed) plus a keyframe grid PNG.
RGBA states are composited over white, matching the paper's figures.

Usage:
    python -m nca.render grow       --checkpoint runs/heart/checkpoint.pt
    python -m nca.render regenerate --checkpoint runs/heart/checkpoint.pt
    python -m nca.render rotate     --checkpoint runs/heart/checkpoint.pt
    python -m nca.render plot       --run-dir runs/heart
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .damage import damage_center_square, damage_circles, damage_half
from .model import NCA, Perception, rgba
from .targets import load_target, make_seed, make_target
from .train import load_checkpoint


def to_image(x: torch.Tensor, scale: int = 6, background: float = 1.0) -> Image.Image:
    """Composite one state's RGBA channels over a background and upscale crisply."""
    img = rgba(x)[0].permute(1, 2, 0).detach().cpu().numpy()
    rgb, alpha = img[..., :3], img[..., 3:4]
    composed = rgb * alpha + background * (1.0 - alpha)
    arr = (composed.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    if scale > 1:
        pil = pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST)
    return pil


def label(pil: Image.Image, text: str) -> Image.Image:
    out = pil.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, out.width, 14], fill=(255, 255, 255))
    draw.text((3, 2), text, fill=(30, 30, 30))
    return out


def save_gif(frames: list[Image.Image], path: Path, duration_ms: int = 60) -> None:
    if not frames:
        raise ValueError("no frames to save")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


def save_grid(frames: list[Image.Image], path: Path, columns: int = 6) -> None:
    if not frames:
        return
    cols = min(columns, len(frames))
    rows = (len(frames) + cols - 1) // cols
    w, h = frames[0].size
    sheet = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, frame in enumerate(frames):
        sheet.paste(frame, ((i % cols) * w, (i // cols) * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _resolve_target(cfg, checkpoint_target: torch.Tensor, device: str) -> torch.Tensor:
    if cfg.target_path:
        return load_target(cfg.target_path, size=cfg.size).to(device)
    if checkpoint_target is not None:
        return checkpoint_target.to(device)
    return make_target(cfg.shape, size=cfg.size).to(device)


def cmd_grow(args: argparse.Namespace) -> None:
    device = args.device
    model, cfg, target, seed, iteration = load_checkpoint(args.checkpoint, device=device)
    target = _resolve_target(cfg, target, device)
    out_dir = Path(args.out_dir or Path(args.checkpoint).parent)

    _, frames = model.rollout(seed.clone(), steps=args.steps, fire_rate=cfg.fire_rate, record_every=args.every)
    images = [label(to_image(f, scale=args.scale), f"step {i * args.every}") for i, f in enumerate(frames, 1)]

    save_gif(images, out_dir / "growth.gif", duration_ms=args.duration)
    save_grid(images[:: max(1, len(images) // 6)][:6], out_dir / "growth_grid.png")
    to_image(target, scale=args.scale).save(out_dir / "target.png")

    iou = float((rgba(frames[-1])[:, 3] > 0.5).float().mean())
    print(f"growth: {len(images)} frames -> {out_dir / 'growth.gif'}  (cov {iou:.3f}, trained for {iteration} iters)")


def cmd_regenerate(args: argparse.Namespace) -> None:
    """Grow to convergence, then apply damage types never seen in training."""
    device = args.device
    model, cfg, target, seed, _ = load_checkpoint(args.checkpoint, device=device)
    target = _resolve_target(cfg, target, device)
    out_dir = Path(args.out_dir or Path(args.checkpoint).parent)

    grown = model.rollout(seed.clone(), steps=args.steps, fire_rate=cfg.fire_rate)[0]

    damages = {
        "circle": lambda s: damage_circles(s, n=1, max_radius_frac=0.35),
        "half-left": lambda s: damage_half(s, "left"),
        "half-top": lambda s: damage_half(s, "top"),
        "square": lambda s: damage_center_square(s, frac=0.5),
    }

    for name, fn in damages.items():
        hurt = fn(grown)
        _, frames = model.rollout(hurt, steps=args.recovery, fire_rate=cfg.fire_rate, record_every=args.every)
        images = [label(to_image(hurt, scale=args.scale), f"{name}: damaged")]
        images += [label(to_image(f, scale=args.scale), f"{name}: step {i * args.every}") for i, f in enumerate(frames, 1)]
        save_gif(images, out_dir / f"regen_{name}.gif", duration_ms=args.duration)
        final_iou = float(
            ((rgba(frames[-1])[:, 3] > 0.5) & (target[:, 3] > 0.5)).sum()
            / ((rgba(frames[-1])[:, 3] > 0.5) | (target[:, 3] > 0.5)).sum().clamp_min(1)
        )
        print(f"  {name:10} recovered IoU {final_iou:.3f} -> {out_dir / f'regen_{name}.gif'}")


def cmd_rotate(args: argparse.Namespace) -> None:
    """Experiment 4: rotate the sensing axes and grow rotated patterns, no retraining."""
    device = args.device
    model, cfg, target, seed, _ = load_checkpoint(args.checkpoint, device=device)

    angles = [float(a) for a in args.angles.split(",")]
    panels: list[Image.Image] = []
    for deg in angles:
        theta = deg * 3.141592653589793 / 180.0
        model.perceive = Perception(cfg.channels, pad_mode=cfg.pad_mode, theta=theta).to(device)
        grown = model.rollout(seed.clone(), steps=args.steps, fire_rate=cfg.fire_rate)[0]
        panels.append(label(to_image(grown, scale=args.scale), f"theta = {deg:g} deg"))

    sheet = Image.new("RGB", (sum(p.width for p in panels), panels[0].height), (255, 255, 255))
    offset = 0
    for panel in panels:
        sheet.paste(panel, (offset, 0))
        offset += panel.width
    out = Path(args.out_dir or Path(args.checkpoint).parent) / "rotation.png"
    sheet.save(out)
    print(f"rotation: {len(panels)} angles -> {out}")


def cmd_plot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(args.run_dir)
    iterations, losses = [], []
    with (run_dir / "log.csv").open() as fh:
        for row in csv.DictReader(fh):
            if row["loss"]:
                iterations.append(int(row["iteration"]))
                losses.append(float(row["loss"]))

    eval_iters, ious, persist = [], [], []
    with (run_dir / "log.csv").open() as fh:
        for row in csv.DictReader(fh):
            if row["alpha_iou"]:
                eval_iters.append(int(row["iteration"]))
                ious.append(float(row["alpha_iou"]))
                if row["persist_alpha_iou"]:
                    persist.append(float(row["persist_alpha_iou"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(iterations, losses, lw=1.2)
    axes[0].set(xlabel="iteration", ylabel="loss", title="training loss", yscale="log")
    axes[0].grid(alpha=0.3)

    axes[1].plot(eval_iters, ious, marker="o", ms=3, label="after training steps")
    if persist:
        axes[1].plot(eval_iters, persist, marker="s", ms=3, label="after 2x steps (stability)")
    axes[1].set(xlabel="iteration", ylabel="alpha IoU", title="shape accuracy", ylim=(0, 1.02))
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = run_dir / "curves.png"
    fig.savefig(out, dpi=130)
    print(f"plot -> {out}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render a trained NCA.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--checkpoint", required=True)
        sp.add_argument("--steps", type=int, default=400)
        sp.add_argument("--scale", type=int, default=6)
        sp.add_argument("--duration", type=int, default=60)
        sp.add_argument("--out-dir", default=None)
        sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        sp.add_argument("--every", type=int, default=8)

    g = sub.add_parser("grow", help="animate growth from the seed")
    add_common(g)
    g.set_defaults(func=cmd_grow)

    r = sub.add_parser("regenerate", help="damage a grown pattern and watch it heal")
    add_common(r)
    r.add_argument("--recovery", type=int, default=300)
    r.set_defaults(func=cmd_regenerate)

    t = sub.add_parser("rotate", help="Experiment 4: rotated perceptive fields")
    add_common(t)
    t.add_argument("--angles", default="0,15,30,45,60,90")
    t.set_defaults(func=cmd_rotate)

    pl = sub.add_parser("plot", help="plot curves from a run directory")
    pl.add_argument("--run-dir", required=True)
    pl.set_defaults(func=cmd_plot)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
