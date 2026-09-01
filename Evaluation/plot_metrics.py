#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training-curve report for any case's <run_name>_metrics.npy: GI loss and
relative L2 error vs. epoch (--loglog for a log-log style). Adaptive-depth
transitions are marked as vertical dotted lines.

Usage
-----
    python Evaluation/plot_metrics.py --case Marmousi
    python Evaluation/plot_metrics.py --case Fluid_cylinder_scattering --loglog
    python Evaluation/plot_metrics.py /path/to/some_metrics.npy --loglog --show
"""
from __future__ import annotations

import argparse, importlib.util, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fd_scattered_grid_sweep import _locate_repo  # noqa: E402


def _load_config(case: str, repo_root: Path):
    cfg_path = repo_root / "Examples" / case / "config.py"
    spec = importlib.util.spec_from_file_location(f"{case}_config", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def find_metrics_file(case: str, repo_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        explicit = Path(explicit)
        if not explicit.exists():
            raise FileNotFoundError(f"metrics file not found: {explicit}")
        return explicit

    if case is None:
        raise ValueError("Pass either a metrics path or --case.")

    sys.path.insert(0, str(repo_root))
    cfg = _load_config(case, repo_root)
    from nap_wave.utils import make_run_name

    run_name = make_run_name(cfg)
    case_dir = repo_root / "Examples" / case
    base_path = Path(str(cfg["problem"]["base_path"]).rstrip("/"))
    candidates = [
        case_dir / "Results" / f"{run_name}_metrics.npy",
        case_dir / "data" / "Results" / f"{run_name}_metrics.npy",
        case_dir / "results" / "Results" / f"{run_name}_metrics.npy",
        base_path / "Results" / f"{run_name}_metrics.npy",  # config's own base_path, if it resolves locally
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No metrics file found for run_name={run_name!r} in any of:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nRun that case's main.py first, or pass an explicit metrics path."
    )


def load_history(path: Path) -> dict:
    return np.load(path, allow_pickle=True).item()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """
    Centered moving average of ``values`` with an edge-shrinking window
    (no NaN padding, no length loss).
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0 or window <= 1:
        return values.copy()
    half = window // 2
    csum = np.cumsum(np.insert(values, 0, 0.0))
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


def _stage_transitions(epoch, depth):
    if depth is None or len(depth) != len(epoch):
        return []
    return [
        (int(epoch[i]), int(depth[i]))
        for i in range(1, len(depth))
        if depth[i] != depth[i - 1]
    ]


def plot_metrics_linear(history: dict, out_path: Path, run_name: str, show: bool = False, ma_window: int = 101) -> Path:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epoch = history.get("epoch", [])
    if len(epoch) == 0:
        raise ValueError("This metrics file has no recorded epochs yet.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # loss (quadratic) is paired with the SQUARED rel-L2 error, not the
    # unsquared one, so both panels are quadratic quantities (same units).
    panels = [
        (axes[0], "loss", r"GI Loss $\mathcal{J}$ ($\propto \frac{1}{2}\|r(v)\|^2$)", "tab:blue"),
        (axes[1], "rel_l2_complex_sq", r"Relative $L_2$ Error$^2$ vs. exact solution, $(\|e(v)\|/\|u\|)^2$", "tab:red"),
    ]

    for ax, key, title, color in panels:
        values = history.get(key, [])
        if len(values) == len(epoch) and len(values) > 0:
            values = np.asarray(values)
            ax.plot(epoch, values, color=color, linewidth=0.8, alpha=0.35, label="raw", zorder=1)
            ma = moving_average(values, ma_window)
            ax.plot(epoch, ma, color=color, linewidth=2.0, label=f"moving avg (w={ma_window})", zorder=2)
            ax.legend(fontsize=8, loc="upper right")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch", fontweight="bold", fontsize=11)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.grid(True, which="both", axis="y", alpha=0.3)
        ax.tick_params(labelsize=10)

    depth = history.get("adaptive_depth")
    for epoch_at, new_depth in _stage_transitions(epoch, depth):
        for ax, _, _, _ in panels:
            ax.axvline(epoch_at, color="gray", linestyle=":", linewidth=1.0, alpha=0.6, zorder=0)
            ax.annotate(
                f"depth {new_depth}", xy=(epoch_at, 1.0), xycoords=("data", "axes fraction"),
                xytext=(3, -3), textcoords="offset points", rotation=90,
                va="top", ha="left", fontsize=8, color="dimgray",
            )

    loss_hist = history.get("loss", [])
    final_rel_l2 = history.get("rel_l2_complex", [None])[-1]
    if len(loss_hist) > 0 and final_rel_l2 is not None:
        min_loss = float(np.min(loss_hist))
        final_loss = loss_hist[-1]
        fig.suptitle(
            f"{run_name}\nepoch {int(epoch[-1])}  |  loss: final={final_loss:.3e}, min={min_loss:.3e}  |  "
            f"rel_L2={final_rel_l2:.3e} ({final_rel_l2 * 100:.2f}%)",
            fontsize=10, fontweight="bold",
        )
    else:
        fig.suptitle(run_name, fontsize=10, fontweight="bold")

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_metrics_loglog(history: dict, out_path: Path, run_name: str, show: bool = False, ma_window: int = 101) -> Path:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epoch = np.asarray(history.get("epoch", []), dtype=float)
    if epoch.size == 0:
        raise ValueError("This metrics file has no recorded epochs yet.")
    epoch_x = np.maximum(epoch, 1.0)  # log-x needs epoch>=1; epoch=0 -> 1

    depth = history.get("adaptive_depth")
    transitions = _stage_transitions(epoch, depth)

    def mark(ax):
        for epoch_at, new_depth in transitions:
            x = max(epoch_at, 1)
            ax.axvline(x, color="gray", linestyle=":", linewidth=1.0, alpha=0.6, zorder=0)
            ax.annotate(
                f"depth {new_depth}", xy=(x, 0.98), xycoords=("data", "axes fraction"),
                xytext=(3, -3), textcoords="offset points", rotation=90,
                va="top", ha="left", fontsize=8, color="dimgray",
            )

    n_panels = 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 5.2), constrained_layout=True)
    axes = list(axes)

    ax = axes[0]
    loss = np.asarray(history.get("loss", []))
    ax.loglog(epoch_x, loss, color="tab:blue", linewidth=0.8, alpha=0.35, label="raw")
    ax.loglog(epoch_x, moving_average(loss, ma_window), color="tab:blue", linewidth=2.0, label=f"moving avg (w={ma_window})")
    ax.set_title(r"GI Loss $\mathcal{J}$  ($\propto \frac{1}{2}\|r(v)\|^2$)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Epoch + 1", fontweight="bold")
    ax.set_ylabel(r"$\mathcal{J}$")
    ax.legend(fontsize=8, loc="upper right")
    mark(ax)

    ax = axes[1]
    rel_l2 = np.asarray(history.get("rel_l2_complex", []))
    ax.loglog(epoch_x, rel_l2, color="tab:red", linewidth=0.8, alpha=0.35, label="raw")
    ax.loglog(epoch_x, moving_average(rel_l2, ma_window), color="tab:red", linewidth=2.0, label=f"moving avg (w={ma_window})")
    ax.set_title(r"Relative $L_2$ Error  $\|e(v)\|/\|u_{ref}\|$ (validation grid)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Epoch + 1", fontweight="bold")
    ax.set_ylabel(r"$\|e(v)\|/\|u_{ref}\|$")
    ax.legend(fontsize=8, loc="upper right")
    mark(ax)

    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=10)

    final_rel_l2 = history.get("rel_l2_complex", [None])[-1]
    if loss.size > 0 and final_rel_l2 is not None:
        min_loss = float(np.min(loss))
        final_loss = loss[-1]
        fig.suptitle(
            f"{run_name}\nepoch {int(epoch[-1])}  |  loss: final={final_loss:.3e}, min={min_loss:.3e}  |  "
            f"rel_L2={final_rel_l2:.3e} ({final_rel_l2 * 100:.4f}%)",
            fontsize=11, fontweight="bold",
        )
    else:
        fig.suptitle(run_name, fontsize=11, fontweight="bold")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", type=Path, nargs="?", default=None, help="explicit *_metrics.npy path (alternative to --case)")
    p.add_argument("--case", default=None, help="any Examples/<case> name -- auto-finds that config's metrics.npy")
    p.add_argument("--loglog", action="store_true", help="3-panel log-log style instead of the default 2-panel semilogy style")
    p.add_argument("--out", type=Path, default=None, help="output PNG path (default: next to the metrics file)")
    p.add_argument("--show", action="store_true", help="also open an interactive window")
    p.add_argument("--ma-window", type=int, default=101, help="moving-average window (in saved points) overlaid on loss/rel_L2 curves")
    p.add_argument("--repo-root", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.metrics is None and args.case is None:
        raise SystemExit("Pass either a metrics path or --case.")

    repo_root = _locate_repo(args.repo_root)
    metrics_path = find_metrics_file(args.case, repo_root, args.metrics)
    history = load_history(metrics_path)
    run_name = history.get("run_name", metrics_path.stem.replace("_metrics", ""))

    suffix = "_metrics_loglog.png" if args.loglog else "_metrics.png"
    out_path = args.out if args.out is not None else metrics_path.with_name(f"{run_name}{suffix}")

    print(f"Loading metrics: {metrics_path}")
    print(f"  epochs recorded : {len(history.get('epoch', []))}")
    if history.get("loss"):
        print(f"  final loss      : {history['loss'][-1]:.6e}")
        print(f"  min loss        : {min(history['loss']):.6e}")
    if history.get("rel_l2_complex"):
        print(f"  final rel_L2    : {history['rel_l2_complex'][-1]:.6e}")

    plot_fn = plot_metrics_loglog if args.loglog else plot_metrics_linear
    out_path = plot_fn(history, out_path, run_name, show=args.show, ma_window=args.ma_window)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
