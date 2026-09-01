#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuilds a problem/model from a checkpoint (2D or 3D, auto-detected) and
reports rel_L2 error plus a report figure. Usable as a library
(``from evaluate import evaluate``) or as a CLI.

Usage
-----
    python evaluate.py path/to/checkpoint.pkl
    python evaluate.py path/to/checkpoint.pkl --out-dir Evaluation/reports
    python evaluate.py path/to/2d_checkpoint.pkl --x0 1.5 --z0 1.0 --show
    python evaluate.py path/to/3d_checkpoint.pkl --n 121 --clip-percentile 99
    python evaluate.py path/to/checkpoint.pkl --base-path /custom/data
    python evaluate.py path/to/2d_checkpoint.pkl --save-npz --npz-out preds.npz
"""

from __future__ import annotations

import argparse, importlib.util, pickle, sys
from pathlib import Path

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repo root
sys.path.insert(0, str(ROOT))

from nap_wave.problem_setup import build_gi_problem, build_gi_problem_3d, resolve_reference_data_path
from nap_wave.archs import build_architecture
from nap_wave.lippmann_schwinger import compute_u0

__all__ = [
    "load_checkpoint", "find_example_root", "resolve_model_name", "is_3d_config",
    "evaluate", "compute_predictions_2d", "make_report_2d",
    "compute_predictions_3d", "make_report_3d",
]


# ============================================================
# Checkpoint loading (shared by 2D and 3D)
# ============================================================

class _TolerantUnpickler(pickle.Unpickler):
    """Swap unresolvable opt_state classes for a placeholder so unpickling
    still succeeds for a pure forward pass."""

    class _Placeholder:
        def __init__(self, *a, **kw):
            pass

        def __setstate__(self, state):
            self.__dict__["_state"] = state

        def __reduce__(self):
            return (_TolerantUnpickler._Placeholder, ())

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return self._Placeholder


def load_checkpoint(path):
    with open(Path(path), "rb") as f:
        return _TolerantUnpickler(f).load()


def _migrate_legacy_config(config):
    """Checkpoints saved before the config was reworked can carry a schema
    validate_config no longer accepts. Patch those in-memory (the checkpoint
    file on disk is never touched) so old checkpoints still evaluate.

    Currently handles: the old top-level `adaptive` section (its
    `stage_iterations` moved into `training.stage_iterations`)."""
    if config is None:
        return config
    adaptive_cfg = config.get("adaptive")
    if adaptive_cfg is not None:
        training_cfg = config.get("training")
        if training_cfg is not None and training_cfg.get("stage_iterations") is None:
            stage_iterations = adaptive_cfg.get("stage_iterations")
            if stage_iterations is not None:
                training_cfg["stage_iterations"] = stage_iterations
        del config["adaptive"]
    return config


KNOWN_MODELS = (
    "Marmousi", "Overthrust", "Otway",
    "Radial_velocity_gradient_3D", "Fluid_cylinder_scattering",
    # legacy names, kept for checkpoints downloaded under the old scheme
    "Exact_solution_3D", "Exact_solution_2D", "Exact_solution",
)


def find_example_root(checkpoint_path) -> Path:
    """
    Walk up from the checkpoint's folder to find the example's ROOT (the
    folder with config.py/data/), however deeply nested. Falls back to
    checkpoint_path.parent.parent if none is found.
    """
    here = Path(checkpoint_path).resolve().parent
    for _ in range(6):  # a handful of levels is plenty; avoids walking to "/"
        if (here / "config.py").exists():
            return here
        if here.parent == here:  # filesystem root
            break
        here = here.parent
    return Path(checkpoint_path).resolve().parent.parent


def resolve_model_name(checkpoint_path, run_name=None) -> str:
    checkpoint_path = Path(checkpoint_path)
    candidates = [run_name or "", checkpoint_path.stem]
    candidates += [p.name for p in checkpoint_path.parents[:4]]
    for cand in candidates:
        cand_lower = str(cand).lower()
        for name in KNOWN_MODELS:
            if name.lower() in cand_lower:
                return name
    return checkpoint_path.stem


def _load_example_module(example_root: Path, rel_path: str, module_tag: str):
    """Dynamically import example_root/rel_path, keyed by module_tag so
    same-named files across examples don't collide in sys.modules."""
    file_path = example_root / rel_path
    spec = importlib.util.spec_from_file_location(module_tag, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _live_config(example_root: Path):
    config_path = example_root / "config.py"
    if not config_path.exists():
        return None
    mod = _load_example_module(example_root, "config.py", f"live_config_{example_root.name}")
    return mod.get_config()


def is_3d_config(cfg) -> bool:
    """True if problem.domain has 6 entries (x0,x1,y0,y1,z0,z1), i.e. 3D."""
    domain = cfg["problem"].get("domain")
    return domain is not None and len(domain) == 6


# ============================================================
# Depth / k_init reconstruction (2D path) -- must match Trainer's own
# model-building exactly, since this reconstructs an already-trained
# checkpoint.
# ============================================================

def depth_at_checkpoint(history: dict, epoch) -> int | None:
    """Adaptive depth at the checkpoint's epoch (nearest-epoch match in
    history), or None if no adaptive-depth trace is recorded."""
    if not history or epoch is None:
        return None
    depths = history.get("adaptive_depth")
    epochs = history.get("epoch")
    if not depths or not epochs or len(depths) != len(epochs):
        return None
    epochs_arr = np.asarray(epochs, dtype=float)
    idx = int(np.argmin(np.abs(epochs_arr - float(epoch))))
    return int(depths[idx])


def depth_from_params(params) -> int:
    """3D depth reconstruction: count dense_* layers in the saved params."""
    return sum(1 for k in params.keys() if k.startswith("dense_"))


def make_k_init(arch_cfg, depth, problem):
    """Mirrors Trainer._make_k_init_for_depth so the forward pass uses the
    same basis the saved coefficients were fit to."""
    width = int(arch_cfg["width"])
    n_basis = int(depth) * (width // 2)

    k_init = getattr(problem, "k_init", None)
    if k_init is None:
        return None
    k_init = jnp.asarray(k_init, dtype=problem.real_dtype).reshape(-1)
    if k_init.size == 1:
        return jnp.repeat(k_init, n_basis)
    if k_init.size < n_basis:
        reps = int(np.ceil(n_basis / k_init.size))
        return jnp.tile(k_init, reps)[:n_basis]
    return k_init[:n_basis]


# ============================================================
# 2D forward pass / predictions
# ============================================================

def compute_predictions_2d(checkpoint_path, base_path=None) -> dict:
    """
    Load a 2D checkpoint, run only the forward pass, and return a dict of
    numpy arrays (pred/true/err fields, grid coords, extent, rel_l2, etc).
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    ckpt = load_checkpoint(checkpoint_path)

    config = _migrate_legacy_config(ckpt["config"])
    model_vars = ckpt["model_vars"]
    epoch = ckpt.get("epoch")
    history = ckpt.get("history", {}) or {}

    # Default base_path to this checkpoint's own example root.
    if base_path is None:
        base_path = find_example_root(checkpoint_path)
    config["problem"]["base_path"] = str(base_path)

    # Fill in any fields missing from the saved config.
    live_cfg = _live_config(Path(base_path))
    if live_cfg is not None:
        for key in ["domain", "v0", "source", "frequency"]:
            if config["problem"].get(key) is None and key in live_cfg.problem:
                config["problem"][key] = live_cfg.problem[key]

    problem = build_gi_problem(config)

    arch_cfg = config["arch"]
    depth = depth_at_checkpoint(history, epoch)
    if depth is None:
        depth = int(arch_cfg["depth"])

    model = build_architecture(
        name=arch_cfg.get("name", "plane_wave"),
        input_dim=problem.input_dim,
        width=arch_cfg["width"],
        depth=depth,
        coord_dim=problem.input_dim,
        k_init=make_k_init(arch_cfg, depth, problem),
        trainable_k=arch_cfg.get("trainable_k", False),
        k_min_clip=arch_cfg.get("k_min_clip", 0.25),
    )

    # ------------------------------------------------------------
    # Forward pass only. Model is fed coords directly -- no encoder
    # (x, z feed the basis functions unchanged).
    # ------------------------------------------------------------
    coords = problem.coords_data
    u_pred = model.apply(model_vars, coords)

    nx = problem.nx_data
    nz = problem.nz_data

    pred_real = np.asarray(jnp.real(u_pred)).reshape(nz, nx)
    pred_imag = np.asarray(jnp.imag(u_pred)).reshape(nz, nx)

    true_real = np.asarray(problem.u_real).reshape(nz, nx)
    true_imag = np.asarray(problem.u_imag).reshape(nz, nx)

    xz = np.asarray(coords)
    x = xz[:, 0].reshape(nz, nx)
    z = xz[:, 1].reshape(nz, nx)

    # Signed error (pred - true), not magnitude -- lets over/under-prediction
    # show up as opposite colors instead of collapsing both into "bright".
    err_real = pred_real - true_real
    err_imag = pred_imag - true_imag

    err_complex = (pred_real - true_real) + 1j * (pred_imag - true_imag)
    true_complex = true_real + 1j * true_imag
    rel_l2 = float(np.linalg.norm(err_complex) / (np.linalg.norm(true_complex) + 1e-16))

    return dict(
        pred_real=pred_real, pred_imag=pred_imag,
        true_real=true_real, true_imag=true_imag,
        err_real=err_real, err_imag=err_imag,
        x=x, z=z, xz=xz,
        extent=np.array([float(x.min()), float(x.max()), float(z.min()), float(z.max())]),
        nx_data=nx, nz_data=nz,
        run_name=ckpt.get("run_name"), epoch=epoch, depth=depth,
        rel_l2=rel_l2,
        model_name=resolve_model_name(checkpoint_path, ckpt.get("run_name")),
        checkpoint_path=str(checkpoint_path),
    )


def save_predictions_npz(predictions: dict, out_path) -> Path:
    """Write everything in `predictions` (except the plain checkpoint_path
    string, which np.savez would otherwise pickle-wrap) to an .npz."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **predictions)
    return out_path


# ============================================================
# 2D plotting helpers
# ============================================================

def _resolve_cmap(name: str, fallback: str = "viridis"):
    import matplotlib.pyplot as plt
    try:
        return plt.get_cmap(name)
    except ValueError:
        return plt.get_cmap(fallback)


def _bold_axes(ax, title=None, xlabel=None, ylabel=None):
    if title is not None:
        ax.set_title(title, fontweight="bold", fontsize=16, pad=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=14)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=14)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_fontsize(12)
    for spine in ax.spines.values():
        spine.set_linewidth(3.0)


def _bold_colorbar(cbar):
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_fontsize(12)
    cbar.outline.set_linewidth(3.0)


def _sci_colorbar(cbar, axis: str = "y"):
    """Force scientific-notation (10^-3 style) tick labels on an error colorbar."""
    from matplotlib.ticker import ScalarFormatter

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    ax_obj = cbar.ax.yaxis if axis == "y" else cbar.ax.xaxis
    ax_obj.set_major_formatter(fmt)
    # NOTE: do NOT call cbar.update_ticks() here -- on this matplotlib
    # version it resets the formatter back to the default (plain decimal)
    # tick labels, undoing set_major_formatter above.

    # The "x10^-N" multiplier text defaults to a tiny, easy-to-miss font --
    # make it big/bold/dark so it actually reads as part of the label.
    offset_text = ax_obj.get_offset_text()
    offset_text.set_fontweight("bold")
    offset_text.set_fontsize(18)
    offset_text.set_color("black")

    # Match the regular tick labels to the same larger size for consistency.
    for lbl in cbar.ax.get_yticklabels() + cbar.ax.get_xticklabels():
        lbl.set_fontsize(14)
        lbl.set_fontweight("bold")


def _err_vmax(err_2d, percentile: float = 99.0) -> float:
    """Percentile clip of |err_2d| (not plain max) so a single near-singular
    pixel (e.g. next to a source) doesn't wash out the whole error
    colorscale. err_2d is signed; this returns a magnitude to use as a
    symmetric +/-vmax around 0."""
    mag = np.abs(err_2d)
    vmax = float(np.percentile(mag, percentile))
    return vmax if vmax > 0 else float(np.max(mag)) or 1e-12


def _sorted_axes_and_data(true_2d, pred_2d, x, z):
    """RegularGridInterpolator needs strictly increasing axes; x[0,:] / z[:,0]
    may run either direction depending on the case's coordinate convention."""
    x_axis = x[0, :]
    z_axis = z[:, 0]
    xi = np.argsort(x_axis)
    zi = np.argsort(z_axis)
    return (
        x_axis[xi], z_axis[zi],
        true_2d[np.ix_(zi, xi)], pred_2d[np.ix_(zi, xi)],
    )


def make_report_2d(
    predictions: dict,
    out_dir: Path | None = None,
    show: bool = False,
    save: bool = True,
    x0: float | None = None,
    z0: float | None = None,
) -> Path | None:
    import warnings
    import matplotlib
    if not show:
        matplotlib.use("Agg")
        warnings.filterwarnings("ignore", message=".*cannot show the figure.*")
    import matplotlib.pyplot as plt
    from scipy.interpolate import RegularGridInterpolator

    x = predictions["x"]; z = predictions["z"]; extent = predictions["extent"]
    true_real, pred_real, err_real = (
        predictions["true_real"], predictions["pred_real"], predictions["err_real"]
    )
    true_imag, pred_imag, err_imag = (
        predictions["true_imag"], predictions["pred_imag"], predictions["err_imag"]
    )
    model_name = predictions["model_name"]
    epoch = predictions["epoch"]

    if x0 is None:
        x0 = float((extent[0] + extent[1]) / 2.0)
    if z0 is None:
        z0 = float((extent[2] + extent[3]) / 2.0)

    field_cmap = _resolve_cmap("seismic", fallback="seismic")
    err_cmap = _resolve_cmap("inferno", fallback="inferno")
    base = dict(extent=extent, origin="upper", aspect="auto")

    fig = plt.figure(figsize=(18, 15), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    def _map_row(row, true_2d, pred_2d, err_2d, label):
        vmax = max(float(np.max(np.abs(true_2d))), float(np.max(np.abs(pred_2d))))
        err_vmax = _err_vmax(err_2d)
        panels = [
            (0, true_2d, dict(vmin=-vmax, vmax=vmax), field_cmap, f"Reference {label}"),
            (1, pred_2d, dict(vmin=-vmax, vmax=vmax), field_cmap, f"Predicted {label}"),
            (2, err_2d, dict(vmin=-err_vmax, vmax=err_vmax), err_cmap, f"Error {label}"),
        ]
        for c, arr, scale_kwargs, cmap_obj, ttl in panels:
            ax = fig.add_subplot(gs[row, c])
            im = ax.imshow(arr, cmap=cmap_obj, **base, **scale_kwargs)
            _bold_axes(ax, title=ttl, xlabel="x", ylabel="z")
            cbar = fig.colorbar(im, ax=ax)
            _bold_colorbar(cbar)
            if c == 2:  # error panel -- scientific-notation ticks
                _sci_colorbar(cbar, axis="y")

    _map_row(0, true_real, pred_real, err_real, r"$Re(u_s)$")
    _map_row(1, true_imag, pred_imag, err_imag, r"$Im(u_s)$")

    # ---- Row 3: three cross-sections, REAL part only, plain background
    # (no grid lines) ----
    x_axis_raw = x[0, :]; z_axis_raw = z[:, 0]
    ix = int(np.argmin(np.abs(x_axis_raw - x0))); x_nearest = x_axis_raw[ix]
    iz = int(np.argmin(np.abs(z_axis_raw - z0))); z_nearest = z_axis_raw[iz]

    re_ref_style = dict(color="black", linestyle="-", linewidth=3.5)
    re_pred_style = dict(color="red", linestyle="--", linewidth=3.5)

    ax_fx = fig.add_subplot(gs[2, 0])
    ax_fx.plot(z[:, ix], true_real[:, ix], label="Reference", **re_ref_style)
    ax_fx.plot(z[:, ix], pred_real[:, ix], label="Prediction", **re_pred_style)
    _bold_axes(ax_fx, title=rf"Fixed $x={x_nearest:.4g}$",
               xlabel="z", ylabel=r"$Re(u_s)$")
    ax_fx.grid(True, which="both", linestyle="--", linewidth=1.0, alpha=0.5)

    ax_fz = fig.add_subplot(gs[2, 1])
    ax_fz.plot(x[iz, :], true_real[iz, :], label="Reference", **re_ref_style)
    ax_fz.plot(x[iz, :], pred_real[iz, :], label="Prediction", **re_pred_style)
    _bold_axes(ax_fz, title=rf"Fixed $z={z_nearest:.4g}$",
               xlabel="x", ylabel=r"$Re(u_s)$")
    ax_fz.grid(True, which="both", linestyle="--", linewidth=1.0, alpha=0.5)

    x_min, x_max, z_min, z_max = [float(v) for v in extent]
    t = np.linspace(0.0, 1.0, 300)
    x_diag = x_min + t * (x_max - x_min)
    z_diag = z_min + t * (z_max - z_min)
    s_diag = t * float(np.hypot(x_max - x_min, z_max - z_min))

    x_sorted, z_sorted, true_re_sorted, pred_re_sorted = _sorted_axes_and_data(
        true_real, pred_real, x, z,
    )

    interp_true_re = RegularGridInterpolator(
        (z_sorted, x_sorted), true_re_sorted, bounds_error=False, fill_value=None,
    )
    interp_pred_re = RegularGridInterpolator(
        (z_sorted, x_sorted), pred_re_sorted, bounds_error=False, fill_value=None,
    )

    pts = np.stack([z_diag, x_diag], axis=-1)
    ax_diag = fig.add_subplot(gs[2, 2])
    ax_diag.plot(s_diag, interp_true_re(pts), label="Reference", **re_ref_style)
    ax_diag.plot(s_diag, interp_pred_re(pts), label="Prediction", **re_pred_style)
    _bold_axes(
        ax_diag, title=r"Diagonal cut $(x_{\min},z_{\min})\to(x_{\max},z_{\max})$",
        xlabel="distance along diagonal", ylabel=r"$Re(u_s)$",
    )
    ax_diag.grid(True, which="both", linestyle="--", linewidth=1.0, alpha=0.5)

    for ax in (ax_fx, ax_fz, ax_diag):
        leg = ax.legend(loc="best", frameon=True, fontsize=9)
        for txt in leg.get_texts():
            txt.set_fontweight("bold")

    save_path = None
    if save:
        # Default: the checkpoint's own folder, not a shared reports/ dir.
        out_dir = Path(out_dir) if out_dir is not None else Path(predictions["checkpoint_path"]).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"{model_name}_epoch{epoch}_error_report.png"
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"  saved -> {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


# ============================================================
# 3D forward pass / predictions / report
# ============================================================

def _face_grid(axis: str, fixed: float, extent, n: int):
    """Coordinates of an n x n plane at a fixed x/y/z. Returns (X, Y, Z)."""
    x0, x1, y0, y1, z0, z1 = extent
    if axis == "z":
        xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        Z = np.full_like(X, fixed)
    elif axis == "y":
        xs, zs = np.linspace(x0, x1, n), np.linspace(z0, z1, n)
        X, Z = np.meshgrid(xs, zs, indexing="xy")
        Y = np.full_like(X, fixed)
    elif axis == "x":
        ys, zs = np.linspace(y0, y1, n), np.linspace(z0, z1, n)
        Y, Z = np.meshgrid(ys, zs, indexing="xy")
        X = np.full_like(Y, fixed)
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}.")
    return X, Y, Z


def box_faces_grid(extent, n: int):
    """The 6 outer faces of the domain box, as (X, Y, Z) grids."""
    x0, x1, y0, y1, z0, z1 = extent
    return [
        _face_grid("x", x0, extent, n),
        _face_grid("x", x1, extent, n),
        _face_grid("y", y0, extent, n),
        _face_grid("y", y1, extent, n),
        _face_grid("z", z0, extent, n),
        _face_grid("z", z1, extent, n),
    ]


def center_slices_grid(extent, n: int):
    """The 3 center slices of the domain box, as (X, Y, Z) grids."""
    x0, x1, y0, y1, z0, z1 = extent
    xc = (x0 + x1) / 2
    yc = (y0 + y1) / 2
    zc = (z0 + z1) / 2
    return [
        _face_grid("x", xc, extent, n),
        _face_grid("y", yc, extent, n),
        _face_grid("z", zc, extent, n),
    ]


def _regular_grid_complex_interpolator(xyz_ref, values_complex):
    """
    Fast cubic interpolator for a complex field on a regular 3D grid;
    returns callable(xyz) -> complex [N]. Not griddata: too slow at scale.
    """
    from scipy.interpolate import RegularGridInterpolator

    xyz_ref = np.asarray(xyz_ref, dtype=np.float64)
    x_axis = np.unique(xyz_ref[:, 0])
    y_axis = np.unique(xyz_ref[:, 1])
    z_axis = np.unique(xyz_ref[:, 2])
    nx, ny, nz = x_axis.size, y_axis.size, z_axis.size
    if nx * ny * nz != xyz_ref.shape[0]:
        raise ValueError(
            f"xyz_ref is not a regular {nx}x{ny}x{nz} grid: "
            f"nx*ny*nz={nx * ny * nz} != {xyz_ref.shape[0]} points."
        )

    # xyz_ref/values follow the (z, y, x) "ij"-meshgrid flatten order used
    # throughout nap_wave.problem_setup (see build_gi_grid_coords_3d).
    grid = np.asarray(values_complex, dtype=np.complex128).reshape(nz, ny, nx)

    # Cubic avoids a faceted/moire pattern at finer plot resolutions.
    method = "cubic" if min(nx, ny, nz) >= 4 else "linear"
    interp_re = RegularGridInterpolator(
        (z_axis, y_axis, x_axis), grid.real, method=method,
        bounds_error=False, fill_value=None,
    )
    interp_im = RegularGridInterpolator(
        (z_axis, y_axis, x_axis), grid.imag, method=method,
        bounds_error=False, fill_value=None,
    )

    def _interp(xyz):
        pts = np.asarray(xyz, dtype=np.float64)[:, [2, 1, 0]]  # -> (z, y, x)
        return interp_re(pts) + 1j * interp_im(pts)

    return _interp


def _eval_fields_3d(X, Y, Z, omega, c0, source, problem, model, model_vars, us_ref_interp):
    """Exact (from stored validation data) and predicted complex total
    field on a slice plane's [n,n] grid."""
    xyz = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    coords = jnp.asarray(xyz, dtype=problem.real_dtype)
    U0 = compute_u0(
        coords=coords, source=jnp.asarray(source, dtype=problem.real_dtype),
        v0=c0, omega=omega, real_dtype=problem.real_dtype,
    )
    U0_c = np.asarray(U0[:, 0]) + 1j * np.asarray(U0[:, 1])

    Us_exact = us_ref_interp(xyz)
    p_exact = (U0_c + Us_exact).reshape(X.shape)

    # Model is fed coords directly -- no encoder (x, y, z feed the basis
    # functions unchanged).
    Us_pred = np.asarray(model.apply(model_vars, coords)).reshape(-1)
    p_pred = (U0_c + Us_pred).reshape(X.shape)

    return p_exact, p_pred


def _add_box_panel(fig, pos, extent, faces, values, vmin, vmax, cmap, title, plt, sci=False):
    """One clean 3D cube panel: the 6 box faces colored by `values`, no
    axes/ticks -- just the cube, a title, and a horizontal colorbar."""
    ax = fig.add_subplot(*pos, projection="3d")
    x0, x1, y0, y1, z0, z1 = extent
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for (X, Y, Z), val in zip(faces, values):
        ax.plot_surface(
            X, Y, Z, facecolors=cmap(norm(val)), rstride=1, cstride=1,
            shade=False, antialiased=False, linewidth=0,
        )

    # Draw black bounding box edges for better 3D clarity
    edges = [
        ([x0, x1], [y0, y0], [z0, z0]), ([x0, x1], [y1, y1], [z0, z0]),
        ([x0, x1], [y0, y0], [z1, z1]), ([x0, x1], [y1, y1], [z1, z1]),
        ([x0, x0], [y0, y1], [z0, z0]), ([x1, x1], [y0, y1], [z0, z0]),
        ([x0, x0], [y0, y1], [z1, z1]), ([x1, x1], [y0, y1], [z1, z1]),
        ([x0, x0], [y0, y0], [z0, z1]), ([x1, x1], [y0, y0], [z0, z1]),
        ([x0, x0], [y1, y1], [z0, z1]), ([x1, x1], [y1, y1], [z0, z1]),
    ]
    for ex, ey, ez in edges:
        ax.plot(ex, ey, ez, color='black', linewidth=1.5)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.view_init(elev=22, azim=45)
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, fontweight='bold', pad=-5)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(
        mappable, ax=ax, orientation="horizontal",
        shrink=0.75, pad=0.0, aspect=20,
    )
    cbar.set_ticks(np.linspace(vmin, vmax, 5))
    cbar.ax.tick_params(labelsize=11, width=1.5, length=5)
    for tick in cbar.ax.get_xticklabels():
        tick.set_fontweight('bold')
    cbar.outline.set_linewidth(1.5)
    if sci:
        _sci_colorbar(cbar, axis="x")
    return ax


def compute_predictions_3d(checkpoint_path, base_path=None, n: int = 121) -> dict:
    """Load a 3D checkpoint (using its own saved config), evaluate the 6
    box faces + 3 center slices against the stored validation data, and
    compute rel_L2 over problem.coords_data."""
    checkpoint_path = Path(checkpoint_path).resolve()
    ckpt = load_checkpoint(checkpoint_path)
    model_vars = ckpt["model_vars"]
    epoch = ckpt.get("epoch")
    run_name = ckpt.get("run_name")

    example_root = Path(base_path) if base_path is not None else find_example_root(checkpoint_path)
    live_cfg = _live_config(example_root)

    cfg = _migrate_legacy_config(ckpt.get("config"))
    if cfg is None:
        cfg = live_cfg
    elif live_cfg is not None:
        # Patch only genuinely missing fields from the live config, same
        # pattern the 2D path uses.
        for key in [
            "domain", "frequency", "v0", "c0", "source",
            "gi_nx", "gi_ny", "gi_nz", "gi_damp", "n_total", "w_damp", "precision",
        ]:
            if cfg["problem"].get(key) is None and key in live_cfg.problem:
                cfg["problem"][key] = live_cfg.problem[key]
        if cfg["arch"].get("width") is None:
            cfg["arch"]["width"] = live_cfg.arch.width

    # Override with this checkpoint's own example root (2D path does the same).
    cfg["problem"]["base_path"] = str(example_root)

    pc = cfg["problem"]
    problem = build_gi_problem_3d(cfg)

    data_path = resolve_reference_data_path(pc)
    ref_data = np.load(data_path)
    us_ref_interp = _regular_grid_complex_interpolator(
        ref_data["xyz_ref"],
        ref_data["U_ref"][:, 0] + 1j * ref_data["U_ref"][:, 1],
    )

    arch_cfg = cfg["arch"]
    depth = depth_from_params(model_vars["params"])
    model = build_architecture(
        name=arch_cfg.get("name", "plane_wave"), input_dim=problem.input_dim, width=arch_cfg["width"],
        depth=depth,
        coord_dim=problem.input_dim,
        k_init=getattr(problem, "k_init", None),
        trainable_k=arch_cfg.get("trainable_k", False), k_min_clip=arch_cfg.get("k_min_clip", 0.25),
    )

    extent = [float(v) for v in pc.domain]
    omega = 2.0 * np.pi * float(pc.frequency)
    # v0 (background velocity): "c0" accepted as a legacy alias.
    c0 = float(pc.get("v0", pc.get("c0")))
    source = tuple(pc.source)

    # The 6 outer faces of the domain box.
    faces = box_faces_grid(extent, n)
    exact_faces, pred_faces = [], []
    for X, Y, Z in faces:
        p_exact, p_pred = _eval_fields_3d(
            X, Y, Z, omega, c0, source, problem, model, model_vars, us_ref_interp,
        )
        exact_faces.append(p_exact)
        pred_faces.append(p_pred)
    # Signed error of the real part (pred - true), matching what the
    # "Predicted"/"Exact" panels themselves plot -- not a magnitude.
    err_faces = [pp.real - pe.real for pp, pe in zip(pred_faces, exact_faces)]

    # The 3 center slices.
    slice_faces = center_slices_grid(extent, n)
    exact_slices, pred_slices = [], []
    for X, Y, Z in slice_faces:
        p_exact, p_pred = _eval_fields_3d(
            X, Y, Z, omega, c0, source, problem, model, model_vars, us_ref_interp,
        )
        exact_slices.append(p_exact)
        pred_slices.append(p_pred)
    err_slices = [pp.real - pe.real for pp, pe in zip(pred_slices, exact_slices)]

    # rel_L2 over problem.coords_data, scattered field on both sides (do not
    # add U0 -- the reference is already Us, not the total field).
    Us_pred_data = np.asarray(model.apply(model_vars, problem.coords_data)).reshape(-1)
    u_ref_c = np.asarray(problem.u_real).reshape(-1) + 1j * np.asarray(problem.u_imag).reshape(-1)
    rel_l2 = float(np.linalg.norm(Us_pred_data - u_ref_c) / (np.linalg.norm(u_ref_c) + 1e-30))

    return dict(
        extent=extent, faces=faces, slice_faces=slice_faces,
        exact_faces=exact_faces, pred_faces=pred_faces, err_faces=err_faces,
        exact_slices=exact_slices, pred_slices=pred_slices, err_slices=err_slices,
        grid_shape=problem.grid_shape, w_damp=problem.w_damp, metric_points=int(u_ref_c.size),
        run_name=run_name, epoch=epoch, depth=depth,
        rel_l2=rel_l2,
        model_name=resolve_model_name(checkpoint_path, run_name),
        checkpoint_path=str(checkpoint_path),
    )


def make_report_3d(
    predictions: dict,
    out_dir: Path | None = None,
    save: bool = True,
    clip_percentile: float = 99.0,
) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = predictions["extent"]
    faces, slice_faces = predictions["faces"], predictions["slice_faces"]
    exact_faces, pred_faces, err_faces = (
        predictions["exact_faces"], predictions["pred_faces"], predictions["err_faces"]
    )
    exact_slices, pred_slices, err_slices = (
        predictions["exact_slices"], predictions["pred_slices"], predictions["err_slices"]
    )

    exact_re = [p.real for p in exact_faces]
    pred_re = [p.real for p in pred_faces]
    exact_re_slices = [p.real for p in exact_slices]
    pred_re_slices = [p.real for p in pred_slices]

    vmax = float(np.percentile(
        np.abs(np.concatenate([v.ravel() for v in exact_re + pred_re])), clip_percentile,
    ))
    # err_faces/err_slices are signed; clip on |error| so the colorbar stays
    # symmetric around 0 (diverging cmap) instead of one-sided.
    err_vmax = float(np.percentile(
        np.abs(np.concatenate([v.ravel() for v in err_faces])), clip_percentile,
    )) or 1e-12

    fig = plt.figure(figsize=(20, 12))

    _add_box_panel(fig, (2, 3, 1), extent, faces, pred_re, -vmax, vmax, plt.cm.seismic, "Predicted", plt)
    _add_box_panel(fig, (2, 3, 2), extent, faces, exact_re, -vmax, vmax, plt.cm.seismic, "Exact", plt)
    _add_box_panel(fig, (2, 3, 3), extent, faces, err_faces, -err_vmax, err_vmax, plt.cm.inferno, "Error", plt, sci=True)

    _add_box_panel(fig, (2, 3, 4), extent, slice_faces, pred_re_slices, -vmax, vmax, plt.cm.seismic, "Predicted (Slices)", plt)
    _add_box_panel(fig, (2, 3, 5), extent, slice_faces, exact_re_slices, -vmax, vmax, plt.cm.seismic, "Exact (Slices)", plt)
    _add_box_panel(fig, (2, 3, 6), extent, slice_faces, err_slices, -err_vmax, err_vmax, plt.cm.inferno, "Error (Slices)", plt, sci=True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = None
    if save:
        out_dir = Path(out_dir) if out_dir is not None else Path(predictions["checkpoint_path"]).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"{predictions['model_name']}_epoch{predictions['epoch']}_error_report_3d.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  saved -> {save_path}")
    plt.close(fig)
    return save_path


# ============================================================
# Dispatcher -- auto-detects 2D vs 3D from the checkpoint's own config
# ============================================================

def evaluate(
    checkpoint_path,
    base_path=None,
    out_dir=None,
    show: bool = False,
    save: bool = True,
    x0: float | None = None,
    z0: float | None = None,
    n: int = 121,
    clip_percentile: float = 99.0,
) -> dict:
    """Auto-detect 2D/3D and run the matching evaluation + report pipeline,
    returning predictions with 'is_3d' and 'report_path' added."""
    checkpoint_path = Path(checkpoint_path).resolve()
    ckpt = load_checkpoint(checkpoint_path)
    example_root = Path(base_path) if base_path is not None else find_example_root(checkpoint_path)

    cfg = ckpt.get("config") or _live_config(example_root)
    if cfg is None:
        raise FileNotFoundError(
            f"Could not determine problem dimensionality for {checkpoint_path}: "
            f"checkpoint has no saved config, and no config.py found under {example_root}."
        )

    if is_3d_config(cfg):
        predictions = compute_predictions_3d(checkpoint_path, base_path=example_root, n=n)
        report_path = make_report_3d(predictions, out_dir=out_dir, save=save, clip_percentile=clip_percentile)
    else:
        predictions = compute_predictions_2d(checkpoint_path, base_path=example_root)
        report_path = make_report_2d(predictions, out_dir=out_dir, show=show, save=save, x0=x0, z0=z0)

    predictions["is_3d"] = is_3d_config(cfg)
    predictions["report_path"] = str(report_path) if report_path else None
    return predictions


# ============================================================
# CLI
# ============================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("checkpoint", type=Path, help="path to a checkpoint .pkl file (2D or 3D -- auto-detected)")
    p.add_argument("--base-path", type=Path, default=None,
                    help="override the example root (default: auto-detected by "
                         "walking up from the checkpoint to the nearest folder "
                         "containing that example's own config.py)")
    p.add_argument("--out-dir", type=Path, default=None,
                    help="where to save the report PNG (default: the same folder "
                         "the checkpoint itself lives in)")
    p.add_argument("--no-save", action="store_true",
                    help="don't write the report PNG to disk (saving is ON by default)")
    p.add_argument("--x0", type=float, default=None,
                    help="[2D only] cross-section x position (default: domain center)")
    p.add_argument("--z0", type=float, default=None,
                    help="[2D only] cross-section z position (default: domain center)")
    p.add_argument("--show", action="store_true",
                    help="[2D only] also open an interactive window (off by default)")
    p.add_argument("--save-npz", action="store_true",
                    help="[2D only] also write the raw predictions to an .npz")
    p.add_argument("--npz-out", type=Path, default=None,
                    help="[2D only] predictions .npz path (only used with --save-npz; "
                         "default: next to the checkpoint)")
    p.add_argument("--n", type=int, default=121,
                    help="[3D only] grid resolution per box face")
    p.add_argument("--clip-percentile", type=float, default=99.0,
                    help="[3D only] color-scale clip, avoids the source singularity saturating everything")
    return p.parse_args()


def main():
    args = _parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    ckpt = load_checkpoint(checkpoint_path)
    example_root = Path(args.base_path) if args.base_path is not None else find_example_root(checkpoint_path)
    cfg = ckpt.get("config") or _live_config(example_root)
    if cfg is None:
        raise FileNotFoundError(
            f"Could not determine problem dimensionality for {checkpoint_path}: "
            f"checkpoint has no saved config, and no config.py found under {example_root}."
        )
    dim3d = is_3d_config(cfg)

    if dim3d:
        predictions = compute_predictions_3d(checkpoint_path, base_path=example_root, n=args.n)
        print(f"model      : {predictions['model_name']}  (3D)")
        print(f"run_name   : {predictions['run_name']}")
        print(f"epoch      : {predictions['epoch']}")
        print(f"depth      : {predictions['depth']}")
        gz, gy, gx = predictions['grid_shape']
        print(f"GI grid    : {gz} x {gy} x {gx}   w_damp: {predictions['w_damp']}")
        print(f"metric grid: {predictions['metric_points']} points (problem.coords_data)")
        print(f"rel_L2     : {predictions['rel_l2']:.6e}  ({predictions['rel_l2'] * 100:.4f}%)")
        make_report_3d(
            predictions, out_dir=args.out_dir, save=not args.no_save,
            clip_percentile=args.clip_percentile,
        )
    else:
        predictions = compute_predictions_2d(checkpoint_path, base_path=example_root)
        print(f"model      : {predictions['model_name']}  (2D)")
        print(f"run_name   : {predictions['run_name']}")
        print(f"epoch      : {predictions['epoch']}")
        print(f"depth      : {predictions['depth']}")
        print(f"grid       : {predictions['nx_data']} x {predictions['nz_data']}")
        print(f"rel_L2     : {predictions['rel_l2']:.6e}  ({predictions['rel_l2'] * 100:.4f}%)")

        if args.save_npz:
            out_path = args.npz_out
            if out_path is None:
                stem = checkpoint_path.stem
                out_name = stem.replace("_checkpoint", "_predictions_reeval") + ".npz"
                out_path = checkpoint_path.with_name(out_name)
            save_predictions_npz(predictions, out_path)
            print(f"saved -> {out_path}")

        make_report_2d(
            predictions, out_dir=args.out_dir, show=args.show,
            save=not args.no_save, x0=args.x0, z0=args.z0,
        )


if __name__ == "__main__":
    main()
