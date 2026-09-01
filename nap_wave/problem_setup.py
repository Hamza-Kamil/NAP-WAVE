"""
Builds the Lippmann-Schwinger problem: loads/derives delta_m, nondimensionalizes
coordinates, and precomputes G0's FFT and the fixed RHS f = G[u0].
"""

from dataclasses import dataclass
from pathlib import Path
import inspect, numpy as np

import jax.numpy as jnp

from .lippmann_schwinger import (
    compute_u0,
    build_green_kernel_fft_2d,
    green_apply_2d,
    green_apply,
    green_apply_complex,
    build_green_kernel_fft,
    compute_G0_regularized,
    green_3d_background_fn,
    to_complex_grid,
)
from .utils import get_dtypes, configure_precision, validate_config


# ============================================================
# Automatic base_path inference
# ============================================================

_PACKAGE_DIR = Path(__file__).resolve().parent


def _infer_base_path():
    for frame_info in inspect.stack():
        caller_path = Path(frame_info.filename).resolve()
        if _PACKAGE_DIR not in caller_path.parents and caller_path != Path(__file__).resolve():
            return str(caller_path.parent)
    return str(Path.cwd())


def _ensure_base_path(problem_cfg):
    """Fill in problem_cfg['base_path'] via _infer_base_path()."""
    if not problem_cfg.get("base_path"):
        problem_cfg["base_path"] = _infer_base_path()
    return problem_cfg["base_path"]


def _sample_k_pool(K_pool, n_basis, seed):
    """Draw n_basis wavenumbers from the quantile local-wavenumber pool."""
    rng = np.random.default_rng(seed=int(seed))
    if K_pool.size >= n_basis:
        return rng.choice(K_pool, size=n_basis, replace=False)
    return rng.choice(K_pool, size=n_basis, replace=True)


def _sample_k_init(K_flat, arch_cfg, problem_cfg, np_real_dtype, real_dtype):
    """k_init for the plane-wave arch: quantile-filter K_flat=omega/c(x), then
    draw depth*(width//2) samples from the filtered pool."""
    width = int(arch_cfg.get("width", 128))
    depth = int(arch_cfg.get("depth", 5))
    n_basis = depth * (width // 2)

    low_q = float(problem_cfg.get("k_low_quantile", 0.05))
    high_q = float(problem_cfg.get("k_high_quantile", 0.95))

    k_low = np.quantile(K_flat, low_q)
    k_high = np.quantile(K_flat, high_q)
    K_pool = K_flat[(K_flat >= k_low) & (K_flat <= k_high)]

    k_init_np = _sample_k_pool(
        K_pool, n_basis, problem_cfg.get("seed", 1999),
    ).astype(np_real_dtype)

    return jnp.asarray(k_init_np, dtype=real_dtype)


def _gi_rhs_outputs(GI_rhs_ri):
    GI_rhs_c = (GI_rhs_ri[:, 0] + 1j * GI_rhs_ri[:, 1]).reshape(-1, 1)

    GI_rhs_real_stacked = jnp.concatenate(
        [GI_rhs_ri[:, 0], GI_rhs_ri[:, 1]], axis=0,
    ).reshape(-1, 1)

    return GI_rhs_c, GI_rhs_real_stacked


def _radial_distance(coords, source):
    """Euclidean distance from each row of coords ([N,D]) to source ([D])."""
    coords = np.asarray(coords, dtype=np.float64)
    source = np.asarray([float(s) for s in source], dtype=np.float64)
    return np.linalg.norm(coords - source[None, :], axis=1)


def _griddata_scatter_to_grid(source_coords, source_values, target_coords):
    """Linear-interpolate scattered samples onto target_coords, filling
    out-of-hull points via nearest-neighbor."""
    import scipy.interpolate

    values = scipy.interpolate.griddata(
        source_coords, source_values, target_coords, method="linear",
    )
    missing = np.isnan(values)
    if np.any(missing):
        values[missing] = scipy.interpolate.griddata(
            source_coords, source_values, target_coords[missing], method="nearest",
        )
    return values


# ============================================================
# Problem state
# ============================================================

@dataclass
class ProblemState:
    real_dtype: object
    complex_dtype: object

    coords_data: jnp.ndarray
    v_data: jnp.ndarray
    u_real: jnp.ndarray
    u_imag: jnp.ndarray

    coords_gi: jnp.ndarray
    dm_gi: jnp.ndarray
    U0_gi: jnp.ndarray
    U0_data: jnp.ndarray

    U0_gi_c: jnp.ndarray

    G0_fft: jnp.ndarray
    GI_scale_c: jnp.ndarray

    GI_rhs_ri: jnp.ndarray
    GI_rhs_c: jnp.ndarray
    GI_rhs_real_stacked: jnp.ndarray

    nx_gi: int
    nz_gi: int
    cell_area: float

    omega: float
    v0: float
    v0_nd: float
    L: float

    nx_data: int
    nz_data: int

    input_dim: int = 2
    n_data: int = 0
    n_gi: int = 0
    w_damp: tuple = (0, 0)
    k_init: jnp.ndarray | None = None
    ndim: int = 2
    grid_shape: tuple | None = None

    # Only for 3D quadrature weights.
    cell_volume: float | None = None

    def lisch_operator_complex(self, Us_c, w_damp=None):
        grid_shape = self.grid_shape or (self.nz_gi, self.nx_gi)
        if w_damp is None:
            w_damp = (0,) * len(grid_shape)
        return green_apply_complex(
            U0_c=self.U0_gi_c,
            Us_c=Us_c,
            GI_scale_c=self.GI_scale_c,
            G0_kernel_fft=self.G0_fft,
            grid_shape=grid_shape,
            w_damp=w_damp,
        )


def resolve_reference_data_path(problem_cfg):
    """
    The path of the validation data if it exists.
    """
    data_dir = Path(_ensure_base_path(problem_cfg)) / "data"
    velocity_model = problem_cfg.get("velocity_model", problem_cfg.get("name", "Marmousi"))
    frequency = float(problem_cfg.get("frequency", 10.0))

    freq_path = data_dir / f"data_{velocity_model}_validation_{frequency:g}Hz.npz"
    if freq_path.exists():
        return freq_path
    return data_dir / f"data_{velocity_model}_validation.npz"


def _data_not_found_message(data_path):
    """
    "Data not found" error text that helps diagnose *why* -- not just the
    expected path (already derived, possibly from a bad config value), but
    what's actually sitting in the data folder so the user can compare.
    """
    data_dir = data_path.parent
    if not data_dir.exists():
        return (
            f"Data not found: {data_path}\n"
            f"The folder {data_dir} does not exist."
        )
    found = sorted(p.name for p in data_dir.glob("*.npz"))
    if not found:
        listing = "(no .npz files in that folder)"
    else:
        listing = "found instead: " + ", ".join(found)
    return f"Data not found: {data_path}\n{listing}"


def _gi_node_spacing(extent, nx, nz, w_damp):
    """
    Physical GI-grid cell size (dx_phys, dz_phys) and interior point counts.
    """
    z_damp, x_damp = int(w_damp[0]), int(w_damp[1])
    nx_inner = nx - 2 * x_damp
    nz_inner = nz - 2 * z_damp
    if nx_inner <= 0 or nz_inner <= 0:
        raise ValueError(
            f"Invalid GI grid: nx={nx}, nz={nz}, w_damp={w_damp}."
        )

    x_min, x_max, z_min, z_max = [float(v) for v in extent]
    dx_phys = (x_max - x_min) / nx_inner
    dz_phys = (z_max - z_min) / nz_inner
    return dx_phys, dz_phys, nx_inner, nz_inner


def _gi_grid_coords(extent, nx, nz, w_damp):
    """Physical node positions (x_phys, z_phys) for a GI grid of size
    (nx, nz)."""
    dx_phys, dz_phys, _, _ = _gi_node_spacing(extent, nx, nz, w_damp)
    x_min, x_max, z_min, z_max = [float(v) for v in extent]
    center_x = 0.5 * (x_min + x_max)
    center_z = 0.5 * (z_min + z_max)
    x_phys = center_x + (np.arange(nx) - (nx - 1) / 2.0) * dx_phys
    z_phys = center_z + (np.arange(nz) - (nz - 1) / 2.0) * dz_phys
    return x_phys, z_phys


def _resample_contrast(contrast_native, extent, nx_native, nz_native, w_damp_native, nx_req, nz_req, w_damp_req):
    """
    Resample a precomputed contrast array from its native GI-grid resolution
    onto a different requested resolution.
    """
    import scipy.interpolate

    x_native, z_native = _gi_grid_coords(extent, nx_native, nz_native, w_damp_native)
    X_native, Z_native = np.meshgrid(x_native, z_native, indexing="xy")
    xz_native = np.column_stack([X_native.ravel(), Z_native.ravel()])
    contrast_flat = np.asarray(contrast_native, dtype=np.float64).reshape(-1)

    x_req, z_req = _gi_grid_coords(extent, nx_req, nz_req, w_damp_req)
    X_req, Z_req = np.meshgrid(x_req, z_req, indexing="xy")
    xz_req = np.column_stack([X_req.ravel(), Z_req.ravel()])

    dm_flat = scipy.interpolate.griddata(
        xz_native, contrast_flat, xz_req, method="linear",
    )
    missing = np.isnan(dm_flat)
    if np.any(missing):
        dm_flat[missing] = scipy.interpolate.griddata(
            xz_native, contrast_flat, xz_req[missing], method="nearest",
        )
    return dm_flat.reshape(nz_req, nx_req)


def _build_gi_grid_internal(
    source_xz,
    source_v,
    extent,
    v0,
    omega,
    nx,
    nz,
    w_damp,
    np_real_dtype,
):
    """
    Build the Green-integral (Lippmann-Schwinger) discretization in-memory.
    """
    import scipy.interpolate

    z_damp, x_damp = int(w_damp[0]), int(w_damp[1])
    dx_phys, dz_phys, nx_inner, nz_inner = _gi_node_spacing(extent, nx, nz, w_damp)

    # Node placement: extent / n_inner, centered on the domain midpoint.
    x_min, x_max, z_min, z_max = [float(v) for v in extent]

    center_x = 0.5 * (x_min + x_max)
    center_z = 0.5 * (z_min + z_max)

    x_phys = center_x + (np.arange(nx) - (nx - 1) / 2.0) * dx_phys
    z_phys = center_z + (np.arange(nz) - (nz - 1) / 2.0) * dz_phys

    X, Z = np.meshgrid(x_phys, z_phys, indexing="xy")
    xz_target = np.column_stack([X.ravel(), Z.ravel()])

    v_flat = scipy.interpolate.griddata(
        source_xz, source_v, xz_target, method="linear",
    )
    missing = np.isnan(v_flat)
    if np.any(missing):
        v_flat[missing] = scipy.interpolate.griddata(
            source_xz, source_v, xz_target[missing], method="nearest",
        )
    v_2d = v_flat.reshape(nz, nx)

    L = 2.0 * np.pi * float(v0) / float(omega)
    v_nd = v_2d / L
    v0_nd = float(v0) / L
    dm_2d = 1.0 / v_nd ** 2 - 1.0 / v0_nd ** 2

    # Absorbing-border taper: ramp dm to zero across w_damp with a separable Hann window.
    def _hann_ramp(n_damp):
        # profile(i) = 0.5*(1 - cos(pi*i/n_damp)), i = 0 (outer edge) .. n_damp-1
        i = np.arange(n_damp)
        return 0.5 * (1.0 - np.cos(np.pi * i / n_damp))

    win_x = np.ones(nx)
    if x_damp > 0:
        ramp_x = _hann_ramp(x_damp)
        win_x[:x_damp] = ramp_x
        win_x[nx - x_damp:] = ramp_x[::-1]

    win_z = np.ones(nz)
    if z_damp > 0:
        ramp_z = _hann_ramp(z_damp)
        win_z[:z_damp] = ramp_z
        win_z[nz - z_damp:] = ramp_z[::-1]

    taper = np.outer(win_z, win_x)  # separable 2D window, shape (nz, nx)
    dm_2d = dm_2d * taper

    dm = dm_2d.reshape(-1)

    return (
        dm.astype(np_real_dtype),
        float(dx_phys / L),
        float(dz_phys / L),
        int(nx),
        int(nz),
        (z_damp, x_damp),
    )


# ============================================================
# Build Helmholtz problem
# ============================================================

def build_gi_problem(config):
    """Build the 2D Helmholtz scattering problem from a stored
    data/data_<name>_validation[_<freq>Hz].npz (xz_ref/v_ref/U_ref, optionally
    contrast/gi_damp)."""

    validate_config(config)

    problem_cfg = config.get("problem", {})
    verbosity = int(problem_cfg.get("verbosity", 1))

    _ensure_base_path(problem_cfg)  # fills in problem_cfg['base_path'] if unset
    velocity_model = problem_cfg.get(
        "velocity_model",
        problem_cfg.get("name", "Marmousi"),
    )

    precision = problem_cfg.get("precision", "float64")
    configure_precision(precision)

    real_dtype, complex_dtype, np_real_dtype, np_complex_dtype = get_dtypes(
        precision,
    )

    frequency = float(problem_cfg.get("frequency", 10.0))
    omega = float(2.0 * np.pi * frequency)

    factor = float(problem_cfg.get("factor", 1.0))

    # Domain extent [x_min, x_max, z_min, z_max]; presence/shape already
    # checked by validate_config above.
    extent = [float(v) for v in problem_cfg["domain"]]

    # Background velocity (problem.v0); presence/sign already checked by
    # validate_config above.
    v0 = float(problem_cfg["v0"])

    # Source location (sx, sz); presence/length already checked by
    # validate_config above.
    s_xz = np.asarray(
        [float(s) for s in problem_cfg["source"]],
        dtype=np_real_dtype,
    )

    # ------------------------------------------------------------
    # Reference field: always a stored validation .npz.
    # ------------------------------------------------------------
    data_path = resolve_reference_data_path(problem_cfg)
    used_frequency_match = f"_{frequency:g}Hz.npz" in data_path.name

    if not data_path.exists():
        raise FileNotFoundError(_data_not_found_message(data_path))

    if verbosity >= 1:
        tag = "frequency-matched" if used_frequency_match else "default"
        print(f"  reference data ({tag}) : {data_path}")

    data = np.load(data_path)

    v_ref = data["v_ref"].astype(np_real_dtype)
    xz_ref = data["xz_ref"].astype(np_real_dtype)

    # U_ref (the reference/exact scattered wavefield) is only used for the
    # rel_L2 validation metric.
    if "U_ref" in data.files:
        U_ref = data["U_ref"].astype(np_real_dtype)
    else:
        print(
            f"  WARNING: reference wavefield 'U_ref' not found under "
            f"{data_path.name} -- continuing without it (no rel_L2 metric "
            "will be available; GI training doesn't require labeled data)."
        )
        U_ref = np.full((xz_ref.shape[0], 2), np.nan, dtype=np_real_dtype)

    # Raw velocity samples kept for optional internal GI-grid generation.
    gi_source_xz = np.asarray(data["xz_ref"], dtype=np.float64)
    gi_source_v = np.asarray(data["v_ref"], dtype=np.float64).reshape(-1)

    # Validation grid size is recovered from the (regular) xz_ref coordinates
    nx_data = int(np.unique(xz_ref[:, 0]).size)
    nz_data = int(np.unique(xz_ref[:, 1]).size)
    if nx_data * nz_data != xz_ref.shape[0]:
        raise ValueError(
            f"Validation grid is not a regular {nx_data}x{nz_data} grid: "
            f"nx_val*nz_val={nx_data * nz_data} != {xz_ref.shape[0]} points."
        )

    x_1d = xz_ref[:, 0].astype(np_real_dtype)
    z_1d = xz_ref[:, 1].astype(np_real_dtype)
    v_1d = v_ref.reshape(-1).astype(np_real_dtype)

    u_real_1d = U_ref[:, 0].astype(np_real_dtype)
    u_imag_1d = U_ref[:, 1].astype(np_real_dtype)

    # ------------------------------------------------------------
    # Nondimensionalization
    # ------------------------------------------------------------
    L = 2.0 * np.pi * v0 / omega

    x_nd = x_1d / L
    z_nd = z_1d / L

    sx_nd = float(s_xz[0]) / L
    sz_nd = float(s_xz[1]) / L

    x_min, x_max, z_min, z_max = extent

    shift_x = 0.5 * (x_min + x_max) / L
    shift_z = 0.5 * (z_min + z_max) / L

    x_nd = x_nd - shift_x
    z_nd = z_nd - shift_z

    sx_nd = sx_nd - shift_x
    sz_nd = sz_nd - shift_z

    coords_data_np = np.stack([x_nd, z_nd], axis=1).astype(np_real_dtype)

    coords_data = jnp.asarray(coords_data_np, dtype=real_dtype)
    v_data = jnp.asarray(v_1d, dtype=real_dtype).reshape(-1, 1)

    u_real = jnp.asarray(u_real_1d, dtype=real_dtype).reshape(-1, 1)
    u_imag = jnp.asarray(u_imag_1d, dtype=real_dtype).reshape(-1, 1)

    source_nd = jnp.asarray([sx_nd, sz_nd], dtype=real_dtype)

    v0_nd = v0 / L

    U0_data = compute_u0(
        coords=coords_data,
        source=source_nd,
        v0=v0_nd,
        omega=omega,
        factor=factor,
        real_dtype=real_dtype,
    )

    # delta_m: stored contrast if it matches resolution, else resampled,
    # else built on the fly.
    nx_req = int(problem_cfg["gi_nx"])
    nz_req = int(problem_cfg["gi_nz"])
    w_damp_req = tuple(int(x) for x in problem_cfg.get("gi_damp", (10, 10)))

    if "contrast" in data.files and "gi_damp" in data.files:
        contrast_native = data["contrast"]
        nz_native, nx_native = contrast_native.shape
        w_damp_native = tuple(int(x) for x in data["gi_damp"])

        if (nx_native, nz_native, w_damp_native) == (nx_req, nz_req, w_damp_req):
            dm_gi_2d = contrast_native.astype(np.float64)
            if verbosity >= 1:
                print(
                    f"GI grid: using stored contrast verbatim "
                    f"({nx_req}x{nz_req}, w_damp={w_damp_req})"
                )
        else:
            dm_gi_2d = _resample_contrast(
                contrast_native, extent,
                nx_native, nz_native, w_damp_native,
                nx_req, nz_req, w_damp_req,
            )
            if verbosity >= 1:
                print(
                    f"GI grid: resampled stored contrast from "
                    f"{nx_native}x{nz_native} (w_damp={w_damp_native}) to "
                    f"{nx_req}x{nz_req} (w_damp={w_damp_req})"
                )

        dm_gi_np = dm_gi_2d.astype(np_real_dtype).reshape(-1)
        dx_phys, dz_phys, _, _ = _gi_node_spacing(extent, nx_req, nz_req, w_damp_req)
        L = 2.0 * np.pi * float(v0) / float(omega)
        dx_gi, dz_gi = float(dx_phys / L), float(dz_phys / L)
        nx_gi, nz_gi, w_damp = nx_req, nz_req, w_damp_req
    else:
        if verbosity >= 1:
            print(
                f"GI grid: no stored contrast in {data_path} -- "
                "building delta_m on the fly."
            )

        (
            dm_gi_np,
            dx_gi,
            dz_gi,
            nx_gi,
            nz_gi,
            w_damp,
        ) = _build_gi_grid_internal(
            source_xz=gi_source_xz,
            source_v=gi_source_v,
            extent=extent,
            v0=v0,
            omega=omega,
            nx=nx_req,
            nz=nz_req,
            w_damp=w_damp_req,
            np_real_dtype=np_real_dtype,
        )

    # ------------------------------------------------------------
    # Centered GI coordinates
    # ------------------------------------------------------------
    x_gi = (
        np.arange(nx_gi, dtype=np_real_dtype)
        - (nx_gi - 1) / 2.0
    ) * dx_gi

    z_gi = (
        np.arange(nz_gi, dtype=np_real_dtype)
        - (nz_gi - 1) / 2.0
    ) * dz_gi

    Z_gi, X_gi = np.meshgrid(z_gi, x_gi, indexing="ij")

    coords_gi_np = np.stack(
        [
            X_gi.reshape(-1),
            Z_gi.reshape(-1),
        ],
        axis=1,
    ).astype(np_real_dtype)

    coords_gi = jnp.asarray(coords_gi_np, dtype=real_dtype)
    dm_gi = jnp.asarray(dm_gi_np, dtype=real_dtype).reshape(-1, 1)

    cell_area = dx_gi * dz_gi

    # ------------------------------------------------------------
    # U0 on the GI grid: analytic Hankel field, U0 = (i/4) H_0^(2)(k0 r).
    U0_gi = compute_u0(
        coords=coords_gi,
        source=source_nd,
        v0=v0_nd,
        omega=omega,
        factor=factor,
        real_dtype=real_dtype,
    )

    U0_gi_c = to_complex_grid(U0_gi, (nz_gi, nx_gi)).astype(complex_dtype)

    # FFT of the free-space Green's function g_0 on the padded grid,
    k0_dimless = omega / v0_nd

    G0_fft = build_green_kernel_fft_2d(
        nx=nx_gi,
        nz=nz_gi,
        dx=dx_gi,
        dz=dz_gi,
        k0=k0_dimless,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )

    # ------------------------------------------------------------
    # GI source scaling: GI_scale_c = -omega^2 W delta_m
    GI_scale_c = (
        -omega**2
        * cell_area
        * dm_gi.reshape(nz_gi, nx_gi)
    ).astype(complex_dtype)

    # Source vector f = K[U0] = G_0 * (-omega^2 W delta_m U0)
    zero_Us_gi = jnp.zeros_like(U0_gi)

    GI_rhs_ri = green_apply_2d(
        dm=dm_gi,
        U0=U0_gi,
        Us=zero_Us_gi,
        omega=omega,
        W=cell_area,
        nz=nz_gi,
        nx=nx_gi,
        G0_kernel_fft=G0_fft,
        GI_scale_c=GI_scale_c,
        w_damp=(0, 0),
    )

    GI_rhs_c, GI_rhs_real_stacked = _gi_rhs_outputs(GI_rhs_ri)

    # ------------------------------------------------------------
    # k_init for plane-wave architecture
    # ------------------------------------------------------------
    arch_cfg = config.get("arch", {})

    K_phys_2d = omega / v_1d.reshape(nz_data, nx_data)
    K_nd_2d = K_phys_2d * L
    K_flat = K_nd_2d.flatten().astype(np_real_dtype)

    k_init = _sample_k_init(K_flat, arch_cfg, problem_cfg, np_real_dtype, real_dtype)

    # ------------------------------------------------------------
    # Return problem
    # ------------------------------------------------------------
    problem = ProblemState(
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,

        coords_data=coords_data,
        v_data=v_data,
        u_real=u_real,
        u_imag=u_imag,

        coords_gi=coords_gi,
        dm_gi=dm_gi,
        U0_gi=U0_gi,
        U0_data=U0_data,
        U0_gi_c=U0_gi_c,

        G0_fft=G0_fft,
        GI_scale_c=GI_scale_c,

        GI_rhs_ri=GI_rhs_ri,
        GI_rhs_c=GI_rhs_c,
        GI_rhs_real_stacked=GI_rhs_real_stacked,

        nx_gi=nx_gi,
        nz_gi=nz_gi,
        cell_area=cell_area,

        ndim=2,
        grid_shape=(nz_gi, nx_gi),

        omega=omega,
        v0=v0,
        v0_nd=v0_nd,
        L=L,

        nx_data=nx_data,
        nz_data=nz_data,

        input_dim=2,
        n_data=coords_data.shape[0],
        n_gi=coords_gi.shape[0],
        w_damp=w_damp,
        k_init=k_init,
    )


    if verbosity >= 1:
        title = f"========== General info: {velocity_model} (2D) =========="
        print(f"\n{title}")
        print("Physics")
        print(f"  frequency            : {frequency:g} Hz")
        print(f"  background velocity  : v0 = {v0:g}")
        print("Grids")
        print(f"  validation grid      : {nx_data} x {nz_data}   ({problem.n_data} points)")
        print(f"  Green-integral grid  : {nx_gi} x {nz_gi}   ({problem.n_gi} points)")
        print("Solver")
        print(f"  precision            : {precision}")
        print("=" * len(title))

    return problem


# ============================================================
# 3D problem-specific physics
# ============================================================

def _hann_taper_ramp(n_damp):
    """Hann ramp profile(i) = 0.5*(1-cos(pi*i/n_damp)), i=0 (outer edge)
    .. n_damp-1. Shared by every absorbing-border taper (2D and 3D)."""
    i = np.arange(n_damp)
    return 0.5 * (1.0 - np.cos(np.pi * i / n_damp))


def _separable_taper_1d(n, w):
    """1D taper window of length n, ramping to zero across a border of
    width w at each end. Shared by every absorbing-border taper (2D and 3D)."""
    win = np.ones(n)
    if w > 0:
        ramp = _hann_taper_ramp(w)
        win[:w] = ramp
        win[n - w:] = ramp[::-1]
    return win


def build_gi_grid_coords_3d(extent, nx_gi, ny_gi, nz_gi, x_damp, y_damp, z_damp):
    """Physical node positions of the 3D GI grid as a flat [N,3] (x,y,z)
    array. Returns (xyz, dx, dy, dz)."""
    x0, x1, y0, y1, z0, z1 = [float(v) for v in extent]
    nx_inner = nx_gi - 2 * int(x_damp)
    ny_inner = ny_gi - 2 * int(y_damp)
    nz_inner = nz_gi - 2 * int(z_damp)
    if nx_inner <= 0 or ny_inner <= 0 or nz_inner <= 0:
        raise ValueError(
            f"nx_gi/ny_gi/nz_gi=({nx_gi},{ny_gi},{nz_gi}) too small for "
            f"damp=({z_damp},{y_damp},{x_damp}) on each side."
        )

    dx = (x1 - x0) / nx_inner
    dy = (y1 - y0) / ny_inner
    dz = (z1 - z0) / nz_inner

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    cz = 0.5 * (z0 + z1)
    x_lin = cx + (np.arange(nx_gi) - (nx_gi - 1) / 2.0) * dx
    y_lin = cy + (np.arange(ny_gi) - (ny_gi - 1) / 2.0) * dy
    z_lin = cz + (np.arange(nz_gi) - (nz_gi - 1) / 2.0) * dz

    Z, Y, X = np.meshgrid(z_lin, y_lin, x_lin, indexing="ij")
    xyz = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)

    return xyz, float(dx), float(dy), float(dz)


def build_gi_problem_3d(config):
    """Build the 3D ProblemState from a stored
    data/data_<name>_validation[_<freq>Hz].npz (xyz_ref/v_ref/U_ref)."""
    validate_config(config)

    problem_cfg = config.get("problem", {})
    arch_cfg = config.get("arch", {})

    _ensure_base_path(problem_cfg)  # fills in problem_cfg['base_path'] if unset

    precision = problem_cfg.get("precision", "float64")
    real_dtype, complex_dtype, np_real_dtype, np_complex_dtype = get_dtypes(precision)

    verbosity = int(problem_cfg.get("verbosity", 1))

    # frequency/v0/domain/source presence, sign, and shape (domain has 6
    # entries, source has 3) already checked by validate_config above.
    frequency = float(problem_cfg["frequency"])
    omega = 2.0 * np.pi * frequency

    # "c0" is accepted as an alias for v0.
    v0 = float(problem_cfg.get("v0", problem_cfg.get("c0")))

    source = tuple(float(s) for s in problem_cfg["source"])

    extent = tuple(float(v) for v in problem_cfg["domain"])

    # GI grid resolution: gi_nx/gi_ny/gi_nz (presence already checked by
    # validate_config above).
    nx_gi = int(problem_cfg["gi_nx"])
    ny_gi = int(problem_cfg["gi_ny"])
    nz_gi = int(problem_cfg["gi_nz"])

    if problem_cfg.get("gi_damp") is not None:
        z_damp, y_damp, x_damp = (int(v) for v in problem_cfg["gi_damp"])
    else:
        w_damp_cells = int(problem_cfg.get("w_damp", 20))
        z_damp = y_damp = x_damp = w_damp_cells

    nx_inner = nx_gi - 2 * x_damp
    ny_inner = ny_gi - 2 * y_damp
    nz_inner = nz_gi - 2 * z_damp
    if nx_inner <= 0 or ny_inner <= 0 or nz_inner <= 0:
        raise ValueError(
            f"problem.gi_nx/gi_ny/gi_nz=({nx_gi},{ny_gi},{nz_gi}) too small "
            f"for problem.gi_damp=({z_damp},{y_damp},{x_damp}) on each side."
        )

    xyz, dx, dy, dz = build_gi_grid_coords_3d(
        extent, nx_gi, ny_gi, nz_gi, x_damp, y_damp, z_damp,
    )

    data_path = resolve_reference_data_path(problem_cfg)
    if not data_path.exists():
        raise FileNotFoundError(_data_not_found_message(data_path))
    if verbosity >= 1:
        print(f"  reference data (stored) : {data_path}")

    data = np.load(data_path)
    xyz_ref = np.asarray(data["xyz_ref"], dtype=np.float64)
    v_ref = np.asarray(data["v_ref"], dtype=np.float64).reshape(-1)
    if "U_ref" in data.files:
        U_ref = np.asarray(data["U_ref"], dtype=np.float64)
    else:
        print(
            f"  WARNING: reference wavefield 'U_ref' not found under "
            f"{data_path.name} -- continuing without it (no rel_L2 metric "
            "will be available; GI training doesn't require labeled data)."
        )
        U_ref = np.full((xyz_ref.shape[0], 2), np.nan, dtype=np.float64)

    grid_matches = (
        xyz_ref.shape == xyz.shape
        and np.allclose(xyz_ref, xyz, rtol=0.0, atol=1e-8)
    )
    if grid_matches:
        if verbosity >= 1:
            print(
                f"GI grid: xyz_ref matches the requested "
                f"{nx_gi}x{ny_gi}x{nz_gi} grid exactly -- using stored "
                "v_ref/U_ref verbatim (no interpolation)."
            )
        c_field_flat = v_ref
    else:
        if verbosity >= 1:
            print(
                f"GI grid: xyz_ref does not match the requested "
                f"{nx_gi}x{ny_gi}x{nz_gi} grid -- interpolating "
                "(scipy.interpolate.griddata; may be slow at high "
                "resolution)."
            )
        c_field_flat = _griddata_scatter_to_grid(xyz_ref, v_ref, xyz)

    c_field = c_field_flat.reshape(nz_gi, ny_gi, nx_gi)
    dm = (1.0 / c_field ** 2 - 1.0 / v0 ** 2)

    win_x = _separable_taper_1d(nx_gi, x_damp)
    win_y = _separable_taper_1d(ny_gi, y_damp)
    win_z = _separable_taper_1d(nz_gi, z_damp)
    taper_3d = win_z[:, None, None] * win_y[None, :, None] * win_x[None, None, :]
    dm = dm * taper_3d

    interior = np.ones((nz_gi, ny_gi, nx_gi), dtype=bool)
    if z_damp > 0:
        interior[:z_damp] = False
        interior[nz_gi - z_damp:] = False
    if y_damp > 0:
        interior[:, :y_damp] = False
        interior[:, ny_gi - y_damp:] = False
    if x_damp > 0:
        interior[:, :, :x_damp] = False
        interior[:, :, nx_gi - x_damp:] = False

    cell_volume = dx * dy * dz
    k0 = omega / v0

    grid_shape = (nz_gi, ny_gi, nx_gi)
    coords_gi = jnp.asarray(xyz, dtype=real_dtype)
    dm_gi = jnp.asarray(dm.reshape(-1, 1), dtype=real_dtype)

    # ------------------------------------------------------------
    # U0 on the GI grid, [N,2] real/imag layout (same convention as 2D).
    source_j = jnp.asarray(source, dtype=real_dtype)
    U0_gi = compute_u0(
        coords=coords_gi, source=source_j, v0=v0, omega=omega, real_dtype=real_dtype,
    )
    U0_gi_c = to_complex_grid(U0_gi, grid_shape).astype(complex_dtype)

    # ------------------------------------------------------------
    # F[g_0] on the zero-padded grid (3D).
    self_term = compute_G0_regularized(dx, dz, k0, dy=dy)
    G0_fft, padded_shape = build_green_kernel_fft(
        grid_shape=grid_shape,
        spacing=(dz, dy, dx),
        background_fn=green_3d_background_fn(k0),
        self_term_value=self_term,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )

    # ------------------------------------------------------------
    # GI_scale_c uses "+" here (NOT 2D's "-" convention).
    GI_scale_c = (
        omega ** 2 * cell_volume * dm_gi.reshape(grid_shape)
    ).astype(complex_dtype)

    # Source vector f = K[U0], same two layouts as 2D.
    zero_Us_gi = jnp.zeros_like(U0_gi)
    GI_rhs_ri = green_apply(
        U0=U0_gi,
        Us=zero_Us_gi,
        GI_scale_c=GI_scale_c,
        G0_kernel_fft=G0_fft,
        grid_shape=grid_shape,
        w_damp=(0, 0, 0),
    )

    GI_rhs_c, GI_rhs_real_stacked = _gi_rhs_outputs(GI_rhs_ri)

    # rel_L2 metric points: exclude a fixed radius around the source
    # (avoids the source singularity).
    r_exclude = 3 * min(dx, dy, dz)

    r_flat = _radial_distance(xyz, source)
    metric_mask = interior.reshape(-1) & (r_flat > r_exclude)

    if grid_matches:
        u_real_masked = U_ref[metric_mask, 0]
        u_imag_masked = U_ref[metric_mask, 1]
    else:
        u_real_masked = _griddata_scatter_to_grid(xyz_ref, U_ref[:, 0], xyz[metric_mask])
        u_imag_masked = _griddata_scatter_to_grid(xyz_ref, U_ref[:, 1], xyz[metric_mask])

    coords_data = jnp.asarray(xyz[metric_mask], dtype=real_dtype)
    v_data = jnp.asarray(
        c_field_flat[metric_mask], dtype=real_dtype,
    ).reshape(-1, 1)
    u_real = jnp.asarray(u_real_masked, dtype=real_dtype).reshape(-1, 1)
    u_imag = jnp.asarray(u_imag_masked, dtype=real_dtype).reshape(-1, 1)

    U0_data = compute_u0(
        coords=coords_data, source=source_j, v0=v0, omega=omega, real_dtype=real_dtype,
    )

    # ------------------------------------------------------------
    # k_init: omega/c(x) over the GI grid, quantile-filtered, then sampled.
    K_flat = (omega / c_field_flat).astype(np_real_dtype)

    k_init = _sample_k_init(K_flat, arch_cfg, problem_cfg, np_real_dtype, real_dtype)

    # ------------------------------------------------------------
    # Return problem
    # ------------------------------------------------------------
    problem = ProblemState(
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,

        coords_data=coords_data,
        v_data=v_data,
        u_real=u_real,
        u_imag=u_imag,

        coords_gi=coords_gi,
        dm_gi=dm_gi,
        U0_gi=U0_gi,
        U0_data=U0_data,
        U0_gi_c=U0_gi_c,

        G0_fft=G0_fft,
        GI_scale_c=GI_scale_c,

        GI_rhs_ri=GI_rhs_ri,
        GI_rhs_c=GI_rhs_c,
        GI_rhs_real_stacked=GI_rhs_real_stacked,

        nx_gi=nx_gi,
        nz_gi=nz_gi,
        cell_area=cell_volume,
        cell_volume=cell_volume,

        omega=omega,
        v0=v0,
        v0_nd=v0,
        L=1.0,

        nx_data=None,
        nz_data=None,

        input_dim=3,
        n_data=int(coords_data.shape[0]),
        n_gi=int(coords_gi.shape[0]),
        w_damp=(z_damp, y_damp, x_damp),
        k_init=k_init,

        ndim=3,
        grid_shape=grid_shape,
    )

    if verbosity >= 1:
        problem_name = problem_cfg.get("name", "problem")
        title = f"========== General info: {problem_name} (3D) =========="
        print(f"\n{title}")
        print("Physics")
        print(f"  frequency            : {frequency:g} Hz")
        print(f"  background velocity  : v0 = {v0:g}")
        print("Grids")
        print(f"  Green-integral grid  : {nz_gi} x {ny_gi} x {nx_gi}   ({problem.n_gi} points)")
        print(f"  metric (interior)    : {problem.n_data} points")
        print("Solver")
        print(f"  precision            : {precision}")
        print("=" * len(title))

    return problem