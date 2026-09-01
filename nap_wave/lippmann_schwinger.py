"""
Free-space Green's function g0 (outgoing Hankel H_0^(2) in 2D,
exp(-i k0 r)/(4 pi r) in 3D) and the FFT-accelerated Green integral operator.
"""

import numpy as np, scipy.special as sp, jax.numpy as jnp
from scipy.fft import next_fast_len


# ============================================================
# Background free-space field (2D and 3D)
# ============================================================

def compute_u0(
    coords,
    source,
    v0,
    omega,
    factor=1.0,
    real_dtype=jnp.float32,
):
    """
    Background free-space Helmholtz field U0, dispatched on dim = len(source):
        2D: U0(x) = factor * (i/4) H_0^{(2)}(k0 |x - xs|).
        3D: U0(x) = factor * exp(-i k0 |x - xs|) / (4 pi |x - xs|).
    coords: [N, dim], source: [dim]. Returns [N, 2] = [Re(U0), Im(U0)].
    """

    coords_np = np.asarray(coords)
    source_np = np.asarray(source, dtype=np.float64)
    dim = source_np.shape[0]

    diff = coords_np[:, :dim] - source_np[None, :dim]
    r = np.sqrt(np.sum(diff ** 2, axis=1, keepdims=True))
    r_safe = np.maximum(r, 1e-9)
    k0 = omega / v0

    if dim == 2:
        U0_np = factor * (1j / 4.0) * sp.hankel2(0, k0 * r_safe)
    elif dim == 3:
        U0_np = factor * np.exp(-1j * k0 * r_safe) / (4.0 * np.pi * r_safe)
    else:
        raise ValueError(f"compute_u0 supports source dim 2 or 3, got {dim}.")

    U0 = np.concatenate(
        [
            np.real(U0_np),
            np.imag(U0_np),
        ],
        axis=1,
    )

    return jnp.asarray(U0, dtype=real_dtype)



def compute_G0_regularized(dx, dz, k0, dy=None):
    """
    Regularized self-interaction value g0(0), dispatched on dim:
        2D (dy=None): circular equivalent-area approximation for the
            singular self-term of cell (dx, dz).
        3D (dy given): equivalent-volume-ball average for cell (dx, dy, dz).
    """

    if dy is None:
        h = np.sqrt(dx * dz / np.pi)
        log_singular_average = np.log(h) - 0.5

        real_part = (
            (1.0 / (2.0 * np.pi))
            * (
                log_singular_average
                + np.log(k0 / 2.0)
                + np.euler_gamma
            )
        )

        return real_part + 1j / 4.0

    V = dx * dy * dz
    a_eq = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)
    real_part = 3.0 / (8.0 * np.pi * a_eq)
    imag_part = -k0 / (4.0 * np.pi)
    return complex(real_part, imag_part)


def green_3d_background_fn(k0):
    """Vectorized outgoing 3D free-space Green's function g0(R) = exp(-i k0 R)/(4 pi R)."""
    def _fn(R):
        return np.exp(-1j * k0 * R) / (4.0 * np.pi * R)
    return _fn
# ============================================================
# Green kernel FFT
# ============================================================

def build_green_kernel_fft_2d(
    nx,
    nz,
    dx,
    dz,
    k0,
    real_dtype=jnp.float32,
    complex_dtype=jnp.complex64,
):
    """
    FFT of the 2D free-space Green kernel on a padded grid.
    """

    Nz, Nx = choose_pad_shape((nz, nx))

    z_k = (np.arange(-Nz // 2, Nz - Nz // 2)) * dz
    x_k = (np.arange(-Nx // 2, Nx - Nx // 2)) * dx

    Z_k, X_k = np.meshgrid(z_k, x_k, indexing="ij")
    R_k = np.sqrt(X_k**2 + Z_k**2)

    G0_np = (1j / 4.0) * sp.hankel2(
        0,
        k0 * np.maximum(R_k, 1e-12),
    )

    G_00 = compute_G0_regularized(
        dx=dx,
        dz=dz,
        k0=k0,
    )
    G0_np[R_k == 0.0] = G_00

    G0_kernel = jnp.asarray(G0_np, dtype=complex_dtype)
    G0_kernel_shift = jnp.fft.ifftshift(G0_kernel)
    G0_kernel_fft = jnp.fft.fft2(G0_kernel_shift)

    return G0_kernel_fft


# ============================================================
# Lippmann-Schwinger operator
# ============================================================

def green_apply_2d(
    dm,
    U0,
    Us,
    omega,
    W,
    nz,
    nx,
    G0_kernel_fft,
    GI_scale_c=None,
    w_damp=(0, 0),
):
    """
    Apply the Lippmann-Schwinger Green integral:
        Uhat_s = G0 * [-omega^2 dm (U0 + Us)] W.
    dm: [N,1]/[N] contrast; U0, Us: [N,2] real/imag. Returns [N,2].
    """

    complex_dtype = G0_kernel_fft.dtype

    if GI_scale_c is None:
        dm_c = dm.reshape(nz, nx).astype(complex_dtype)
        GI_scale_c = dm_c * (-omega**2 * W)
    else:
        GI_scale_c = GI_scale_c.astype(complex_dtype)

    return green_apply(
        U0=U0,
        Us=Us,
        GI_scale_c=GI_scale_c,
        G0_kernel_fft=G0_kernel_fft,
        grid_shape=(nz, nx),
        w_damp=w_damp,
    )


# ============================================================================
# N-D GENERIC Green-integral machinery (2D and 3D)
# ============================================================================

def choose_pad_shape(grid_shape):
    """One next_fast_len(2n-1) per axis."""
    return tuple(next_fast_len(2 * n - 1) for n in grid_shape)


def to_complex_grid(U_ri, grid_shape):
    """U_ri [N,2] (real, imag) -> complex array reshaped to grid_shape
    (any number of axes)."""
    Uc = U_ri[:, 0] + 1j * U_ri[:, 1]
    return Uc.reshape(grid_shape)


def from_complex_grid(Uc):
    """Complex grid of any shape -> flat [N,2] (real, imag)."""
    Uc_flat = Uc.reshape(-1)
    return jnp.stack([jnp.real(Uc_flat), jnp.imag(Uc_flat)], axis=-1)


def pad_complex_grid(f, padded_shape, orig_shape):
    """Center-pad a complex grid f from orig_shape to padded_shape. 
    Returns (f_pad, starts)."""
    pad_width = []
    starts = []
    for Ni, ni in zip(padded_shape, orig_shape):
        p0 = (Ni - ni) // 2
        p1 = Ni - ni - p0
        pad_width.append((p0, p1))
        starts.append(p0)
    f_pad = jnp.pad(f, pad_width=tuple(pad_width), mode="constant", constant_values=0)
    return f_pad, tuple(starts)


def pad_complex_batch(f, padded_shape, orig_shape):
    """Center-pad a batch f [B, *orig_shape] to [B, *padded_shape]."""
    pad_width = [(0, 0)]
    starts = []
    for Ni, ni in zip(padded_shape, orig_shape):
        p0 = (Ni - ni) // 2
        p1 = Ni - ni - p0
        pad_width.append((p0, p1))
        starts.append(p0)
    f_pad = jnp.pad(f, pad_width=tuple(pad_width), mode="constant", constant_values=0)
    return f_pad, tuple(starts)


def build_green_kernel_fft(
    grid_shape,
    spacing,
    background_fn,
    self_term_value,
    real_dtype=jnp.float64,
    complex_dtype=jnp.complex128,
):
    """
    Generic N-D free-space Green kernel FFT builder.
    """
    padded_shape = choose_pad_shape(grid_shape)

    axes_lin = [
        (np.arange(-(Ni // 2), Ni - Ni // 2)) * d
        for Ni, d in zip(padded_shape, spacing)
    ]
    mesh = np.meshgrid(*axes_lin, indexing="ij")
    R = np.sqrt(sum(m ** 2 for m in mesh))

    G0_np = np.asarray(background_fn(np.maximum(R, 1e-12)), dtype=np.complex128)
    G0_np[R == 0.0] = self_term_value

    G0_kernel = jnp.asarray(G0_np, dtype=complex_dtype)
    G0_kernel_shift = jnp.fft.ifftshift(G0_kernel)
    G0_kernel_fft = jnp.fft.fftn(G0_kernel_shift)

    return G0_kernel_fft, padded_shape


def green_apply(U0, Us, GI_scale_c, G0_kernel_fft, grid_shape, w_damp=None):
    """
    N-D generalization of green_apply_2d (real/imag [N,2] I/O).
    """
    ndim = len(grid_shape)
    if w_damp is None:
        w_damp = (0,) * ndim

    complex_dtype = G0_kernel_fft.dtype

    U0_c = to_complex_grid(U0, grid_shape).astype(complex_dtype)
    Us_c = to_complex_grid(Us, grid_shape).astype(complex_dtype)

    f = GI_scale_c.astype(complex_dtype) * (U0_c + Us_c)

    padded_shape = G0_kernel_fft.shape
    f_pad, starts = pad_complex_grid(f, padded_shape, grid_shape)

    axes = tuple(range(-ndim, 0))
    Fk = jnp.fft.fftn(f_pad, axes=axes)
    U_pad = jnp.fft.ifftn(G0_kernel_fft * Fk, axes=axes)

    slices = tuple(
        slice(s + w, s + n - w) for s, n, w in zip(starts, grid_shape, w_damp)
    )
    Uhat = U_pad[slices]

    return from_complex_grid(Uhat)


def green_apply_complex(U0_c, Us_c, GI_scale_c, G0_kernel_fft, grid_shape, w_damp=None):
    """Fast complex-valued Lippmann-Schwinger operator."""
    ndim = len(grid_shape)
    if w_damp is None:
        w_damp = (0,) * ndim

    f = GI_scale_c * (U0_c + Us_c)

    padded_shape = G0_kernel_fft.shape
    f_pad, starts = pad_complex_grid(f, padded_shape, grid_shape)

    axes = tuple(range(-ndim, 0))
    Fk = jnp.fft.fftn(f_pad, axes=axes)
    U_pad = jnp.fft.ifftn(G0_kernel_fft * Fk, axes=axes)

    slices = tuple(
        slice(s + w, s + n - w) for s, n, w in zip(starts, grid_shape, w_damp)
    )
    Uhat = U_pad[slices]

    return Uhat.reshape(-1, 1)


def green_apply_batch(
    Us_batch_c, GI_scale_c, G0_kernel_fft, grid_shape, w_damp=None,
):
    """
    Batched LS-column-assembly operator used by least_squares.py.
    Us_batch_c: [B, *grid_shape] complex. Returns [B, prod(grid_shape)]
    """
    if not jnp.iscomplexobj(Us_batch_c):
        raise TypeError("Us_batch_c must be a complex JAX array.")

    ndim = len(grid_shape)
    if w_damp is None:
        w_damp = (0,) * ndim

    complex_dtype = Us_batch_c.dtype

    F_theta = GI_scale_c.astype(complex_dtype)[None, ...] * Us_batch_c

    padded_shape = G0_kernel_fft.shape
    F_theta_pad, starts = pad_complex_batch(F_theta, padded_shape, grid_shape)

    axes = tuple(range(-ndim, 0))
    Fk = jnp.fft.fftn(F_theta_pad, axes=axes)
    G0_fft = G0_kernel_fft.astype(complex_dtype)
    Fk_times_G = Fk * G0_fft[None, ...]
    U_pad = jnp.fft.ifftn(Fk_times_G, axes=axes)

    slices = (slice(None),) + tuple(
        slice(s + w, s + n - w) for s, n, w in zip(starts, grid_shape, w_damp)
    )
    Uhat = U_pad[slices]

    return Uhat.reshape(Us_batch_c.shape[0], -1)






