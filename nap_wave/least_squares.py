"""
Variable-projection coefficient solve: for fixed theta, c is recovered by
the regularized normal equations (B^*B + mu I) c = B^*f, B = Phi - G[Phi].
"""

import jax, jax.numpy as jnp
from .lippmann_schwinger import (
    green_apply_batch,
)


# ============================================================
# LS regularization schedule
# ============================================================

def get_ls_reg(
    epoch: int,
    total_epochs: int,
    ls_reg_start: float = 1e-3,
    ls_reg_end: float = 1e-6,
):
    """
    Damping mu for the LS solve (B^*B + mu I)c = B^*f.
    mu decays exponentially from ls_reg_start to ls_reg_end.
    """

    t = min(1.0, max(0.0, epoch / max(1, total_epochs)))

    reg = ls_reg_start * (ls_reg_end / ls_reg_start) ** t

    return float(reg)


# ============================================================
# Update coefficient parameters in Flax model_vars
# ============================================================

def assign_coefficients_to_model_vars(model_vars, c_complex):
    """Return a new model_vars with updated coeff_real/coeff_imag."""

    params = dict(model_vars["params"])
    params["coeff_real"] = jnp.real(c_complex)
    params["coeff_imag"] = jnp.imag(c_complex)

    return {"params": params}


# ============================================================
# Fused, JIT-compiled LS core
# ============================================================

def make_jitted_ls_core(model, problem, column_batch):
    """Return a jit-compiled solve(model_vars, reg) -> c, fusing basis
    build + column-batched Green-integral assembly + ridge LS solve."""

    grid_shape = problem.grid_shape or (problem.nz_gi, problem.nx_gi)
    coords = problem.coords_gi
    b = problem.GI_rhs_c
    G0_fft = problem.G0_fft
    GI_scale_c = problem.GI_scale_c
    n_gi = coords.shape[0]

    @jax.jit
    def solve(model_vars, reg):
        Phi = model.apply(
            model_vars,
            coords,
            return_basis=True,
        )

        n_basis = Phi.shape[1]

        B_gi_blocks = []
        for start in range(0, n_basis, column_batch):
            end = min(start + column_batch, n_basis)
            Bcols = end - start

            Phi_batch = Phi[:, start:end]
            Phi_batch_grid = Phi_batch.T.reshape(
                (Bcols,) + tuple(grid_shape),
            )

            KPhi_batch_grid = green_apply_batch(
                Us_batch_c=Phi_batch_grid,
                GI_scale_c=GI_scale_c,
                G0_kernel_fft=G0_fft,
                grid_shape=grid_shape,
                w_damp=(0,) * len(grid_shape),
            )

            KPhi_batch = KPhi_batch_grid.reshape(Bcols, n_gi).T
            B_gi_block = Phi_batch - KPhi_batch
            B_gi_blocks.append(B_gi_block)

        B_gi = jnp.concatenate(B_gi_blocks, axis=1)

        n_unknowns = B_gi.shape[1]
        BHB = B_gi.conj().T @ B_gi
        BHb = B_gi.conj().T @ b
        eye = jnp.eye(n_unknowns, dtype=B_gi.dtype)

        # reg is a traced arg here
        BHB_reg = BHB + reg * eye

        L = jnp.linalg.cholesky(BHB_reg)
        y = jax.scipy.linalg.solve_triangular(L, BHb, lower=True)
        c = jax.scipy.linalg.solve_triangular(
            L.conj().T, y, lower=False,
        ).reshape(-1)

        return c

    return solve
