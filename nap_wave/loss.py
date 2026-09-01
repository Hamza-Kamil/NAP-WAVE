"""
The Green-integral (Lippmann-Schwinger) residual loss, and the Optax
trunk-parameter optimizer/LR schedule used to minimize it.
"""

import optax, jax.numpy as jnp


# ============================================================
# Green integral loss
# ============================================================

class GreenIntegralLoss:
    """
    Empirical Green-integral loss:

        J = || B_theta c - f ||_2^2 / N_q .
    """

    def __call__(
        self,
        model,
        model_vars,
        state,
        batch_indices=None,
        epoch=0,
        aux_state=None,
    ):
        # u_sc(x) = sum_j c_j varphi_j(x; theta^n) on the GI grid.
        u_sc = model.apply(model_vars, state.coords_gi)

        # Reshape [N_gi, 1] complex to the GI grid.
        grid_shape = state.grid_shape or (state.nz_gi, state.nx_gi)
        Us_c = u_sc.reshape(grid_shape)

        # G[u_sc] via FFT with the precomputed kernel F[g_0]
        Uhat_c = state.lisch_operator_complex(
            Us_c=Us_c,
            w_damp=(0,) * len(grid_shape),
        )

        # Lippmann-Schwinger residual = B c - f.
        diff_c = u_sc - Uhat_c

        loss = jnp.mean(jnp.abs(diff_c) ** 2)

        return loss, aux_state


# ============================================================
# Loss factory
# ============================================================

def build_loss(config=None, custom_loss_fn=None):
    """
    The model is trained with the Green-integral (Lippmann-Schwinger)
    """
    return GreenIntegralLoss()


# ============================================================
# Main optimizer builder
# ============================================================

def build_optimizer(
    trunk_optimizer: str = "adam",
    lr: float = 1e-3,
    **kwargs,
):
    """
    Adam optimizer factory.
    """

    return "optax", optax.adam(lr)


# ============================================================
# Learning-rate schedule helper for Optax
# ============================================================

def build_lr_schedule(
    lr: float = 1e-3,
    decay_steps: int = 1000,
    decay_rate: float = 0.9,
):
    """
    Exponential-decay learning-rate schedule (always applied -- there is no
    constant-LR option).
    """

    return optax.exponential_decay(
        init_value=lr,
        transition_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False,
    )
