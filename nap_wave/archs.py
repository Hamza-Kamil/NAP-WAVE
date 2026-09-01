"""
Neural basis architectures: each hidden-neuron pair of a sine MLP forms one
complex basis phi_j(x; theta), carried by exp(i k_j d_j . x) in PlaneWaveBasisNet.
"""

import math, jax, jax.numpy as jnp, flax.linen as nn


# ============================================================
# Initializers
# ============================================================

def kernel_init():
    """Trunk-parameter initializer: uniform U(-sqrt(6/fan_in), sqrt(6/fan_in))."""

    def uniform_init(rng, shape, dtype=jnp.float32):
        fan_in = shape[0]
        limit = math.sqrt(6.0 / fan_in)
        return jax.random.uniform(
            rng,
            shape,
            dtype,
            minval=-limit,
            maxval=limit,
        )

    return uniform_init


def bias_init():
    return nn.initializers.zeros


# ============================================================
# Helpers
# ============================================================

def pair_to_complex(h):
    """Pair consecutive real neurons into complex basis: [h1,h2,h3,h4,...] -> [h1+ih2, h3+ih4,...]."""
    real = h[:, 0::2]
    imag = h[:, 1::2]

    return real + 1j * imag


def make_directions(coord_dim, n_basis):

    # Deterministic uniform on the unit sphere.

    if coord_dim == 2:
        angles = 2.0 * math.pi * jnp.arange(n_basis) / n_basis
        return jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    if coord_dim == 3:
        i = jnp.arange(n_basis) + 0.5
        phi = jnp.arccos(1.0 - 2.0 * i / n_basis)          # polar angle in [0, pi]
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))    
        theta = golden_angle * jnp.arange(n_basis)  

        x = jnp.sin(phi) * jnp.cos(theta)
        y = jnp.sin(phi) * jnp.sin(theta)
        z = jnp.cos(phi)
        return jnp.stack([x, y, z], axis=1)

    raise ValueError("Plane-wave directions support dimensions = 2 or 3.")


def normalize_k_init(k_init, n_basis, k_min_clip):
    if k_init is None:
        k = jnp.ones((n_basis,))
    else:
        k = jnp.asarray(k_init, dtype=jnp.float32).reshape(-1)

        if k.size == 1:
            k = jnp.repeat(k, n_basis)

        elif k.size < n_basis:
            repeat_factor = math.ceil(n_basis / k.size)
            k = jnp.tile(k, repeat_factor)[:n_basis]

        elif k.size > n_basis:
            k = k[:n_basis]

    raw_k = jnp.log(
        jnp.expm1(
            jnp.maximum(k - k_min_clip, 1e-6)
        )
    )

    return raw_k


# ============================================================
# Plane-wave basis network
# ============================================================

class PlaneWaveBasisNet(nn.Module):
    """Plane-wave basis: phi_j(x;theta) = A_j(x;theta) * exp(i k_j d_j . x), A_j
    from a sine MLP. u_N(x) = sum_j c_j phi_j(x), c solved by least squares."""

    input_dim: int
    width: int
    depth: int
    coord_dim: int = 2
    k_init: object = None
    k_min_clip: float = 0.25

    def coefficients(self, n_basis):
        coeff_real = self.param(
            "coeff_real",
            nn.initializers.zeros,
            (n_basis,),
        )

        coeff_imag = self.param(
            "coeff_imag",
            nn.initializers.zeros,
            (n_basis,),
        )

        return coeff_real + 1j * coeff_imag       # c  =  Re(c) + i Im(c)

    def combine_basis(self, Phi):
        # u_N(x)  =  sum_{j=1}^{N_\ell}  c_j  varphi_j(x; theta).
        n_basis = Phi.shape[1]
        c = self.coefficients(n_basis)

        if Phi.shape[1] != c.shape[0]:
            raise RuntimeError(
                f"Basis/coeff mismatch: Phi has {Phi.shape[1]} columns, "
                f"but coefficient vector has length {c.shape[0]}."
            )

        # Sum  Phi * c  over the basis index j  ->  u_N evaluated at each x.
        return jnp.sum(Phi * c[None, :], axis=1, keepdims=True)

    def plane_wave(self, coords, raw_k, start, end, total_basis_dim):

        directions = make_directions(self.coord_dim, total_basis_dim)
        directions = directions[start:end]

        k = nn.softplus(raw_k) + self.k_min_clip
        k = k[start:end]

        x = coords[:, :self.coord_dim]

        phase = x @ directions.T
        phase = phase * k[None, :]

        return jnp.exp(1j * phase)

    @nn.compact
    def __call__(self, coords, return_basis=False):
        # Builds Phi_theta one layer at a time.
        act = jnp.sin  # sine activation 
        kernel_init_fn = kernel_init()

        n_layer_basis = self.width // 2

        if self.k_init is not None:
            k_init_flat = jnp.asarray(self.k_init, dtype=jnp.float32).reshape(-1)
        else:
            k_init_flat = None

        h = coords
        basis_list = []

        for layer_id in range(self.depth):
            h = nn.Dense(
                self.width,
                kernel_init=kernel_init_fn,
                bias_init=bias_init(),
                name=f"dense_{layer_id}",
            )(h)
            h = act(h)

            amp = pair_to_complex(h)                 # complex amplitudes A_j
            n_layer_basis = amp.shape[1]             # = width // 2

            if k_init_flat is not None:
                start_idx = layer_id * n_layer_basis
                end_idx = start_idx + n_layer_basis
                if k_init_flat.size >= end_idx:
                    layer_k_init = k_init_flat[start_idx:end_idx]
                else:
                    # Pool too short - normalize_k_init below tiles/repeats it.
                    layer_k_init = k_init_flat
            else:
                layer_k_init = None

            raw_init = normalize_k_init(
                layer_k_init,
                n_layer_basis,
                self.k_min_clip,
            )
            # Wavenumbers are always trained.
            param_name = f"raw_k_{layer_id}"
            if self.is_initializing() or self.has_variable("params", param_name):
                raw_k = self.param(param_name, lambda rng, shape: raw_init, (n_layer_basis,))
            else:
                # Legacy checkpoint saved before wavenumbers were always
                # trainable (back when `trainable_k=False` was the
                # default): raw_k_i was never stored as a param, it was
                # this same constant. Reconstruct it so old checkpoints
                # still evaluate.
                raw_k = raw_init

            carrier = self.plane_wave(
                coords,
                raw_k,
                start=0,
                end=n_layer_basis,
                total_basis_dim=n_layer_basis,
            )                                        # exp(i k_j d_j . x)
            basis_list.append(amp * carrier)         # varphi_j  for this layer

        # Phi_{\theta}  =  [ varphi_1, ..., varphi_{N_\ell} ]
        Phi = jnp.concatenate(basis_list, axis=1)

        if return_basis:
            # Used by the LS solver to assemble  B = Phi - G Phi.
            return Phi

        # Otherwise, return u_N(x) = sum_j c_j varphi_j(x; theta).
        return self.combine_basis(Phi)


# ============================================================
# Build architecture
# ============================================================

def build_architecture(
    input_dim: int,
    width: int,
    depth: int,
    verbosity: int = 0,
    **kwargs,
):
    """Build the plane-wave basis network u(x; theta, c) = sum_j c_j phi_j(x; theta)."""

    if width % 2 != 0:
        width += 1
        if verbosity > 1:
            print(
                f"[build_architecture] width must be even - "
                f"bumped up to {width}.",
                flush=True,
            )

    return PlaneWaveBasisNet(
        input_dim=input_dim,
        width=width,
        depth=depth,
        coord_dim=kwargs.get("coord_dim", input_dim),
        k_init=kwargs.get("k_init", None),
        k_min_clip=kwargs.get("k_min_clip", 0.25),
    )