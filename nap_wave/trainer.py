"""
Training loop: each iteration solves the linear coefficients c by
regularized least squares, then updates theta by one Adam step on the
Green-integral loss, with an adaptive-depth schedule that grows the
network and warm-starts new layers at prescribed stage boundaries.
"""

import os, glob, time, numpy as np
from functools import partial

import jax, jax.numpy as jnp

import optax

from .archs import build_architecture
from .loss import build_optimizer, build_lr_schedule, build_loss
from .least_squares import (
    get_ls_reg,
    make_jitted_ls_core,
    assign_coefficients_to_model_vars,
)
from .utils import (
    resolve_precision,
    cast_value_precision,
    cast_tree_precision,
    tree_to_numpy,
    save_pickle,
    flatten_param_count,
    is_oom_error,
    make_run_name,
    rule,
    section,
    kv_block,
    validate_config,
)


# ============================================================
# Trainer
# ============================================================

class Trainer:
    """JAX-Flax trainer for u(x) = sum_j c_j phi_j(x; theta); c_j always
    solved by least squares, theta by Adam."""

    def __init__(self, problem, config, custom_loss_fn=None):
        validate_config(config)

        self.problem = problem
        self.config = config

        # ------------------------------------------------------------
        # Precision
        # ------------------------------------------------------------
        self.precision, self.real_dtype, self.complex_dtype = resolve_precision(config)

        if self.precision == "float64" and not jax.config.read("jax_enable_x64"):
            print(
                "WARNING: config['precision']='float64', but JAX x64 is not enabled. "
                "Put `import jax; "
                "jax.config.update('jax_enable_x64', True)` before importing JAX/trainer.",
                flush=True,
            )

        # Cast common problem arrays so model/loss/data use one precision.
        self._cast_problem_to_precision()

        # ------------------------------------------------------------
        # Output paths
        # ------------------------------------------------------------
        self.run_name = make_run_name(config)

        problem_cfg = config.get("problem", {})

        self.problem_name = problem_cfg.get(
            "velocity_model", problem_cfg.get("name", "problem")
        )

        self.verbosity = problem_cfg.get("verbosity", 1)

        base_path = problem_cfg.get("base_path", ".")


        self.output_dir = os.path.join(base_path, "Results")
        os.makedirs(self.output_dir, exist_ok=True)

        self.metrics_path = os.path.join(
            self.output_dir,
            f"{self.run_name}_metrics.npy",
        )

        self.checkpoint_path = os.path.join(
            self.output_dir,
            f"{self.run_name}_checkpoint.pkl",
        )

        seed = int(config["problem"].get("seed", 1999))
        self.rng = jax.random.PRNGKey(seed)

        dummy_coords = jnp.zeros(
            (1, problem.input_dim),
            dtype=self.real_dtype,
        )

        # ------------------------------------------------------------
        # Build architecture
        # ------------------------------------------------------------
        arch_cfg = config["arch"]


        # Training always starts at depth 1 and grows to arch['depth'].
        stage_cfg = config.get("training", {})
        self.adaptive_final_depth = int(arch_cfg.get("depth", 5))
        self.adaptive_stage_start_epoch = 0
        self.current_depth = 1

        if self.adaptive_final_depth < 1:
            raise ValueError(
                f"arch['depth'] must be >= 1 (got depth={self.adaptive_final_depth})."
            )

        # depth == 1 means a single fixed-depth network -- no growth stages,
        # adaptive training is simply a no-op.
        self.adaptive_enabled = self.adaptive_final_depth > 1

        # Iterations per depth stage.
        self.adaptive_stage_iterations = [
            int(x) for x in stage_cfg.get("stage_iterations", [])
        ]

        n_stages = self.adaptive_final_depth

        if len(self.adaptive_stage_iterations) != n_stages:
            raise ValueError(
                "training['stage_iterations'] must be a list of length "
                f"{n_stages} (one entry per depth stage, "
                f"1..{self.adaptive_final_depth}); "
                f"got length {len(self.adaptive_stage_iterations)}."
            )

        if any(n <= 0 for n in self.adaptive_stage_iterations):
            raise ValueError(
                "training['stage_iterations'] entries must all be > 0 "
                f"(got {self.adaptive_stage_iterations})."
            )
        # Sum-vs-budget reconciliation happens later in _adaptive_stage_lengths.

        self.model = self._build_model_for_depth(self.current_depth)

        # Initialize model variables.
        self.rng, model_key = jax.random.split(self.rng)


        self.model_vars = self.model.init(
            model_key,
            dummy_coords,
            return_basis=False,
        )

        self.model_vars = {
            "params": cast_tree_precision(
                self.model_vars["params"],
                self.real_dtype,
                self.complex_dtype,
            )
        }

        # ------------------------------------------------------------
        # Build loss
        # ------------------------------------------------------------
        self.loss_fn = build_loss(
            config=config,
            custom_loss_fn=custom_loss_fn,
        )

        self.aux_loss_state = {}

        # Build optimizer.
        opt_cfg = config["optimizer"]
        self._configure_optimizer(opt_cfg, reset_state=False)

        # ------------------------------------------------------------
        # Training settings
        # ------------------------------------------------------------
        train_cfg = config["training"]

        # Total iteration budget is the stage table's sum, not training.epochs.
        self.epochs = sum(self.adaptive_stage_iterations)

        self._batch_size_uses_problem_default = "batch_size" not in train_cfg
        self.batch_size = int(train_cfg.get("batch_size", problem.n_data))
        self.print_every = int(train_cfg.get("print_every", 100))
        self.save_every = int(train_cfg.get("save_every", 1000))
        self.save_at_epochs = set(
            int(e) for e in train_cfg.get("save_at_epochs", [])
        )

        # print_every/save_every=0 disable that cadence (the final epoch
        # still always prints/saves).
        if self.verbosity > 1 and self.print_every == 0:
            print(
                "[log] training.print_every=0 -- no per-epoch console line "
                "will be printed (still prints on the final epoch).",
                flush=True,
            )

        if self.verbosity > 1 and self.save_every == 0:
            print(
                "[checkpoint] training.save_every=0 -- no periodic checkpoint "
                "will be saved (still saves on the final epoch, or on any "
                "epoch listed in save_at_epochs).",
                flush=True,
            )

        # Total training time is always tracked.
        self.time_iterations = bool(train_cfg.get("time_iterations", False))

        max_train_minutes = train_cfg.get("max_train_minutes", None)

        if max_train_minutes in [None, "", 0, 0.0]:
            self.max_train_seconds = None
        else:
            self.max_train_seconds = 60.0 * float(max_train_minutes)

        # Gradient-norm clipping is set at 1.0.
        self.grad_clip_norm = 1.0

        # keep_chpts=m keeps only the last m epoch checkpoints. m=0 saves
        # none. keep_chpts=None saves every checkpoint (no pruning).
        keep_chpts_cfg = train_cfg.get("keep_chpts", None)
        if keep_chpts_cfg in (None, ""):
            self.keep_chpts = None
        else:
            self.keep_chpts = int(keep_chpts_cfg)
            if self.keep_chpts < 0:
                raise ValueError(
                    "training.keep_chpts must be a non-negative integer or "
                    f"None (got {keep_chpts_cfg!r})."
                )

        if self.verbosity > 1 and self.keep_chpts == 0:
            print(
                "[checkpoint] training.keep_chpts=0 -- no epoch checkpoints "
                "will be saved. Use keep_chpts=None (not 0) to save every "
                "checkpoint.",
                flush=True,
            )

        # ------------------------------------------------------------
        # LS settings
        # ------------------------------------------------------------
        ls_cfg = config.get("least_squares", {})

        self.ls_reg_start = ls_cfg.get("ls_reg_start", 1e-1)
        self.ls_reg_end = ls_cfg.get("ls_reg_end", 1e-4)

        # Column batching is automatic (backs off by half on OOM).
        self.column_batch = None

        # optax.adam's init already infers dtype from these (already-cast)
        # params, so its mu/nu moment buffers come out at the right
        # precision without an extra cast here.
        self.opt_state = self.optimizer.init(self.model_vars["params"])

        self.jitted_optax_step = self.make_jitted_optax_step()

        # ------------------------------------------------------------
        # Fused JIT LS solver (basis build + GI assembly + ridge solve,
        # all in one compiled program -- see least_squares.make_jitted_ls_core).
        # Rebuilt whenever column_batch or the model (depth growth) change.
        # ------------------------------------------------------------
        self.ls_solver = None
        self._ls_solver_column_batch = None

        # ------------------------------------------------------------
        # History
        # ------------------------------------------------------------
        self.history = self._init_history()
        self._problem_data_signature_snapshot = self._problem_data_signature()

    # ============================================================
    # Optimizer setup
    # ============================================================

    def _configure_optimizer(self, opt_cfg, reset_state):
        """Build the Adam optimizer + LR schedule from config["optimizer"]."""
        self.trunk_optimizer_name = "adam"

        lr_or_schedule = build_lr_schedule(
            lr=opt_cfg.get("lr", 1e-3),
            decay_steps=opt_cfg.get("decay_steps", 1000),
            decay_rate=opt_cfg.get("decay_rate", 0.9),
        )

        self.optimizer_family, self.optimizer = build_optimizer(
            trunk_optimizer=self.trunk_optimizer_name,
            lr=lr_or_schedule,
        )

        if reset_state:
            params_for_opt = cast_tree_precision(
                self.model_vars["params"],
                self.real_dtype,
                self.complex_dtype,
            )
            # init already infers dtype from params_for_opt -- no extra cast needed.
            self.opt_state = self.optimizer.init(params_for_opt)
            self.jitted_optax_step = self.make_jitted_optax_step()

    # ============================================================
    # Adaptive-depth helpers
    # ============================================================

    def _basis_count_for_depth(self, depth):
        """
        Number of complex basis functions for pair-based architectures
        """
        arch_cfg = self.config["arch"]
        width = int(arch_cfg.get("width", 128))
        return int(depth) * (width // 2)  # --> N_\ell

    def _make_k_init_for_depth(self, depth):
        """Raw k_init pool sampled from the problem's local-wavenumber field.

        Resizing (repeat/tile/truncate) to fit the model's per-layer basis
        count is handled once, in archs.normalize_k_init - not duplicated
        here.
        """
        k_init = getattr(self.problem, "k_init", None)
        if k_init is None:
            return None
        return jnp.asarray(k_init, dtype=self.real_dtype).reshape(-1)

    def _build_model_for_depth(self, depth):
        """Build the same architecture at a prescribed adaptive depth."""
        arch_cfg = self.config["arch"]
        return build_architecture(
            input_dim=self.problem.input_dim,
            width=arch_cfg.get("width", 128),
            depth=int(depth),
            coord_dim=self.problem.input_dim,
            k_init=self._make_k_init_for_depth(depth),
            k_min_clip=arch_cfg.get("k_min_clip", 0.25),
            verbosity=self.verbosity,
        )

    def _init_model_vars_for_depth(self, depth):
        """Initialize a fresh parameter tree for a model of a given depth."""
        model = self._build_model_for_depth(depth)
        self.rng, model_key = jax.random.split(self.rng)

        dummy_coords = jnp.zeros(
            (1, self.problem.input_dim),
            dtype=self.real_dtype,
        )

        model_vars = model.init(
            model_key,
            dummy_coords,
            return_basis=False,
        )

        model_vars = {
            "params": cast_tree_precision(
                model_vars["params"],
                self.real_dtype,
                self.complex_dtype,
            )
        }

        return model, model_vars

    def _copy_array_compatible(self, old, new):
        """
        Copy an old parameter into a new parameter when shapes are compatible.
        """
        if not (hasattr(old, "shape") and hasattr(new, "shape")):
            return new

        if old.shape == new.shape:
            return old.astype(new.dtype) if hasattr(old, "astype") else old

        if old.ndim == new.ndim == 1 and old.shape[0] <= new.shape[0]:
            return new.at[: old.shape[0]].set(old.astype(new.dtype))

        return new

    def _copy_compatible_params(self, old_params, new_params):
        """
        Copy old layer/coefficient params into the grown model.
        """

        def merge_dict(old_d, new_d, path=()):
            out = {}

            for key, new_value in new_d.items():
                current_path = path + (key,)

                if key not in old_d:
                    # Brand-new subtree (e.g. the new hidden layer) -- keep its fresh init.
                    out[key] = new_value
                    continue

                old_value = old_d[key]

                if isinstance(old_value, dict) and isinstance(new_value, dict):
                    out[key] = merge_dict(old_value, new_value, current_path)
                    continue

                # --------------------------------------------------------
                # Special treatment for output coefficients
                # --------------------------------------------------------
                if key in ["coeff_real", "coeff_imag"]:
                    if (
                        hasattr(old_value, "shape")
                        and hasattr(new_value, "shape")
                        and old_value.ndim == new_value.ndim == 1
                        and old_value.shape[0] <= new_value.shape[0]
                    ):
                        copied = jnp.zeros_like(new_value)
                        copied = copied.at[: old_value.shape[0]].set(
                            old_value.astype(new_value.dtype)
                        )
                        out[key] = copied
                        continue

                # --------------------------------------------------------
                # Default copy rule
                # --------------------------------------------------------
                out[key] = self._copy_array_compatible(old_value, new_value)

            return out

        return merge_dict(old_params, new_params)
    
    def _merge_optimizer_state_compatible(self, old_state, new_state):
        """
        Transfer Optax state where possible: matching leaves (Adam moments) are
        copied, new leaves keep the freshly initialized state.
        """
        if old_state is None:
            return new_state

        if hasattr(old_state, "_fields") and hasattr(new_state, "_fields"):
            if type(old_state) is type(new_state) and old_state._fields == new_state._fields:
                values = [
                    self._merge_optimizer_state_compatible(getattr(old_state, f), getattr(new_state, f))
                    for f in old_state._fields
                ]
                return type(new_state)(*values)

        if isinstance(old_state, tuple) and isinstance(new_state, tuple) and len(old_state) == len(new_state):
            return type(new_state)(
                self._merge_optimizer_state_compatible(o, n)
                for o, n in zip(old_state, new_state)
            )

        if isinstance(old_state, list) and isinstance(new_state, list) and len(old_state) == len(new_state):
            return [
                self._merge_optimizer_state_compatible(o, n)
                for o, n in zip(old_state, new_state)
            ]

        if isinstance(old_state, dict) and isinstance(new_state, dict):
            out = {}
            for key, new_value in new_state.items():
                if key in old_state:
                    out[key] = self._merge_optimizer_state_compatible(old_state[key], new_value)
                else:
                    out[key] = new_value
            return out

        return self._copy_array_compatible(old_state, new_state)

    def _adaptive_stage_lengths(self):
        """
        Per-stage iteration counts (T_1, ..., T_L), from training['stage_iterations'].
        """

        first = 1
        last = int(self.adaptive_final_depth)

        depths = list(range(first, last + 1))
        n_stages = len(depths)

        T_total = int(self.epochs)

        lengths = list(self.adaptive_stage_iterations)

        if len(lengths) != n_stages:
            raise ValueError(
                f"training['stage_iterations'] must have length {n_stages} "
                f"(got {len(lengths)})."
            )

        diff = T_total - sum(lengths)
        if diff != 0:
            lengths[-1] += diff
            if lengths[-1] <= 0:
                raise ValueError(
                    "training['stage_iterations'] sums to "
                    f"{sum(lengths) - diff}, off from the iteration budget "
                    f"T_total={T_total} by {diff}; adjusting the last stage "
                    f"by this difference would make it non-positive "
                    f"({lengths[-1]})."
                )
            if self.verbosity > 1:
                print(
                    "[training] stage_iterations "
                    f"{list(self.adaptive_stage_iterations)} sums to "
                    f"{sum(self.adaptive_stage_iterations)}, not T_total="
                    f"{T_total}; adding the difference ({diff}) to the last "
                    f"depth stage -> final stage_iterations={lengths}.",
                    flush=True,
                )

        return [int(x) for x in lengths]

    def _adaptive_boundaries(self):
        lengths = self._adaptive_stage_lengths()
        if len(lengths) <= 1:
            return []
        return list(np.cumsum(lengths)[:-1])

    def _rebuild_data_dependent_jitted_functions(self):
        """Rebuild JIT closures because they close over the model and problem."""
        self.jitted_optax_step = self.make_jitted_optax_step()

        self.ls_solver = None
        self._ls_solver_column_batch = None

    def _rebuild_jitted_functions_after_growth(self):
        """Rebuild JIT closures after changing the adaptive model."""
        self._rebuild_data_dependent_jitted_functions()

    def _grow_adaptive_model(self, new_depth, epoch=None):
        """
        Grow from depth ell to ell+1: layers 1..ell-1 and their Adam moments are copied.
        """
        old_params = self.model_vars["params"]
        old_opt_state = self.opt_state

        new_model, new_model_vars = self._init_model_vars_for_depth(new_depth)
        new_params = self._copy_compatible_params(
            old_params=old_params,
            new_params=new_model_vars["params"],
        )

        self.model = new_model
        self.model_vars = {
            "params": cast_tree_precision(
                new_params,
                self.real_dtype,
                self.complex_dtype,
            )
        }
        self.current_depth = int(new_depth)

        if epoch is not None:
             self.adaptive_stage_start_epoch = int(epoch)


        fresh_state = self.optimizer.init(self.model_vars["params"])

        # The LR step count is never reset after a depth change -- it keeps decaying continuously.
        self.opt_state = self._merge_optimizer_state_compatible(
            old_opt_state,
            fresh_state,
        )

        self._rebuild_jitted_functions_after_growth()

        if self.verbosity > 2:
            print(
                f"[adaptive] grew network to depth {self.current_depth}; "
                f"n_basis={self._basis_count_for_depth(self.current_depth)}",
                flush=True,
            )

    def _grow_adaptive_model_if_scheduled(self, epoch):
        boundaries = self._adaptive_boundaries()
        while (
            self.current_depth < self.adaptive_final_depth
            and epoch >= boundaries[self.current_depth - 1]
        ):
            self._grow_adaptive_model(self.current_depth + 1, epoch=epoch)

    def _cast_problem_to_precision(self):
        """
        Cast floating/complex array attributes of the problem object to the trainer precision.
        """

        if not hasattr(self.problem, "__dict__"):
            return

        for name, value in vars(self.problem).items():
            if isinstance(value, (jax.Array, np.ndarray)):
                try:
                    setattr(
                        self.problem,
                        name,
                        cast_value_precision(
                            value,
                            self.real_dtype,
                            self.complex_dtype,
                        ),
                    )
                except Exception:
                    pass

        # Keep these conventional dtype attributes synchronized if present.
        try:
            self.problem.real_dtype = self.real_dtype
        except Exception:
            pass

        try:
            self.problem.complex_dtype = self.complex_dtype
        except Exception:
            pass

    def _problem_data_signature(self):
        """
        Return metadata that changes when problem data is replaced or resized.
        """
        if not hasattr(self.problem, "__dict__"):
            return ()

        signature = []
        for name, value in sorted(vars(self.problem).items()):
            if isinstance(value, (jax.Array, np.ndarray)):
                signature.append(
                    (
                        name,
                        id(value),
                        tuple(value.shape),
                        str(value.dtype),
                    )
                )
            elif name in {
                "n_data",
                "nx",
                "nz",
                "nx_data",
                "nz_data",
                "nx_gi",
                "nz_gi",
            }:
                try:
                    signature.append((name, int(value)))
                except (TypeError, ValueError):
                    signature.append((name, repr(value)))

        return tuple(signature)

    def _validate_problem_grid(self):
        """Fail early when grid metadata and regenerated arrays disagree."""
        problem_cfg = self.config.get("problem", {})
        for name in ("nx", "nz", "nx_data", "nz_data", "nx_gi", "nz_gi"):
            configured = problem_cfg.get(name, None)
            actual = getattr(self.problem, name, None)
            if configured is not None and actual is not None:
                if int(configured) != int(actual):
                    raise ValueError(
                        f"config['problem']['{name}']={int(configured)}, but "
                        f"problem.{name}={int(actual)}. The config was changed "
                        "without regenerating the problem data."
                    )

        n_data = getattr(self.problem, "n_data", None)
        coords_data = getattr(self.problem, "coords_data", None)

        if n_data is not None and coords_data is not None:
            if int(n_data) != int(coords_data.shape[0]):
                raise ValueError(
                    "Problem data is stale: problem.n_data="
                    f"{int(n_data)}, but coords_data has {int(coords_data.shape[0])} rows. "
                    "Regenerate the problem arrays after changing nx/nz."
                )

        nx_data = getattr(self.problem, "nx_data", None)
        nz_data = getattr(self.problem, "nz_data", None)
        if n_data is not None and nx_data is not None and nz_data is not None:
            grid_size = int(nx_data) * int(nz_data)
            if grid_size != int(n_data):
                raise ValueError(
                    "Problem grid metadata is inconsistent: "
                    f"nx_data*nz_data={grid_size}, but n_data={int(n_data)}. "
                    "Regenerate coords_data and GI data after changing nx/nz."
                )

        for field_name in ("u_real", "u_imag"):
            field = getattr(self.problem, field_name, None)
            if n_data is not None and field is not None and int(field.size) != int(n_data):
                raise ValueError(
                    f"Problem data is stale: {field_name} has {int(field.size)} values, "
                    f"but n_data={int(n_data)}."
                )

    def _refresh_problem_data_if_changed(self, force=False):
        """
        Rebuild data-dependent JIT closures after problem arrays/grid changes.
        """
        self._validate_problem_grid()
        current_signature = self._problem_data_signature()
        previous_signature = getattr(
            self,
            "_problem_data_signature_snapshot",
            None,
        )

        if not force and current_signature == previous_signature:
            return False

        self._cast_problem_to_precision()
        self._validate_problem_grid()

        if self._batch_size_uses_problem_default:
            self.batch_size = int(self.problem.n_data)

        self.aux_loss_state = {}
        self._rebuild_data_dependent_jitted_functions()
        self._problem_data_signature_snapshot = self._problem_data_signature()

        if self.verbosity >= 1:
            print(
                "[problem data] grid/arrays changed; rebuilt GI loss JIT functions "
                f"(n_data={int(self.problem.n_data)})",
                flush=True,
            )

        return True

    # ============================================================
    # Save checkpoint and predictions
    # ============================================================

    def save_checkpoint(self, epoch=None, primary=False):
        # primary=True writes the plain "<run_name>_checkpoint.pkl" (final
        # model); otherwise writes an epoch-numbered snapshot, pruned by
        # keep_chpts.
        is_primary = primary or epoch is None

        # keep_chpts=0 disables epoch checkpoints entirely -- skip before
        # doing any work (the primary self.checkpoint_path save is
        # unaffected).
        if not is_primary and self.keep_chpts == 0:
            return

        path = self.checkpoint_path if is_primary else os.path.join(
            self.output_dir,
            f"{self.run_name}_checkpoint_epoch_{epoch}.pkl",
        )

        checkpoint = {
            "run_name": self.run_name,
            "epoch": epoch,
            "model_vars": tree_to_numpy(self.model_vars),
            "opt_state": tree_to_numpy(self.opt_state)
            if self.opt_state is not None
            else None,
            "config": self.config,
            "history": self.history,
        }

        save_pickle(path, checkpoint)

        if self.verbosity >= 2:
            print(f"[checkpoint] saved -> {path}", flush=True)

        if not is_primary:
            self._prune_old_checkpoints()

    def _prune_old_checkpoints(self):
        """Delete epoch checkpoints beyond the last keep_chpts."""
        if self.keep_chpts is None:
            return

        prefix = f"{self.run_name}_checkpoint_epoch_"
        pattern = os.path.join(self.output_dir, f"{prefix}*.pkl")

        def epoch_of(path):
            name = os.path.basename(path)
            epoch_str = name[len(prefix):-len(".pkl")]
            try:
                return int(epoch_str)
            except ValueError:
                return -1

        existing = sorted(glob.glob(pattern), key=epoch_of)

        for old_path in existing[:len(existing) - self.keep_chpts]:
            try:
                os.remove(old_path)
                if self.verbosity >= 2:
                    print(f"[checkpoint] removed -> {old_path}", flush=True)
            except OSError:
                pass

    # ============================================================
    # History
    # ============================================================

    def _init_history(self):
        return {
            "epoch": [],
            "time_elapsed": [],
            "loss": [],

            # full complex metrics only
            "rel_l2_complex": [],
            "rel_l2_complex_sq": [],

            "lr": [],
            "ls_reg": [],

            "run_name": self.run_name,
            "arch": "plane_wave",
            "loss_type": "GI",
            "trunk_optimizer": self.trunk_optimizer_name,
            "optimizer_family": self.optimizer_family,
            "c_update": "ls",
            "precision": self.precision,
            "n_params": flatten_param_count(self.model_vars["params"]),
            "adaptive_enabled": self.adaptive_enabled,
            "adaptive_stage_lengths": self._adaptive_stage_lengths(),
            "g0_regularization": "circle",
            "g0_reg_tag": "cirlc",
            "adaptive_depth": [],
        }

    # ============================================================
    # LS update
    # ============================================================

    def solve_coefficients_ls(self, epoch, batch_indices):
        current_ls_reg = get_ls_reg(
            epoch=epoch,
            total_epochs=self.epochs,
            ls_reg_start=self.ls_reg_start,
            ls_reg_end=self.ls_reg_end,
        )

        if self.column_batch is not None:
            column_batch = self.column_batch
        else:
            # "auto": start at the full basis count the model will ever
            # reach (no batching), then back off by half on OOM.
            column_batch = self._basis_count_for_depth(self.adaptive_final_depth)

        while True:
            try:
                if self.ls_solver is None or self._ls_solver_column_batch != column_batch:
                    self.ls_solver = make_jitted_ls_core(
                        model=self.model,
                        problem=self.problem,
                        column_batch=column_batch,
                    )
                    self._ls_solver_column_batch = column_batch

                c = self.ls_solver(self.model_vars, current_ls_reg)
                jax.block_until_ready(c)
                break
            except Exception as e:
                if not is_oom_error(e) or column_batch <= 1:
                    raise
                column_batch = max(1, column_batch // 2)
                self.ls_solver = None
                if self.verbosity >= 1:
                    print(
                        f"[column_batch] out of memory; retrying with "
                        f"column_batch={column_batch}.",
                        flush=True,
                    )

        self.column_batch = column_batch

        n_basis = int(c.shape[0])
        n_gi = int(self.problem.coords_gi.shape[0])
        n_column_batches = -(-n_basis // column_batch)  # ceil division

        # Sanity check: GI-LS must actually enter the column-batch loop.
        if n_column_batches <= 0:
            raise RuntimeError(
                "GI-LS did not execute any column-batch loop. "
                f"n_column_batches={n_column_batches}, column_batch={self.column_batch}."
            )

        self.model_vars = assign_coefficients_to_model_vars(
            model_vars=self.model_vars,
            c_complex=c,
        )

        info = {
            "ls_type": "GI_complex",
            "n_basis": n_basis,
            "n_gi": n_gi,
            "n_unknowns": n_basis,
            "column_batch": int(column_batch),
            "n_column_batches": int(n_column_batches),
            "complex_ls": True,
            "reg": float(current_ls_reg),
            "ls_reg": current_ls_reg,
        }

        return info

    def make_jitted_optax_step(self):
        """
        One JIT-compiled Optax step: loss, value_and_grad, coefficient-gradient
        masking, gradient clipping, apply_updates.
        """

        optimizer = self.optimizer
        model = self.model
        problem = self.problem
        loss_fn = self.loss_fn
        grad_clip_norm = self.grad_clip_norm
        real_dtype = self.real_dtype
        complex_dtype = self.complex_dtype

        # donate_argnums=(0, 1): params/opt_state are dead once the caller
        # reassigns them, so XLA can reuse their memory for the outputs.
        @partial(jax.jit, donate_argnums=(0, 1))
        def step(params, opt_state, batch_indices, epoch, aux_loss_state):
            params = cast_tree_precision(params, real_dtype, complex_dtype)
            opt_state = cast_tree_precision(opt_state, real_dtype, complex_dtype)

            def objective(params):
                model_vars = {"params": params}

                loss_value, new_aux_state = loss_fn(
                    model=model,
                    model_vars=model_vars,
                    state=problem,
                    batch_indices=batch_indices,
                    epoch=epoch,
                    aux_state=aux_loss_state,
                )

                return jnp.asarray(loss_value, dtype=real_dtype), new_aux_state

            (loss_value, new_aux_state), grads = jax.value_and_grad(
                objective,
                has_aux=True,
            )(params)

            # Coefficients are always recovered by least squares.
            if "coeff_real" in grads:
                grads["coeff_real"] = jnp.zeros_like(grads["coeff_real"])

            if "coeff_imag" in grads:
                grads["coeff_imag"] = jnp.zeros_like(grads["coeff_imag"])

            # Always clip the trunk gradient's global norm to 1.0.
            clipper = optax.clip_by_global_norm(grad_clip_norm)
            clip_state = clipper.init(params)

            grads, _ = clipper.update(
                grads,
                clip_state,
                params,
            )

            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                params,
            )

            new_params = optax.apply_updates(params, updates)

            return new_params, opt_state, loss_value, new_aux_state

        return step


    # ============================================================
    # One optimizer step
    # ============================================================

    def optimizer_step(self, batch_indices, epoch):
        """One JIT-compiled Adam step ."""
        # No eager precision cast: params are already at the right
        # precision, and jitted_optax_step casts internally anyway.
        epoch_arr = jnp.asarray(epoch)

        new_params, self.opt_state, loss_value, self.aux_loss_state = self.jitted_optax_step(
            self.model_vars["params"],
            self.opt_state,
            batch_indices,
            epoch_arr,
            self.aux_loss_state,
        )

        self.model_vars = {"params": new_params}

        return jnp.asarray(loss_value, dtype=self.real_dtype)

    # ============================================================
    # Logging
    # ============================================================

    def current_lr(self, epoch):
        opt_cfg = self.config["optimizer"]

        lr0 = float(opt_cfg.get("lr", 1e-3))
        decay_steps = float(opt_cfg.get("decay_steps", 1000))
        decay_rate = float(opt_cfg.get("decay_rate", 0.9))

        local_epoch = int(epoch)

        return lr0 * decay_rate ** (local_epoch / decay_steps)

    def evaluate_metrics(self, loss_value=None):
        coords = self.problem.coords_data

        # Predicted complex scattered field.
        u_pred_complex = self.model.apply(
            self.model_vars,
            coords,
        )

        # True complex scattered field
        u_true_complex = (
            self.problem.u_real
            + 1j * self.problem.u_imag
        )

        err_complex = u_pred_complex - u_true_complex

        # ============================================================
        # Full complex metrics
        # ============================================================

        rel_l2_complex = jnp.linalg.norm(err_complex) / (
            jnp.linalg.norm(u_true_complex) + 1e-16
        )

        rel_l2_complex_sq = rel_l2_complex ** 2

        if loss_value is None:
            loss_float = np.nan
        else:
            loss_float = float(jax.device_get(loss_value))

        return {
            "loss": loss_float,
            "rel_l2_complex": float(jax.device_get(rel_l2_complex)),
            "rel_l2_complex_sq": float(jax.device_get(rel_l2_complex_sq)),
        }
    
    def append_history(self, epoch, elapsed, metrics, current_lr, ls_info=None):
        self.history["epoch"].append(epoch)
        self.history["time_elapsed"].append(elapsed)

        self.history["loss"].append(metrics["loss"])
        self.history["rel_l2_complex"].append(metrics["rel_l2_complex"])
        self.history["rel_l2_complex_sq"].append(metrics["rel_l2_complex_sq"])
        self.history["lr"].append(float(current_lr))
        self.history["adaptive_depth"].append(int(self.current_depth))

        if ls_info is not None:
            self.history["ls_reg"].append(float(ls_info.get("ls_reg", np.nan)))
        else:
            self.history["ls_reg"].append(np.nan)

    def print_epoch(self, epoch, metrics, current_lr, iter_time=None, ls_info=None):
        if self.verbosity >= 1:
            total_width = len(str(self.epochs))

            # rel_l2_complex is stored as a raw fraction (||err|| / ||true||);
            # *100 here is only for display.
            fields = [
                f"epoch {epoch:>{total_width}d}/{self.epochs}",
                f"depth {self.current_depth}/{self.adaptive_final_depth}",
                f"loss {metrics['loss']:.3e}",
                f"rel_L2 {metrics['rel_l2_complex'] * 100:7.4f}%",
            ]
            if iter_time is not None:
                fields.append(f"{iter_time:.3f}s/it")
            if self.verbosity >= 2 and ls_info is not None:
                fields.append(
                    f"col_batch {ls_info.get('column_batch')}/"
                    f"{ls_info.get('n_basis')} "
                    f"(x{ls_info.get('n_column_batches')})"
                )

            msg = f"[{self.problem_name}] " + "  │  ".join(fields)
            print(msg, flush=True)


    def save_epoch(self, epoch):
        np.save(self.metrics_path, self.history, allow_pickle=True)
        self.save_checkpoint(epoch=epoch)


    def log_epoch(self, epoch, elapsed, loss, ls_info=None, save=False,
                  print_msg=False, log_metrics=False, iter_time=None):
        if not print_msg and not save and not log_metrics:
            return

        current_lr = self.current_lr(epoch)

        metrics = self.evaluate_metrics(loss)

        # History row cadence is log_metrics OR save, independent of print_msg.
        if log_metrics or save:
            self.append_history(
                epoch=epoch,
                elapsed=elapsed,
                metrics=metrics,
                current_lr=current_lr,
                ls_info=ls_info,
            )

        if print_msg:
            self.print_epoch(
                epoch=epoch,
                metrics=metrics,
                current_lr=current_lr,
                iter_time=iter_time,
                ls_info=ls_info,
            )

        if save:
            self.save_epoch(epoch)

    # ============================================================
    # Main fit + evaluate loop
    # ============================================================

    def fit_evaluate(self):
        self._refresh_problem_data_if_changed()

        if self.verbosity >= 1:
            width = int(self.config["arch"].get("width", 128))
            depth = int(self.config["arch"].get("depth", 5))
            n_basis = depth * (width // 2)

            # The linear coefficients c are always solved by least squares.
            trunk = self.trunk_optimizer_name

            depths = list(range(1, self.adaptive_final_depth + 1))
            adaptive_row = (
                f"depths {depths}"
                if self.adaptive_enabled
                else "disabled (fixed depth=1)"
            )

            print()
            print(section(self.problem_name))
            print(kv_block([
                ("backend", f"{jax.default_backend()}  ·  {jax.devices()}"),
                ("basis functions", f"{n_basis:,}"),
                ("parameters", f"{flatten_param_count(self.model_vars['params']):,}"),
                ("optimizer", f"{trunk} [trunk]  ·  least squares [coeffs]"),
                ("adaptive", [adaptive_row, f"stages {self.adaptive_stage_iterations}"]),
                ("metrics file", self.metrics_path),
            ]))
            print(rule())
            print()

        t_start = time.perf_counter()

        # Hybrid LS/neural optimization with adaptive depth.
        for epoch in range(self.epochs + 1):
            self._refresh_problem_data_if_changed()
            self._grow_adaptive_model_if_scheduled(epoch)

            epoch_t0 = time.perf_counter() if self.time_iterations else None

            # Training always uses the GI loss, evaluated on the full grid.
            batch_indices = None

            # LS solve for c: build Phi, assemble B = Phi - G Phi,
            # solve (B^* B + mu I) c = B^* f.
            ls_info = self.solve_coefficients_ls(
                epoch=epoch,
                batch_indices=batch_indices,
            )

            # Adam step for theta. Gradients w.r.t. (coeff_real, coeff_imag)
            # are always masked to zero, so only theta moves.
            loss = self.optimizer_step(
                batch_indices=batch_indices,
                epoch=epoch,
            )

            elapsed = time.perf_counter() - t_start

            # Wall time of this iteration.
            iter_time = (time.perf_counter() - epoch_t0) if epoch_t0 is not None else None

            # ----------------------------------------------------
            # 3. Print / save
            # ----------------------------------------------------
            # print_every=0 disables that cadence (still fires on do_final).
            # Metrics history logs on the same cadence as the console line.
            do_print = self.print_every > 0 and epoch % self.print_every == 0
            # save_every=0, like print_every=0, disables that cadence (still
            # fires on do_final) -- guarded to avoid a ZeroDivisionError.
            do_save = (
                (self.save_every > 0 and epoch % self.save_every == 0)
                or epoch in self.save_at_epochs
            )
            do_final = epoch == self.epochs

            self.log_epoch(
                epoch=epoch,
                elapsed=elapsed,
                loss=loss,
                ls_info=ls_info,
                save=do_save or do_final,
                print_msg=do_print or do_final,
                log_metrics=do_print or do_final,
                iter_time=iter_time,
            )

            # ----------------------------------------------------
            # 4. Time limit
            # ----------------------------------------------------
            if self.max_train_seconds is not None and elapsed >= self.max_train_seconds:
                if self.verbosity >= 1:
                    print()
                    print(section("Stopping · time limit reached"))
                    print(kv_block([
                        ("epoch", epoch),
                        ("elapsed", f"{elapsed / 60.0:.2f} min"),
                    ]))
                    print(rule())
                    print()

                self.history["stopped_epoch"] = epoch
                self.history["stop_reason"] = "time_limit"
                self.history["total_training_time"] = elapsed
                self.history["total_training_time_minutes"] = elapsed / 60.0

                np.save(self.metrics_path, self.history, allow_pickle=True)
                self.save_checkpoint(epoch=epoch)

                break

        total_time = time.perf_counter() - t_start

        self.history["total_training_time"] = total_time
        self.history["total_training_time_minutes"] = total_time / 60.0

        np.save(self.metrics_path, self.history, allow_pickle=True)

        final_epoch = (
            self.history["epoch"][-1]
            if len(self.history["epoch"]) > 0
            else None
        )

        # Final "<run_name>_checkpoint.pkl" save (the epoch-numbered file
        # for this same epoch was already written inside the loop above).
        self.save_checkpoint(epoch=final_epoch, primary=True)

        if self.verbosity >= 1:
            print()
            print(section(f"Finished · {self.problem_name}"))
            print(kv_block([
                ("total time", f"{total_time / 60.0:.2f} min"),
                ("final epoch", final_epoch),
                ("metrics file", self.metrics_path),
                ("checkpoint", self.checkpoint_path),
            ]))
            print(rule())
            print()

        return self.history
