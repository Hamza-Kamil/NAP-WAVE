"""
Generic helpers used across the library.
"""

import pickle, numpy as np, jax, jax.numpy as jnp


# ============================================================
# Precision
# ============================================================

def resolve_precision(config):
    """"float32"/"float64" (accepting common aliases) -> (precision
    string, jnp real dtype, jnp complex dtype)."""
    precision = (
        config.get("problem", {}).get("precision", None)
        or config.get("precision", None)
        or "float64"
    )

    precision = str(precision).lower()

    if precision in ["float64", "64", "double", "fp64"]:
        return "float64", jnp.float64, jnp.complex128

    if precision in ["float32", "32", "single", "fp32"]:
        return "float32", jnp.float32, jnp.complex64

    raise ValueError(
        f"Unknown precision={precision}. Use 'float32' or 'float64'."
    )


def get_dtypes(precision="float64"):
    """"float32"/"float64" -> (jnp real, jnp complex, np real, np complex)."""
    precision = precision.lower()

    if precision in ["float32", "fp32", "single"]:
        return jnp.float32, jnp.complex64, np.float32, np.complex64

    if precision in ["float64", "fp64", "double"]:
        return jnp.float64, jnp.complex128, np.float64, np.complex128

    raise ValueError("precision must be 'float32' or 'float64'.")


def configure_precision(precision="float64"):
    """Enable/disable JAX x64 globally to match the requested precision."""
    precision = precision.lower()

    if precision in ["float64", "fp64", "double"]:
        jax.config.update("jax_enable_x64", True)
    else:
        jax.config.update("jax_enable_x64", False)


def cast_value_precision(x, real_dtype, complex_dtype):
    """
    Cast floating/complex JAX or NumPy arrays to the requested precision.
    Integer, boolean, string, and non-array values are kept unchanged.
    """

    if hasattr(x, "dtype"):
        if jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(real_dtype)
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            return x.astype(complex_dtype)

    return x


def cast_tree_precision(tree, real_dtype, complex_dtype):
    """Cast every floating/complex leaf in a PyTree to the requested precision."""

    return jax.tree_util.tree_map(
        lambda x: cast_value_precision(x, real_dtype, complex_dtype),
        tree,
    )


# ============================================================
# PyTree / pickle I/O
# ============================================================

def tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def flatten_param_count(params):
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(x.size for x in leaves))


# ============================================================
# Naming
# ============================================================

def make_run_name(config):
    """
    Run-name/checkpoint-filename tag: "<problem>_width<W>_depth<D>_<stages>".
    """
    problem_name = config["problem"].get("name", "problem")

    width = config["arch"].get("width", 128)
    depth = config["arch"].get("depth", 5)

    training_cfg = config.get("training", {})
    stage_iterations = list(training_cfg.get("stage_iterations", []))
    stages_tag = "-".join(str(int(n)) for n in stage_iterations)

    return f"{problem_name}_width{width}_depth{depth}_{stages_tag}"


# ============================================================
# Error classification
# ============================================================

def is_oom_error(exc):
    """True if `exc` looks like an out-of-memory failure (JAX/XLA
    RESOURCE_EXHAUSTED or a plain Python MemoryError)."""
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return "resource_exhausted" in msg or "out of memory" in msg or "oom" in msg


# ============================================================
# Console-output formatting -- small, dependency-free helpers for a
# clean, aligned log (banners + key/value blocks + a fixed-column
# per-epoch line), in the style of common deep-learning frameworks.
# ============================================================

RULE_WIDTH = 64


def rule(char="─", width=RULE_WIDTH):
    return char * width


def section(title, width=RULE_WIDTH):
    """`──── title ─────────`, centered and padded to `width`."""
    label = f" {title} "
    pad = max(0, width - len(label))
    left = pad // 2
    right = pad - left
    return f"{'─' * left}{label}{'─' * right}"


def kv_block(rows, key_width=16, indent="  "):
    """rows: list of (key, value); value may be a list for multi-line entries."""
    lines = []
    for key, value in rows:
        if isinstance(value, (list, tuple)) and not isinstance(value, str):
            value = list(value)
            lines.append(f"{indent}{key:<{key_width}} {value[0]}")
            for extra in value[1:]:
                lines.append(f"{indent}{'':<{key_width}} {extra}")
        else:
            lines.append(f"{indent}{key:<{key_width}} {value}")
    return "\n".join(lines)


# ============================================================
# Config validation
# ============================================================

_MISSING = object()


def _is_bool(x):
    return isinstance(x, bool)


def _is_int(x):
    """True for a real integer (Python int or numpy integer); bool is
    technically an int subclass but is never accepted as one here."""
    return isinstance(x, (int, np.integer)) and not _is_bool(x)


def _is_number(x):
    """True for a real int or float (Python or numpy); bool is not a
    number here even though bool is technically an int subclass."""
    return isinstance(x, (int, float, np.integer, np.floating)) and not _is_bool(x)


def _is_str(x):
    return isinstance(x, str)


class ConfigError(ValueError):
    """Raised by validate_config; message lists every problem found, not
    just the first one."""


def validate_config(config):
    """
    Check every config entry's type, sign, and choices. Raises one
    ConfigError listing all problems found, not just the first.
    """

    errors = []

    def require_section(name):
        section_cfg = config.get(name, _MISSING)
        if section_cfg is _MISSING:
            errors.append(f"config['{name}'] is missing (required section).")
            return {}
        return section_cfg

    def err(path, msg):
        errors.append(f"config['{path}']: {msg}")

    def check_int(section_cfg, section_name, key, default=_MISSING,
                   required=False, positive=False, nonneg=False, allow_none=False,
                   max_value=None):
        path = f"{section_name}.{key}"
        value = section_cfg.get(key, default)

        if value is _MISSING:
            if required:
                err(path, "is required but was not set.")
            return None

        if value is None:
            if allow_none:
                return None
            err(path, "cannot be None.")
            return None

        if not _is_int(value):
            err(
                path,
                f"must be an integer, got {value!r} ({type(value).__name__}). "
                f"E.g. 128, not '128' or 128.0.",
            )
            return None

        if positive and value <= 0:
            err(path, f"must be a positive integer, got {value}.")
        elif nonneg and value < 0:
            err(path, f"must be a non-negative integer, got {value}.")

        if max_value is not None and value > max_value:
            err(path, f"must be <= {max_value}, got {value}.")

        return value

    def check_number(section_cfg, section_name, key, default=_MISSING,
                      required=False, positive=False, nonneg=False,
                      max_value=None, allow_none=False, allow_empty_str=False):
        path = f"{section_name}.{key}"
        value = section_cfg.get(key, default)

        if value is _MISSING:
            if required:
                err(path, "is required but was not set.")
            return None

        if value is None and allow_none:
            return None

        if allow_empty_str and value in ("",):
            return None

        if not _is_number(value):
            err(
                path,
                f"must be a number, got {value!r} ({type(value).__name__}). "
                f"E.g. 1e-3, not '1e-3'.",
            )
            return None

        if positive and value <= 0:
            err(path, f"must be strictly positive, got {value}.")
        elif nonneg and value < 0:
            err(path, f"must be non-negative, got {value}.")

        if max_value is not None and value > max_value:
            err(path, f"must be <= {max_value}, got {value}.")

        return value

    def check_bool(section_cfg, section_name, key, default=_MISSING):
        path = f"{section_name}.{key}"
        value = section_cfg.get(key, default)

        if value is _MISSING or value is None:
            return None

        if not _is_bool(value):
            err(
                path,
                f"must be a bool (True/False), got {value!r} "
                f"({type(value).__name__}).",
            )
        return value

    def check_str(section_cfg, section_name, key, default=_MISSING, required=False,
                   non_empty=False):
        path = f"{section_name}.{key}"
        value = section_cfg.get(key, default)

        if value is _MISSING or value is None:
            if required:
                err(path, "is required but was not set.")
            return None

        if not _is_str(value):
            err(
                path,
                f"must be a string, got {value!r} ({type(value).__name__}). "
                f"E.g. \"{value}\", not {value!r} without quotes.",
            )
            return None

        if non_empty and value == "":
            err(path, "must not be an empty string.")

        return value

    def check_choice(section_cfg, section_name, key, choices, default=_MISSING,
                      case_insensitive=True):
        path = f"{section_name}.{key}"
        value = section_cfg.get(key, default)

        if value is _MISSING or value is None:
            return None

        if not _is_str(value):
            err(path, f"must be a string, got {value!r} ({type(value).__name__}).")
            return None

        check_value = value.lower() if case_insensitive else value
        allowed = [c.lower() for c in choices] if case_insensitive else list(choices)

        if check_value not in allowed:
            err(path, f"must be one of {choices}, got {value!r}.")

        return value

    def check_sequence_of_numbers(section_cfg, section_name, key, length=None,
                                   required=False, positive=False, nonneg=False):
        path = f"{section_name}.{key}"
        value = section_cfg.get(key, _MISSING)

        if value is _MISSING or value is None:
            if required:
                err(path, "is required but was not set.")
            return None

        if isinstance(value, str) or not hasattr(value, "__len__"):
            err(path, f"must be a list/tuple of numbers, got {value!r}.")
            return None

        if length is not None and len(value) != length:
            err(path, f"must have exactly {length} entries, got {len(value)}: {value!r}.")

        for v in value:
            if not _is_number(v):
                err(path, f"every entry must be a number, got {v!r} in {value!r}.")
                return None

        if positive and any(v <= 0 for v in value):
            err(path, f"every entry must be positive, got {value!r}.")
        elif nonneg and any(v < 0 for v in value):
            err(path, f"every entry must be non-negative, got {value!r}.")

        return value

    # ------------------------------------------------------------
    # problem
    # ------------------------------------------------------------
    problem_cfg = require_section("problem")

    check_number(problem_cfg, "problem", "frequency", required=True, positive=True)

    v0 = problem_cfg.get("v0", problem_cfg.get("c0", _MISSING))
    if v0 is _MISSING:
        err("problem.v0", "is required but was not set (v0, or its legacy alias c0).")
    elif not _is_number(v0):
        err("problem.v0", f"must be a number, got {v0!r} ({type(v0).__name__}).")
    elif v0 <= 0:
        err("problem.v0", f"must be a positive background velocity, got {v0}.")

    domain = problem_cfg.get("domain", _MISSING)
    ndim = None
    if domain is _MISSING or domain is None:
        err("problem.domain", "is required but was not set.")
    elif isinstance(domain, str) or not hasattr(domain, "__len__"):
        err("problem.domain", f"must be a list/tuple of numbers, got {domain!r}.")
    elif len(domain) not in (4, 6):
        err(
            "problem.domain",
            f"must have 4 entries (x_min,x_max,z_min,z_max, 2D) or 6 entries "
            f"(x0,x1,y0,y1,z0,z1, 3D), got {len(domain)}: {domain!r}.",
        )
    elif not all(_is_number(v) for v in domain):
        err("problem.domain", f"every entry must be a number, got {domain!r}.")
    else:
        ndim = 2 if len(domain) == 4 else 3
        pairs = [(domain[i], domain[i + 1]) for i in range(0, len(domain), 2)]
        axis_names = ["x", "z"] if ndim == 2 else ["x", "y", "z"]
        for (lo, hi), axis in zip(pairs, axis_names):
            if lo >= hi:
                err(
                    "problem.domain",
                    f"{axis} bounds must be increasing (min < max), got "
                    f"{axis}_min={lo} >= {axis}_max={hi} in {domain!r}.",
                )

    check_sequence_of_numbers(
        problem_cfg, "problem", "source",
        length=(ndim if ndim is not None else None),
        required=True,
    )

    check_number(problem_cfg, "problem", "factor", nonneg=False)

    check_choice(
        problem_cfg, "problem", "precision",
        choices=["float32", "float64", "32", "64", "single", "double", "fp32", "fp64"],
    )

    check_int(problem_cfg, "problem", "seed", nonneg=True)
    check_int(problem_cfg, "problem", "verbosity", nonneg=True, max_value=3)

    check_str(problem_cfg, "problem", "name", non_empty=True)
    check_str(problem_cfg, "problem", "velocity_model", non_empty=True)
    check_str(problem_cfg, "problem", "base_path", non_empty=True)

    if ndim == 3:
        check_int(problem_cfg, "problem", "gi_nx", required=True, positive=True)
        check_int(problem_cfg, "problem", "gi_ny", required=True, positive=True)
        check_int(problem_cfg, "problem", "gi_nz", required=True, positive=True)
        check_int(problem_cfg, "problem", "w_damp", nonneg=True)
        check_sequence_of_numbers(problem_cfg, "problem", "gi_damp", length=3, nonneg=True)
    else:
        check_int(problem_cfg, "problem", "gi_nx", required=True, positive=True)
        check_int(problem_cfg, "problem", "gi_nz", required=True, positive=True)
        check_sequence_of_numbers(problem_cfg, "problem", "gi_damp", length=2, nonneg=True)

    k_low = check_number(problem_cfg, "problem", "k_low_quantile", nonneg=True, max_value=1.0)
    k_high = check_number(problem_cfg, "problem", "k_high_quantile", nonneg=True, max_value=1.0)
    if k_low is not None and k_high is not None and k_low >= k_high:
        err(
            "problem.k_low_quantile/k_high_quantile",
            f"k_low_quantile must be < k_high_quantile, got "
            f"{k_low} >= {k_high}.",
        )

    # ------------------------------------------------------------
    # arch
    # ------------------------------------------------------------
    arch_cfg = require_section("arch")

    check_int(arch_cfg, "arch", "width", default=128, positive=True)
    depth = check_int(arch_cfg, "arch", "depth", default=5, positive=True)

    check_number(arch_cfg, "arch", "k_min_clip", positive=True)

    # Wavenumbers are always trained; only reject an explicit opt-out.
    if arch_cfg.get("trainable_k", True) is False:
        err(
            "arch.trainable_k",
            "is not a config option -- plane-wave wavenumbers are always "
            "trained. Remove this key.",
        )

    # ------------------------------------------------------------
    # training
    # ------------------------------------------------------------
    training_cfg = require_section("training")

    stage_iterations = check_sequence_of_numbers(
        training_cfg, "training", "stage_iterations", required=True, positive=True,
    )
    if stage_iterations is not None:
        if any(not _is_int(n) for n in stage_iterations):
            err("training.stage_iterations", f"every entry must be an integer, got {stage_iterations!r}.")
        elif depth is not None and len(stage_iterations) != depth:
            err(
                "training.stage_iterations",
                f"must have exactly one entry per depth stage (arch.depth={depth}), "
                f"got {len(stage_iterations)} entries: {stage_iterations!r}.",
            )

    check_int(training_cfg, "training", "batch_size", positive=True)
    check_int(training_cfg, "training", "print_every", nonneg=True)
    check_int(training_cfg, "training", "save_every", nonneg=True)
    check_sequence_of_numbers(training_cfg, "training", "save_at_epochs", nonneg=True)
    check_bool(training_cfg, "training", "time_iterations")
    check_number(
        training_cfg, "training", "max_train_minutes",
        positive=True, allow_none=True, allow_empty_str=True,
    )
    check_int(training_cfg, "training", "keep_chpts", nonneg=True, allow_none=True)

    # ------------------------------------------------------------
    # optimizer
    # ------------------------------------------------------------
    optimizer_cfg = require_section("optimizer")

    check_number(optimizer_cfg, "optimizer", "lr", positive=True)
    check_int(optimizer_cfg, "optimizer", "decay_steps", positive=True)
    check_number(optimizer_cfg, "optimizer", "decay_rate", positive=True, max_value=1.0)

    # LR decay is always on; only reject an explicit opt-out.
    if optimizer_cfg.get("use_lr_decay", True) is False:
        err(
            "optimizer.use_lr_decay",
            "is not a config option -- learning-rate decay is always on. "
            "Remove this key; use decay_steps/decay_rate to control it.",
        )

    # Only Adam is implemented; only reject an explicit different choice.
    trunk_optimizer = optimizer_cfg.get("trunk_optimizer", "adam")
    if _is_str(trunk_optimizer) and trunk_optimizer.lower() != "adam":
        err(
            "optimizer.trunk_optimizer",
            f"is not a config option -- the trunk optimizer is always Adam, "
            f"got {trunk_optimizer!r}. Remove this key.",
        )

    # ------------------------------------------------------------
    # least_squares
    # ------------------------------------------------------------
    ls_cfg = config.get("least_squares", {})

    check_number(ls_cfg, "least_squares", "ls_reg_start", positive=True)
    check_number(ls_cfg, "least_squares", "ls_reg_end", positive=True)

    # ------------------------------------------------------------
    # top-level
    # ------------------------------------------------------------
    # run_name is always auto-built; only reject an explicit override.
    if config.get("run_name", None) is not None:
        err(
            "run_name",
            "is not a config option -- it's auto-built from problem.name, "
            "arch.width, arch.depth, and training.stage_iterations. Remove "
            "this key.",
        )

    if errors:
        bullet_list = "\n".join(f"  - {e}" for e in errors)
        raise ConfigError(
            f"Invalid config -- {len(errors)} problem(s) found before any "
            f"problem/model building started:\n{bullet_list}"
        )
