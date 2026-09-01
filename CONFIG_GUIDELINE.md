# Config guideline

Every entry a `config.py` (see any `Examples/*/config.py`) can set, grouped by
section, with its type, whether it's required, its default if optional, and
the rule/choices it must satisfy.

All of this is enforced automatically by `nap_wave.utils.validate_config`,
which runs first -- before any problem/model building starts -- inside
`build_gi_problem`, `build_gi_problem_3d`, and `Trainer.__init__`. A config
with several mistakes gets one error listing every one of them, not just the
first, e.g.:

```
Invalid config -- 3 problem(s) found before any problem/model building started:
  - config['arch.width']: must be an integer, got '128' (str). E.g. 128, not '128' or 128.0.
  - config['optimizer.lr']: must be strictly positive, got -0.001.
  - config['training.stage_iterations']: must have exactly one entry per depth stage (arch.depth=5), got 2 entries: [100, 1000].
```

Dimensionality (2D vs 3D) is not a config flag -- it's inferred from
`len(problem.domain)` (4 entries -> 2D, 6 entries -> 3D), and it's *also*
determined externally by which loader `main.py` calls
(`build_gi_problem` for 2D, `build_gi_problem_3d` for 3D). These must agree:
use the 2D loader with a 4-entry domain, the 3D loader with a 6-entry domain.

---

## `problem`

| Key | Type | Required | Default | Rule / choices |
|---|---|---|---|---|
| `frequency` | number | yes | -- | Hz, must be `> 0`. |
| `v0` | number | yes | -- | Background velocity, must be `> 0`. |
| `domain` | tuple/list of numbers | yes | -- | 2D: `(x_min, x_max, z_min, z_max)`, 4 entries. 3D: `(x0, x1, y0, y1, z0, z1)`, 6 entries. Every `min < max` pair. |
| `source` | tuple/list of numbers | yes | -- | 2D: `(sx, sz)`, 2 entries. 3D: `(sx, sy, sz)`, 3 entries -- must match `domain`'s dimensionality. |
| `factor` | number | no | `1.0` | Amplitude scale on the background field `U0`; any real number. |
| `precision` | string | no | `"float64"` | One of `float32`, `float64`, `32`, `64`, `single`, `double`, `fp32`, `fp64` (case-insensitive), same default for 2D and 3D. **Always use `float64`** -- see the README's "How it works" warning; the ridge-regularized LS solve is not stable in `float32`. |
| `seed` | int | no | `1999` | Non-negative; JAX PRNG seed. |
| `verbosity` | int | no | `1` | `0..3` (nothing above `3` has any additional effect). `0` = quiet, `1` = per-epoch log lines + start/finish banners, `2` = also checkpoint save/prune messages, LS column-batch info in the per-epoch line, and the one-time `print_every=0`/`save_every=0`/`keep_chpts=0` notices (see `training` below), `3` = also adaptive-depth growth messages. |
| `name` / `velocity_model` | string | no | `"Marmousi"` | Must be a non-empty string, e.g. `"Marmousi"` not `Marmousi` or `1`. Used to build the run name and to locate `data/data_<name>_validation[_<freq>Hz].npz`. A non-string value (e.g. `problem.name = 1`) is now caught here with a clear message, instead of surfacing later as a confusing `FileNotFoundError: ... data_1_validation.npz`. |
| `base_path` | string/path | no | auto-inferred | Must be a non-empty string if set. Root folder containing `data/` and where `Results/` is written; auto-detected by walking the call stack to the caller's directory if not set. |
| `gi_nx`, `gi_nz` (2D) | int | yes (2D) | -- | Green-integral grid resolution; each `> 0`. |
| `gi_nx`, `gi_ny`, `gi_nz` (3D) | int | yes (3D) | -- | Green-integral grid resolution per axis; each `> 0`. |
| `gi_damp` | tuple/list of ints | no | `(10, 10)` 2D / `w_damp` on all axes 3D | Absorbing-border width in grid cells. 2D: `(z_damp, x_damp)`, 2 entries. 3D: `(z_damp, y_damp, x_damp)`, 3 entries. Every entry `>= 0`, and small enough to leave a non-empty interior grid (checked when the GI grid is built). |
| `w_damp` (3D) | int | no | `20` | Fallback border width applied to all 3 axes when `gi_damp` isn't set; `>= 0`. |
| `k_low_quantile`, `k_high_quantile` | number | no | `0.05` / `0.95` | Each in `[0, 1]`; `k_low_quantile < k_high_quantile`. Quantile filter on the local-wavenumber field `omega/c(x)` before sampling `k_init`. |

## `arch`

| Key | Type | Required | Default | Rule / choices |
|---|---|---|---|---|
| `width` | int | no | `128` | `> 0`. Number of hidden trunk neurons per layer; `width // 2` complex plane-wave basis functions per layer. Odd values are silently bumped up by 1 (a warning is printed if `verbosity > 1`) -- not an error. |
| `depth` | int | no | `5` | `> 0`. Target (final) network depth; training grows from depth 1 up to this over `training.stage_iterations`, one entry per stage. |
| `k_min_clip` | number | no | `0.25` | `> 0`. Floor added to `softplus(raw_k)` so every wavenumber stays bounded away from 0. |

Plane-wave wavenumbers are always trained -- there is no `trainable_k` option.

## `training`

| Key | Type | Required | Default | Rule / choices |
|---|---|---|---|---|
| `stage_iterations` | list of ints | yes | -- | Every entry `> 0`. Length must equal `arch.depth` exactly -- one entry per depth stage, `1..depth`. Sum is the total training-iteration budget (`training.epochs` is not used for this; the schedule is). This is the adaptive-depth growth schedule -- there is no separate `adaptive` section. |
| `batch_size` | int | no | `problem.n_data` | `> 0` if set. |
| `print_every` | int | no | `100` | `>= 0`. Cadence for both the per-epoch console log line and metrics-history logging. `0` disables it (still prints/logs on the final epoch). If `verbosity > 1`, a one-time notice is printed at startup confirming `print_every=0` was intentional. |
| `save_every` | int | no | `1000` | `>= 0`. `0` disables periodic checkpoint saving (still saves on the final epoch, or on any epoch listed in `save_at_epochs`). If `verbosity > 1`, a one-time notice is printed at startup confirming `save_every=0` was intentional. |
| `save_at_epochs` | list of ints | no | `[]` | Every entry `>= 0`. Extra specific epochs to force-save a checkpoint at, regardless of `save_every`. |
| `time_iterations` | bool | no | `False` | Whether to record per-iteration wall-clock time in the log/history. |
| `max_train_minutes` | number or `None`/`""` | no | `""` (disabled) | `> 0` if set to a number. Training stops early once this many minutes have elapsed, even mid-schedule. |
| `keep_chpts` | int or `None` | no | `None` | `>= 0`, or `None`. `keep_chpts=m` keeps only the last `m` epoch checkpoints on disk (older ones deleted as new ones are saved). `keep_chpts=0` disables epoch-checkpoint saving entirely; `keep_chpts=None` saves every checkpoint (no pruning). |

## `optimizer`

| Key | Type | Required | Default | Rule / choices |
|---|---|---|---|---|
| `lr` | number | no | `1e-3` | `> 0`. Initial learning rate for the trunk parameters `theta` (the linear coefficients `c` are always solved by least squares, never by this optimizer). `lr` always follows an exponential-decay schedule -- there is no constant-LR option. |
| `decay_steps` | int | no | `1000` | `> 0`. Transition-steps parameter of the exponential decay. |
| `decay_rate` | number | no | `0.9` | `0 < decay_rate <= 1`. Decay factor per `decay_steps`. |

The trunk optimizer is always Adam -- there is no `trunk_optimizer` option.

## `least_squares`

| Key | Type | Required | Default | Rule / choices |
|---|---|---|---|---|
| `ls_reg_start` | number | no | `1e-3` | `> 0`. Ridge regularization `mu` at epoch 0, in `(B^*B + mu I)c = B^*f`. |
| `ls_reg_end` | number | no | `1e-6` | `> 0`. Ridge regularization `mu` at the final epoch; decays exponentially from `ls_reg_start` to `ls_reg_end` over training. |
