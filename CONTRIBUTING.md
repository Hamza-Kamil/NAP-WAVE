# Contributing to NAP-Wave

Thanks for your interest in extending NAP-Wave. This page lists what the
library currently does *not* do (so you don't rediscover the same gaps the
hard way), a set of concrete directions that would make good contributions,
and how to actually send a pull request.

---

## Current limitations

- **Single GPU / single device only.** Training runs on one JAX device. 
  There's no `pmap`/`jit`-based sharding across multiple GPUs or hosts, so
  problem size is capped by a single card's memory. This will be suitable for 3D problems that needs more training points.
- **Single frequency per run.** Each `config.py` sets one `problem.frequency`
  and trains one model for it. Multi-frequency problems are solved today by
  launching separate independent runs, not by one model that generalizes
  across frequency.
- **Single source per run.** Similarly, `problem.source` is one point; there's
  no batching over multiple source locations inside a single training run.
- **No operator-learning / generalization across velocity models.** Each
  trained network is tied to one velocity model (`problem.name`). The model
  doesn't take the velocity field as an input, so it can't be reused on a new
  model without retraining from scratch.

If you run into a limitation not listed here, please open an issue. It
helps prioritize what to document or fix next.

---

## Potential improvements

### Multi-GPU / parallelization

- Shard the Green's-integral grid and FFT across devices with `jax.pmap` or
  `jax.experimental.shard_map`, so larger 3D grids fit in memory.
- Parallelize the least-squares coefficient solve (`least_squares.py`) across
  devices when the plane-wave basis count is large.
- Data-parallel training across multiple sources or frequencies at once, rather than one GPU per independent run.

### Multi-frequency

- Extend `problem_setup.py` / `Trainer` to accept a list of frequencies and
  either (a) train one network per frequency with shared infrastructure and
  batched dispatch, or (b) condition a single network on frequency as an
  extra input, so one model covers a frequency band.
- Add a frequency-sweep example config and a small utility to launch/collect
  a batch of single-frequency runs.

### Multi-source

- Batch `problem.source` over multiple source points within one training run
  (shared trunk parameters, per-source linear coefficients solved by least
  squares), instead of one run per source.
- Investigate whether the plane-wave basis can be shared across sources to
  amortize the Green's-integral setup cost.

### Operator learning

- Condition the trunk network on the velocity field itself (e.g. a
  DeepONet-style branch net, or a Fourier neural operator front end) so a
  single trained model generalizes to unseen velocity models instead of
  requiring retraining per `problem.name`.
- Explore meta-learning / few-shot fine-tuning: pretrain across several
  velocity models, then adapt quickly to a new one.

### Other potential improvements

- **Checkpointing**: pluggable checkpoint formats (e.g. Orbax) instead of the
  current pickle-based save/load, for better cross-version compatibility.
- **Adaptive-depth schedule search**: `training.stage_iterations` is
  currently hand-tuned per example config; a small utility to suggest a
  schedule from `arch.depth` and a target iteration budget would reduce
  trial and error for new problems.
- **More architectures**: `archs.py` currently implements one plane-wave
  basis network (`PlaneWaveBasisNet`); alternative trunk architectures could
  be added behind the same `build_architecture` entry point.
- **Documentation**: more worked examples in `Examples/` for building your own velocity model dataset.

---

## How to submit a pull request

1. **Open an issue first** for anything beyond a small fix (typo, doc
   clarification, obvious bug) -- a short discussion up front avoids
   wasted work on approaches that won't be merged.
2. **Fork the repository** and create a branch off `main` named for what
   it does, e.g. `multi-gpu-fft` or `fix-checkpoint-pruning`.
3. **Keep changes focused.** One pull request per concern -- don't mix a
   feature with an unrelated refactor.
4. **Run the checks you have locally** before opening the PR:
   - The package should still import cleanly: `pip install -e .`
   - `python -m py_compile nap_wave/*.py` should pass with no errors.
   - If you touched `nap_wave/utils.py`'s `validate_config`, run it against
     every example config in `Examples/*/config.py` and confirm each still
     validates.
5. **Update `CONFIG_GUIDELINE.md`** if you add, remove, or change the
   default of any config option -- it's the single source of truth for what
   `config.py` accepts.
6. **Write a clear PR description**: what problem it solves, how you tested
   it, and any config/behavior changes a user of the library needs to know
   about.
7. **Open the pull request** against `main` and respond to review feedback --
   most reviews focus on correctness of the physics/numerics (Green's
   integral, least-squares regularization, adaptive-depth growth) and on
   keeping the config surface consistent between 2D and 3D.

Questions before starting on something larger (multi-GPU, operator learning)
are welcome as a GitHub issue -- happy to discuss design direction before
you invest the time.
