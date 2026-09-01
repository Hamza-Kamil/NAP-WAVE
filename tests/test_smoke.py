"""
End-to-end smoke test: build a tiny problem (small grid, depth=1, a couple
of iterations) and run Trainer.fit_evaluate() to completion.

This is NOT a correctness/accuracy test -- it exists to catch the class of
break that matters most for a forked research repo: an install that
succeeds but a training run that crashes, hangs, or produces NaNs because
of an incompatible dependency version, a bad refactor, or an API change
in JAX/Flax/Optax. Runtime is a few seconds on CPU.

It reuses Examples/Fluid_cylinder_scattering's real config.py and validation data
(the smallest example in the repo) so it stays honest to the actual code
path users run, but overrides the architecture/grid/iteration settings to
be small, and redirects all output to a pytest tmp_path so it never writes
into the tracked Examples/*/Results folder.
"""
import shutil
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "Examples" / "Fluid_cylinder_scattering"


def _load_example_config():
    """Import Examples/Fluid_cylinder_scattering/config.py the same way main.py does."""
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        from config import get_config  # noqa: PLC0415 (deliberately late import)
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
    return get_config()


def test_short_training_run_completes_without_nan(tmp_path):
    from nap_wave.problem_setup import build_gi_problem
    from nap_wave.trainer import Trainer

    # Copy just the reference data into an isolated scratch dir so nothing
    # is written into the real example folder.
    shutil.copytree(EXAMPLE_DIR / "data", tmp_path / "data")

    cfg = _load_example_config()

    cfg.problem.base_path = str(tmp_path)
    cfg.problem.verbosity = 0
    # Small Green-integral grid -> fast FFT-based assembly.
    cfg.problem.gi_nx = 40
    cfg.problem.gi_nz = 30

    # Fixed depth-1 network (adaptive growth is exercised separately by the
    # full examples; this test is about the training loop wiring, not
    # accuracy or the growth schedule).
    cfg.arch.depth = 1
    cfg.arch.width = 8

    cfg.training.stage_iterations = [2]
    cfg.training.print_every = 0
    cfg.training.save_every = 0

    problem = build_gi_problem(cfg)
    trainer = Trainer(problem=problem, config=cfg)
    history = trainer.fit_evaluate()

    assert len(history["loss"]) > 0, "fit_evaluate() produced no logged loss values"

    final_loss = history["loss"][-1]
    assert final_loss == final_loss, "final loss is NaN"  # NaN != NaN
    assert final_loss < float("inf"), "final loss is infinite"
