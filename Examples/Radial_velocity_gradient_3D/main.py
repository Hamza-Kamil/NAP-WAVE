import jax
jax.config.update("jax_enable_x64", True)

from config import get_config

from nap_wave.problem_setup import build_gi_problem_3d
from nap_wave.trainer import Trainer


def main():
    cfg = get_config()
    problem = build_gi_problem_3d(cfg)
    trainer = Trainer(problem=problem, config=cfg)
    trainer.fit_evaluate()


if __name__ == "__main__":
    main()
