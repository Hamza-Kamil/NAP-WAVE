import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # ============================================================
    # Problem
    # ============================================================
    config.problem = problem = ml_collections.ConfigDict()
    problem.name = "Otway"
    problem.frequency = 20.0
    problem.v0 = 1650.0  # background velocity
    problem.source = (1220.4000244140625, 3.5999999046325684)  # (sx, sz)
    problem.domain = (0.0, 2438.0, 0.0, 1966.0)  # (x_min, x_max, z_min, z_max)
    problem.gi_nx = 330
    problem.gi_nz = 532
    problem.gi_damp = (20, 10)  # (z_damp, x_damp)

    problem.precision = "float64"
    problem.seed = 1999

    problem.verbosity = 1

    # ============================================================
    # Architecture
    # ============================================================
    config.arch = arch = ml_collections.ConfigDict()
    arch.width = 128
    arch.depth = 5

    # ============================================================
    # Training
    # ============================================================
    config.training = training = ml_collections.ConfigDict()
    training.stage_iterations = [1000, 3000, 5000, 10000, 200000]
    training.print_every = 10000
    training.save_every = 50000
    training.max_train_minutes = ""
    training.keep_chpts = 1

    # ============================================================
    # Optimizer
    # ============================================================
    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.lr = 1e-3
    optimizer.decay_steps = 10000
    optimizer.decay_rate = 0.9

    # ============================================================
    # Least Squares
    # ============================================================
    config.least_squares = least_squares = ml_collections.ConfigDict()
    least_squares.ls_reg_start = 1e-3
    least_squares.ls_reg_end = 1e-6

    return config
