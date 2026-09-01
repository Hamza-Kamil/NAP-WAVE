import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # ============================================================
    # Problem
    # ============================================================
    config.problem = problem = ml_collections.ConfigDict()
    problem.name = "Overthrust"
    problem.frequency = 10.0
    problem.v0 = 2.856261730194092  # background velocity
    problem.source = (6.264591217041016, 0.052033282816410065)  # (sx, sz)
    problem.domain = (0.0, 12.5, 0.0, 4.0)  # (x_min, x_max, z_min, z_max)
    problem.factor = 3.9119086
    problem.gi_nx = 730
    problem.gi_nz = 254
    problem.gi_damp = (15, 15)  # (z_damp, x_damp)

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
    training.stage_iterations = [500, 2500, 5000, 12000, 80000]
    training.print_every = 2000
    training.save_every = 50000
    training.max_train_minutes = ""
    training.keep_chpts = 1

    # ============================================================
    # Optimizer
    # ============================================================
    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.lr = 1e-3
    optimizer.decay_steps = 2000
    optimizer.decay_rate = 0.9

    # ============================================================
    # Least Squares
    # ============================================================
    config.least_squares = least_squares = ml_collections.ConfigDict()
    least_squares.ls_reg_start = 1e-3
    least_squares.ls_reg_end = 1e-6

    return config
