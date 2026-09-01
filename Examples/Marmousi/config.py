import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # ============================================================
    # Problem
    # ============================================================
    config.problem = problem = ml_collections.ConfigDict()
    problem.name = "Marmousi"
    problem.frequency = 10.0
    problem.v0 = 1.5  # background velocity
    problem.source = (1.5133534669876099, 0.041681427508592606)  # (sx, sz)
    problem.domain = (0.0, 3.0, 0.0, 2.0)  # (x_min, x_max, z_min, z_max)
    problem.gi_nx = 180
    problem.gi_nz = 260
    problem.gi_damp = (10, 10)


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
    training.stage_iterations = [100, 1000, 2000, 3000, 13900]
    training.print_every = 1000
    training.save_every = 20000
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
