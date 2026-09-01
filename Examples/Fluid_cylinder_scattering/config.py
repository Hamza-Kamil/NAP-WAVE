import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # ============================================================
    # Problem
    # ============================================================
    config.problem = problem = ml_collections.ConfigDict()
    problem.name = "Fluid_cylinder_scattering"
    problem.frequency = 6.0 # frequency in Hz
    problem.v0 = 1.5  # background velocity
    problem.source = (0.3, 0.1)  # (sx, sz)
    problem.domain = (0.0, 3.0, 0.0, 2.0)  # (x_min, x_max, z_min, z_max)

    # Green-integral grid
    problem.gi_nx = 340
    problem.gi_nz = 240
    problem.gi_damp = (8, 8)  # (z_damp, x_damp)

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
    training.stage_iterations = [1000, 1000, 3000, 3000, 12000]
    training.print_every = 2500
    training.save_every = 10000
    training.max_train_minutes = ""
    training.keep_chpts = 1

    # ============================================================
    # Optimizer
    # ============================================================
    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.lr = 1e-3
    optimizer.decay_steps = 1000
    optimizer.decay_rate = 0.9

    # ============================================================
    # Least Squares
    # ============================================================
    config.least_squares = least_squares = ml_collections.ConfigDict()
    least_squares.ls_reg_start = 1e-3
    least_squares.ls_reg_end = 1e-6

    return config
