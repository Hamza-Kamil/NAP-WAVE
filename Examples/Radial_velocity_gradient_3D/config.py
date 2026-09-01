import ml_collections


def get_config():
    """
    3D radial-gradient exact-solution.
    """
    config = ml_collections.ConfigDict()

    # ============================================================
    # Problem
    # ============================================================
    config.problem = problem = ml_collections.ConfigDict()
    problem.name = "Radial_velocity_gradient_3D"

    problem.frequency = 2.0  # Hz 
    problem.v0 = 1.5  # km/s, background velocity 
    problem.source = (0.0, 0.0, 0.0)  # (sx, sy, sz) km, domain center
    problem.domain = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)  # (x0,x1,y0,y1,z0,z1) km

    # Green-integral grid 
    problem.gi_nx = 60
    problem.gi_ny = 60
    problem.gi_nz = 60
    problem.gi_damp = (20, 20, 20)  # (z_damp, y_damp, x_damp)

    problem.precision = "float64"
    problem.seed = 1999
    problem.verbosity = 1

    # ============================================================
    # Architecture
    # ============================================================
    config.arch = arch = ml_collections.ConfigDict()
    arch.width = 128
    arch.depth = 3

    # ============================================================
    # Training
    # ============================================================
    config.training = training = ml_collections.ConfigDict()
    training.stage_iterations = [100, 2000, 50000]
    training.print_every = 1000
    training.save_every = 10000
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
