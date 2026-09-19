def init_config(config_ref):
    config_ref.use_floor = False
    config_ref.pcg_iter_count = 100
    config_ref.use_ccd_linesearch = True
    config_ref.nonlinear_iter_count = 1
    config_ref.use_gpu = False
    config_ref.use_untangling = True
    config_ref.gravity.y = 0.0
    config_ref.contact_energy_type = 0
    config_ref.use_quasi_static_mode = True
    config_ref.untangling_response_depth = 1e-0
    config_ref.print_collision_info = True
    config_ref.print_pcg_info = True
