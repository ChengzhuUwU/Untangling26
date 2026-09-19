function(luisa_compute_solver_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE luisa::compute luisa-compute-solver-lib)
    install(TARGETS ${name} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()



luisa_compute_solver_add_test(test_xtypes test_xtypes.cpp)
luisa_compute_solver_add_test(test_lc_features test_lc_features.cpp)
luisa_compute_solver_add_test(test_lbvh_refit test_lbvh_refit.cpp)
luisa_compute_solver_add_test(test_jit test_jit.cpp)
luisa_compute_solver_add_test(test_host_functions test_host_functions.cpp)
luisa_compute_solver_add_test(test_abd_energy test_abd_energy.cpp)
luisa_compute_solver_add_test(test_gradient_hessian test_gradient_hessian.cpp)
luisa_compute_solver_add_test(test_parallel_contour_construction test_parallel_contour_construction.cpp)
luisa_compute_solver_add_test(test_buffer_upload_stress test_buffer_upload_stress.cpp)
luisa_compute_solver_add_test(test_upload_race_min test_upload_race_min.cpp)
luisa_compute_solver_add_test(test_prp_selection test_prp_selection.cpp)
luisa_compute_solver_add_test(test_pcg_adaptive_lm test_pcg_adaptive_lm.cpp)
luisa_compute_solver_add_test(test_quasi_static_inertia test_quasi_static_inertia.cpp)

