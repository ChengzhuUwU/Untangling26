cmake_minimum_required(VERSION 3.26)

set(CMAKE_CXX_STANDARD 20)

project(LuisaComputeSimulator)


# Global options
option(LCS_ENABLE_GUI "Enable GUI support" OFF)
option(LCS_ENABLE_TEST "Enable Unit Tests" OFF)
option(LCS_USE_SYSTEM_PARALLEL_FOR "Enable system parallel_for" OFF)
option(LCS_BUILD_MAIN_APPLICATION "Build Executable Simulation Application" ON)
option(LCS_BUILD_PYBINDINGS "Build Python bindings (requires pybind11)" OFF)
set(LCS_PYTHON_EXECUTABLE "" CACHE FILEPATH "Path to the Python executable to build python bindings for (optional; auto-detected if empty)")

# In scikit-build-core (pip install / pip wheel) context, override defaults
# for Python package builds. This avoids duplicating these settings in
# pyproject.toml's [tool.scikit-build.cmake.define].
if(SKBUILD)
    set(LCS_BUILD_PYBINDINGS ON CACHE BOOL "" FORCE)
endif()

if(LCS_BUILD_PYBINDINGS)
    # Check if user explicitly specified LCS_PYTHON_EXECUTABLE via CLI
    set(_IS_EXPLICIT FALSE)
    if(DEFINED LCS_PYTHON_EXECUTABLE AND NOT LCS_PYTHON_EXECUTABLE STREQUAL "")
        if(NOT DEFINED LCS_AUTO_PYTHON_EXECUTABLE OR NOT "${LCS_PYTHON_EXECUTABLE}" STREQUAL "${LCS_AUTO_PYTHON_EXECUTABLE}")
            set(_IS_EXPLICIT TRUE)
        endif()
    endif()

    set(_CURRENT_ENV_SIG "")
    set(_DETECTED_PYTHON "")

    if(_IS_EXPLICIT)
        # Explicit CLI override has highest priority
        set(_DETECTED_PYTHON "${LCS_PYTHON_EXECUTABLE}")
        set(_CURRENT_ENV_SIG "EXPLICIT:${LCS_PYTHON_EXECUTABLE}")
    elseif(DEFINED Python_EXECUTABLE AND NOT "${Python_EXECUTABLE}" STREQUAL "" AND NOT DEFINED LCS_CACHED_ENV_SIG)
        # Passed by caller (e.g., scikit-build-core / FindPython)
        set(_DETECTED_PYTHON "${Python_EXECUTABLE}")
        set(_CURRENT_ENV_SIG "CMAKE_PYTHON_EXECUTABLE:${Python_EXECUTABLE}")
    elseif(DEFINED Python3_EXECUTABLE AND NOT "${Python3_EXECUTABLE}" STREQUAL "" AND NOT DEFINED LCS_CACHED_ENV_SIG)
        # Passed by caller (e.g., scikit-build-core / FindPython3)
        set(_DETECTED_PYTHON "${Python3_EXECUTABLE}")
        set(_CURRENT_ENV_SIG "CMAKE_PYTHON3_EXECUTABLE:${Python3_EXECUTABLE}")
    elseif(DEFINED ENV{VIRTUAL_ENV} AND NOT "$ENV{VIRTUAL_ENV}" STREQUAL "")
        # Activated venv / virtualenv / uv
        set(_CURRENT_ENV_SIG "VIRTUAL_ENV:$ENV{VIRTUAL_ENV}")
        if(WIN32)
            set(_DETECTED_PYTHON "$ENV{VIRTUAL_ENV}/Scripts/python.exe")
        else()
            set(_DETECTED_PYTHON "$ENV{VIRTUAL_ENV}/bin/python")
        endif()
    elseif(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
        # Activated Conda / Mamba environment
        set(_CURRENT_ENV_SIG "CONDA:$ENV{CONDA_PREFIX}")
        if(WIN32)
            set(_DETECTED_PYTHON "$ENV{CONDA_PREFIX}/python.exe")
        else()
            set(_DETECTED_PYTHON "$ENV{CONDA_PREFIX}/bin/python")
        endif()
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv")
        # Local .venv in repository root (auto-fallback if not activated)
        set(_CURRENT_ENV_SIG "LOCAL_VENV:${CMAKE_CURRENT_SOURCE_DIR}/.venv")
        if(WIN32)
            set(_DETECTED_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe")
        else()
            set(_DETECTED_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
        endif()
    endif()

    # 2. Check if active environment signature changed; if so, invalidate stale CMake cache
    if(DEFINED LCS_CACHED_ENV_SIG AND NOT "${_CURRENT_ENV_SIG}" STREQUAL "${LCS_CACHED_ENV_SIG}")
        message(STATUS "[LCS] Active Python environment changed from '${LCS_CACHED_ENV_SIG}' to '${_CURRENT_ENV_SIG}'. Invalidating stale Python cache...")
        unset(LCS_PYTHON_EXECUTABLE CACHE)
        unset(LCS_AUTO_PYTHON_EXECUTABLE CACHE)
        unset(PYBIND11_PYTHON_EXECUTABLE CACHE)
        unset(PYTHON_EXECUTABLE CACHE)
        unset(Python_EXECUTABLE CACHE)
        unset(Python_INCLUDE_DIR CACHE)
        unset(Python_INCLUDE_DIRS CACHE)
        unset(Python_LIBRARIES CACHE)
        unset(Python_LIBRARY_RELEASE CACHE)
        unset(Python_LIBRARY_DEBUG CACHE)
        unset(Python_SOABI CACHE)
        unset(Python_SITEARCH CACHE)
        unset(Python_SITELIB CACHE)
        unset(_Python_EXECUTABLE CACHE)
    endif()
    set(LCS_CACHED_ENV_SIG "${_CURRENT_ENV_SIG}" CACHE INTERNAL "Cached Python environment signature for cache invalidation" FORCE)

    # 3. If a specific python executable was detected, validate and hint it to FindPython
    if(NOT _DETECTED_PYTHON STREQUAL "" AND EXISTS "${_DETECTED_PYTHON}")
        file(TO_CMAKE_PATH "${_DETECTED_PYTHON}" _DETECTED_PYTHON_NORM)
        set(LCS_PYTHON_EXECUTABLE "${_DETECTED_PYTHON_NORM}" CACHE FILEPATH "Path to the Python executable" FORCE)
        if(NOT _IS_EXPLICIT)
            set(LCS_AUTO_PYTHON_EXECUTABLE "${_DETECTED_PYTHON_NORM}" CACHE INTERNAL "Auto-detected Python executable" FORCE)
        else()
            unset(LCS_AUTO_PYTHON_EXECUTABLE CACHE)
        endif()
        get_filename_component(_DETECTED_PYTHON_DIR "${_DETECTED_PYTHON_NORM}" DIRECTORY)
        get_filename_component(_DETECTED_PYTHON_HOME "${_DETECTED_PYTHON_DIR}" DIRECTORY)
        set(Python_ROOT_DIR "${_DETECTED_PYTHON_HOME}")
        set(Python_EXECUTABLE "${_DETECTED_PYTHON_NORM}" CACHE FILEPATH "Python executable" FORCE)
    endif()

    # 4. Use FindPython with VIRTUALENV FIRST and LOCATION strategy
    set(Python_FIND_VIRTUALENV FIRST)
    set(Python_FIND_STRATEGY LOCATION)
    if(DEFINED LCS_PYTHON_EXECUTABLE AND NOT LCS_PYTHON_EXECUTABLE STREQUAL "")
        find_package(Python COMPONENTS Interpreter Development.Module)
    else()
        find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    endif()

    if(Python_FOUND)
        set(LCS_PYTHON_EXECUTABLE "${Python_EXECUTABLE}" CACHE FILEPATH "Path to the Python executable" FORCE)
    endif()

    # 5. Handle Windows .exe suffix and verify existence
    if(WIN32 AND NOT EXISTS "${LCS_PYTHON_EXECUTABLE}")
        if(EXISTS "${LCS_PYTHON_EXECUTABLE}.exe")
            set(LCS_PYTHON_EXECUTABLE "${LCS_PYTHON_EXECUTABLE}.exe" CACHE FILEPATH "Path to the Python executable" FORCE)
        endif()
    endif()

    if(NOT EXISTS "${LCS_PYTHON_EXECUTABLE}")
        message(FATAL_ERROR "LCS_BUILD_PYBINDINGS is enabled, but no valid Python executable was found. Please activate a virtualenv (uv / conda / venv) or configure with -D LCS_PYTHON_EXECUTABLE=/path/to/python")
    endif()

    message(STATUS "[LCS] Using Python for bindings: ${LCS_PYTHON_EXECUTABLE} (${_CURRENT_ENV_SIG})")
endif()

# LuisaCompute backends
option(LUISA_COMPUTE_ENABLE_CUDA "Enable CUDA backend" ON)
option(LUISA_COMPUTE_ENABLE_METAL "Enable Metal backend" ON)
option(LUISA_COMPUTE_ENABLE_VULKAN "Enable Vulkan backend" OFF)
option(LUISA_COMPUTE_ENABLE_DX "Enable DirectX backend" OFF)
option(LUISA_COMPUTE_ENABLE_FALLBACK "Enable Fallback (CPU TBB) backend" OFF)

# Solver
file(GLOB_RECURSE LUISA_COMPUTE_SOLVER_LIB_SOURCES CONFIGURE_DEPENDS Solver/*.cpp Solver/*.cc)
file(GLOB_RECURSE LUISA_COMPUTE_SOLVER_LIB_HEADERS CONFIGURE_DEPENDS Solver/*.h CONFIGURE_DEPENDS Solver/*.hpp)
add_library(luisa-compute-solver-lib STATIC ${LUISA_COMPUTE_SOLVER_LIB_SOURCES} ${LUISA_COMPUTE_SOLVER_LIB_HEADERS})
set_target_properties(luisa-compute-solver-lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(luisa-compute-solver-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solver>)



# External
set(EXT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ext)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext)
include(${LuisaCompute_SOURCE_DIR}/scripts/setup_output_dirs.cmake)

# Rousource
set(LCSV_RESOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Resources)
target_compile_definitions(luisa-compute-solver-lib PUBLIC LCSV_RESOURCE_PATH="${LCSV_RESOURCE_PATH}")


# Executable application
if(LCS_BUILD_MAIN_APPLICATION)
    add_executable(app_simulation Application/app_simulation.cpp Application/app_simulation_demo_config.cpp Application/app_simulation_demo_config.h)
    add_executable(app_integration Application/app_integration.cpp)
    target_link_libraries(app_simulation PRIVATE
        luisa::compute
        luisa-compute-solver-lib
        )
    if (TARGET luisa-compute-solver-ext-yyjson)
        target_link_libraries(app_simulation PRIVATE luisa-compute-solver-ext-yyjson)
    endif ()
    target_link_libraries(app_integration PRIVATE 
        luisa::compute 
        luisa-compute-solver-lib
        )
    if (UNIX AND NOT APPLE)
        target_link_libraries(app_simulation PRIVATE atomic)
    endif()

    if (LCS_ENABLE_GUI)
        target_link_libraries(app_simulation PRIVATE polyscope)
        target_compile_definitions(app_simulation PRIVATE SIMULATION_APP_USE_GUI=1)
    endif()
endif()


if(LCS_ENABLE_TEST)
    add_subdirectory(UnitTest)
endif()

if(LCS_BUILD_PYBINDINGS)
    add_subdirectory(PythonBindings)
endif()
