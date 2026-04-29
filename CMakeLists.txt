cmake_minimum_required(VERSION 3.14)
project(AdS4Oscillons LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address -DDEBUG")

# Find Eigen (header-only) — optional, not currently used in source
find_package(Eigen3 3.3 QUIET)

# Optional: OpenMP
find_package(OpenMP QUIET)

# Optional: FFTW
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFTW fftw3)
endif()

# Core library
add_library(ads4osc STATIC
    src/spectral/chebyshev.cpp
    src/spectral/legendre.cpp
    src/spectral/fourier.cpp
    src/geometry/ads4.cpp
    src/geometry/adm.cpp
    src/geometry/equations.cpp
    src/solver/field3d.cpp
    src/solver/newton.cpp
    src/solver/gmres.cpp
    src/solver/oscillon_system.cpp
    src/solver/oscillon_driver.cpp
    src/diagnostics/diagnostics.cpp
    src/diagnostics/charges.cpp
    src/io/output.cpp
)

target_include_directories(ads4osc PUBLIC ${CMAKE_SOURCE_DIR}/src)

# LAPACK/BLAS for Newton solver (SVD least-squares for handling Bianchi null space)
# Strategy: try CMake's built-in finder first (works with module load, MKL, etc.),
# then fall back to hardcoded Ubuntu paths.
set(LAPACK_FOUND FALSE)

# Method 1: CMake's find_package (respects module-loaded paths, MKL, etc.)
find_package(LAPACK QUIET)
if(LAPACK_FOUND)
    target_link_libraries(ads4osc PUBLIC ${LAPACK_LIBRARIES})
    target_compile_definitions(ads4osc PUBLIC HAVE_LAPACK)
    message(STATUS "Found LAPACK via find_package: ${LAPACK_LIBRARIES}")
else()
    # Method 2: Hardcoded Ubuntu/Debian paths
    foreach(_arch aarch64-linux-gnu x86_64-linux-gnu)
        if(EXISTS "/usr/lib/${_arch}/liblapack.so.3" AND EXISTS "/usr/lib/${_arch}/libblas.so.3")
            target_link_libraries(ads4osc PUBLIC
                /usr/lib/${_arch}/liblapack.so.3
                /usr/lib/${_arch}/libblas.so.3)
            if(EXISTS "/usr/lib/${_arch}/libgfortran.so.5")
                target_link_libraries(ads4osc PUBLIC /usr/lib/${_arch}/libgfortran.so.5)
            endif()
            target_compile_definitions(ads4osc PUBLIC HAVE_LAPACK)
            set(LAPACK_FOUND TRUE)
            message(STATUS "Found LAPACK: /usr/lib/${_arch}/liblapack.so.3")
            break()
        endif()
    endforeach()
    if(NOT LAPACK_FOUND)
        message(STATUS "LAPACK not found — SVD solver disabled, arclength continuation will not work")
    endif()
endif()

if(Eigen3_FOUND)
    target_link_libraries(ads4osc PUBLIC Eigen3::Eigen)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(ads4osc PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(ads4osc PUBLIC USE_OPENMP)
endif()
if(FFTW_FOUND)
    target_link_libraries(ads4osc PUBLIC ${FFTW_LIBRARIES})
    target_include_directories(ads4osc PUBLIC ${FFTW_INCLUDE_DIRS})
    target_compile_definitions(ads4osc PUBLIC USE_FFTW)
endif()

# Main executable — production CLI
add_executable(ads4osc_cli src/main.cpp)
target_link_libraries(ads4osc_cli ads4osc)

# Legacy alias
add_executable(run_benchmark src/main.cpp)
target_link_libraries(run_benchmark ads4osc)

# Tests
enable_testing()

add_executable(test_spectral tests/test_spectral.cpp)
target_link_libraries(test_spectral ads4osc)
add_test(NAME spectral COMMAND test_spectral)

add_executable(test_background tests/test_background.cpp)
target_link_libraries(test_background ads4osc)
add_test(NAME background COMMAND test_background)

add_executable(test_adm tests/test_adm.cpp)
target_link_libraries(test_adm ads4osc)
add_test(NAME adm COMMAND test_adm)

add_executable(test_equations tests/test_equations.cpp)
target_link_libraries(test_equations ads4osc)
add_test(NAME equations COMMAND test_equations)

add_executable(test_solver tests/test_solver.cpp)
target_link_libraries(test_solver ads4osc)
add_test(NAME solver COMMAND test_solver)

add_executable(test_diagnostics tests/test_diagnostics.cpp)
target_link_libraries(test_diagnostics ads4osc)
add_test(NAME diagnostics COMMAND test_diagnostics)

add_executable(test_assembler tests/test_assembler.cpp)
target_link_libraries(test_assembler ads4osc)
add_test(NAME assembler COMMAND test_assembler)

add_executable(test_newton_oscillon tests/test_newton_oscillon.cpp)
target_link_libraries(test_newton_oscillon ads4osc)
add_test(NAME newton_oscillon COMMAND test_newton_oscillon)

# Run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_spectral test_background test_adm test_equations test_solver test_diagnostics test_assembler test_newton_oscillon
)
