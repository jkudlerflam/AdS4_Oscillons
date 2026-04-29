#!/bin/bash
# Build and run all tests for AdS4 Oscillon solver
# Works both locally and on cluster (uses cmake if available, falls back to manual build)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$BASE_DIR/build}"
SRC_DIR="$BASE_DIR/src"
TEST_DIR="$BASE_DIR/tests"

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O3 -march=native -DNDEBUG"
INCFLAG="-I$SRC_DIR"

# Detect OpenMP
OMP_FLAGS=""
if $CXX -fopenmp -x c++ -c /dev/null -o /dev/null 2>/dev/null; then
    OMP_FLAGS="-fopenmp -DUSE_OPENMP"
    echo "OpenMP: enabled"
else
    echo "OpenMP: not available (serial build)"
fi

# Detect LAPACK
LAPACK_FLAGS=""
LAPACK_LIBS=""
for lib in /usr/lib/*/liblapack.so* /usr/lib/liblapack.so*; do
    if [ -f "$lib" ]; then
        LAPACK_LIBS="-llapack -lblas -lgfortran"
        LAPACK_FLAGS="-DHAVE_LAPACK"
        echo "LAPACK: found"
        break
    fi
done
# Also check via pkg-config
if [ -z "$LAPACK_LIBS" ] && pkg-config --exists lapack 2>/dev/null; then
    LAPACK_LIBS="$(pkg-config --libs lapack) -lgfortran"
    LAPACK_FLAGS="-DHAVE_LAPACK"
    echo "LAPACK: found (pkg-config)"
fi
if [ -z "$LAPACK_LIBS" ]; then
    echo "LAPACK: not found (GMRES fallback)"
fi

mkdir -p "$BUILD_DIR"

echo
echo "=== Building AdS4 Oscillon Solver ==="
echo "Compiler: $CXX"
echo "Flags:    $CXXFLAGS $OMP_FLAGS $LAPACK_FLAGS"
echo "Base dir: $BASE_DIR"
echo

# All source files
SOURCES=(
    "$SRC_DIR/spectral/chebyshev.cpp"
    "$SRC_DIR/spectral/legendre.cpp"
    "$SRC_DIR/spectral/fourier.cpp"
    "$SRC_DIR/geometry/ads4.cpp"
    "$SRC_DIR/geometry/adm.cpp"
    "$SRC_DIR/geometry/equations.cpp"
    "$SRC_DIR/solver/field3d.cpp"
    "$SRC_DIR/solver/newton.cpp"
    "$SRC_DIR/solver/gmres.cpp"
    "$SRC_DIR/solver/oscillon_system.cpp"
    "$SRC_DIR/solver/oscillon_driver.cpp"
    "$SRC_DIR/diagnostics/diagnostics.cpp"
    "$SRC_DIR/diagnostics/charges.cpp"
    "$SRC_DIR/io/output.cpp"
)

# Build tests
TESTS=(test_spectral test_background test_adm test_equations test_solver test_diagnostics test_assembler test_newton_oscillon)
PASS=0
FAIL=0

for test in "${TESTS[@]}"; do
    echo "--- Building $test ---"
    "$CXX" $CXXFLAGS $OMP_FLAGS $LAPACK_FLAGS "$INCFLAG" \
      "${SOURCES[@]}" \
      "$TEST_DIR/$test.cpp" \
      -o "$BUILD_DIR/$test" -lm $LAPACK_LIBS $OMP_FLAGS
    echo "--- Running $test ---"
    if timeout 300 "$BUILD_DIR/$test"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
    echo
done

# Build production CLI
echo "--- Building ads4osc_cli ---"
"$CXX" $CXXFLAGS $OMP_FLAGS $LAPACK_FLAGS "$INCFLAG" \
  "${SOURCES[@]}" \
  "$SRC_DIR/main.cpp" \
  -o "$BUILD_DIR/ads4osc_cli" -lm $LAPACK_LIBS $OMP_FLAGS
echo "Built: $BUILD_DIR/ads4osc_cli"
echo

echo "==============================="
echo "Test suites: $PASS passed, $FAIL failed out of ${#TESTS[@]}"
echo "==============================="

if [ $FAIL -gt 0 ]; then
    exit 1
fi
