#!/bin/bash
# Build AdS4 Oscillon solver on IAS Typhon cluster
# Usage: bash scripts/build_typhon.sh [Release|Debug]
set -e

BUILD_TYPE="${1:-Release}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$BASE_DIR/build_typhon"

echo "=== Building on Typhon ==="
echo "Build type: $BUILD_TYPE"
echo

# Load modules — adjust these to match what's actually available on Typhon.
# Run 'module avail' on the login node to see exact names.
# Common patterns on CentOS/RHEL HPC:
module purge 2>/dev/null || true
module load gcc/10.2.0 2>/dev/null || module load gcc 2>/dev/null || true
module load cmake/3.20 2>/dev/null || module load cmake 2>/dev/null || true
module load lapack 2>/dev/null || module load openblas 2>/dev/null || module load mkl 2>/dev/null || true

echo "gcc:    $(gcc --version | head -1)"
echo "cmake:  $(cmake --version | head -1)"
echo

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
cmake "$BASE_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CXX_COMPILER=g++

# Build (use all available cores on login node for compilation)
make -j$(nproc) ads4osc_cli

echo
echo "=== Build complete ==="
echo "Executable: $BUILD_DIR/ads4osc_cli"
echo

# Optional: build and run tests
read -p "Run tests? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    make -j$(nproc)
    ctest --output-on-failure
fi
