#!/bin/bash
# Profile-Guided Optimization (PGO) build script for gradientlss
#
# This script builds an optimized version of the library using PGO.
# PGO can improve performance by 10-20% by optimizing based on actual usage patterns.
#
# Prerequisites:
#   - Rust toolchain with llvm-tools-preview: rustup component add llvm-tools-preview
#   - cargo-pgo (optional but recommended): cargo install cargo-pgo
#
# Usage:
#   ./scripts/pgo-build.sh          # Full PGO build
#   ./scripts/pgo-build.sh --bolt   # PGO + BOLT (Linux only, requires llvm-bolt)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PGO_DATA_DIR="/tmp/gradientlss-pgo-data"
TARGET=$(rustc -vV | sed -n 's|host: ||p')

echo "=== GradientLSS PGO Build ==="
echo "Project: $PROJECT_DIR"
echo "Target: $TARGET"
echo "PGO Data: $PGO_DATA_DIR"
echo ""

# Check if cargo-pgo is available
if command -v cargo-pgo &> /dev/null; then
    USE_CARGO_PGO=true
    echo "Using cargo-pgo for streamlined workflow"
else
    USE_CARGO_PGO=false
    echo "Using manual PGO workflow (install cargo-pgo for easier usage)"
fi

# Check for BOLT flag
USE_BOLT=false
if [[ "$1" == "--bolt" ]]; then
    if [[ "$(uname)" != "Linux" ]]; then
        echo "Error: BOLT is only supported on Linux"
        exit 1
    fi
    if ! command -v llvm-bolt &> /dev/null; then
        echo "Error: llvm-bolt not found. Install LLVM with BOLT support."
        exit 1
    fi
    USE_BOLT=true
    echo "BOLT optimization enabled"
fi

echo ""

cd "$PROJECT_DIR"

if $USE_CARGO_PGO; then
    # ========================================
    # cargo-pgo workflow (recommended)
    # ========================================

    echo "=== Step 1: Building instrumented binary ==="
    cargo pgo build

    echo ""
    echo "=== Step 2: Gathering profile data from benchmarks ==="
    cargo pgo bench -- --sample-size 20

    echo ""
    echo "=== Step 3: Building optimized binary ==="
    if $USE_BOLT; then
        cargo pgo bolt build --with-pgo
        echo ""
        echo "=== Step 4: Gathering BOLT profile data ==="
        # Run benchmarks with BOLT-instrumented binary
        ./target/$TARGET/release/*-bolt-instrumented || true
        cargo pgo bolt optimize --with-pgo
        echo ""
        echo "=== Build complete! ==="
        echo "Optimized binary: target/$TARGET/release/*-bolt-optimized"
    else
        cargo pgo optimize
        echo ""
        echo "=== Build complete! ==="
        echo "Optimized library: target/$TARGET/release/libgradientlss.*"
    fi

else
    # ========================================
    # Manual PGO workflow
    # ========================================

    # Clean previous PGO data
    rm -rf "$PGO_DATA_DIR"
    mkdir -p "$PGO_DATA_DIR"

    echo "=== Step 1: Building instrumented binary ==="
    RUSTFLAGS="-Cprofile-generate=$PGO_DATA_DIR" \
        cargo build --release --target="$TARGET"

    echo ""
    echo "=== Step 2: Gathering profile data ==="
    echo "Running benchmarks to collect profile data..."

    # Run benchmarks to generate profile data
    RUSTFLAGS="-Cprofile-generate=$PGO_DATA_DIR" \
        cargo bench --target="$TARGET" -- --sample-size 10 || true

    # Also run tests for additional coverage
    RUSTFLAGS="-Cprofile-generate=$PGO_DATA_DIR" \
        cargo test --release --target="$TARGET" || true

    echo ""
    echo "=== Step 3: Merging profile data ==="
    # Find llvm-profdata
    LLVM_PROFDATA=$(find ~/.rustup -name llvm-profdata -type f 2>/dev/null | head -1)
    if [[ -z "$LLVM_PROFDATA" ]]; then
        echo "Error: llvm-profdata not found. Run: rustup component add llvm-tools-preview"
        exit 1
    fi

    "$LLVM_PROFDATA" merge -o "$PGO_DATA_DIR/merged.profdata" "$PGO_DATA_DIR"

    echo ""
    echo "=== Step 4: Building optimized binary ==="
    RUSTFLAGS="-Cprofile-use=$PGO_DATA_DIR/merged.profdata -Cllvm-args=-pgo-warn-missing-function" \
        cargo build --release --target="$TARGET"

    echo ""
    echo "=== Build complete! ==="
    echo "Optimized library: target/$TARGET/release/libgradientlss.*"
fi

echo ""
echo "To use the PGO-optimized build, copy the library from the release directory"
echo "or run your application with: cargo run --release"
