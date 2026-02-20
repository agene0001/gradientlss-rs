# Performance Optimization Guide

This document describes the performance optimizations available in gradientlss-rs and how to enable them.

## Quick Start

For maximum performance, build with:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Optimization Techniques

### 1. Compiler Optimizations (Cargo.toml)

The release profile is configured with:
- `opt-level = 3` - Maximum optimization
- `lto = "fat"` - Link-time optimization across all crates
- `codegen-units = 1` - Better optimization (slower compile)

### 2. SIMD Vectorization

The `simd_ops` module provides SIMD-accelerated versions of response functions:
- `exp_simd` - Exponential function
- `sigmoid_simd` - Sigmoid activation
- `softplus_simd` - Softplus activation
- `squareplus_simd` - Squareplus activation
- `relu_simd` - ReLU activation

These use the `wide` crate for portable SIMD on stable Rust. To maximize SIMD performance:

```bash
# Enable native CPU features (AVX2/AVX-512 on modern x86)
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### 3. Parallel Processing

The library uses `rayon` for parallelization:
- Cross-validation folds run in parallel
- Gradient computation parallelizes across samples (for n > 256)

Control thread count via:
```bash
export RAYON_NUM_THREADS=8
```

### 4. Memory Optimizations

- Pre-allocated buffers for gradients/hessians in training loops
- In-place response function transformations (`apply_into`)
- Buffer reuse in numerical differentiation

### 5. Cached Constants

Mathematical constants like `ln(2*pi)` are pre-computed in the `constants` module to avoid repeated computation.

## Profile-Guided Optimization (PGO)

PGO can provide an additional 10-20% performance improvement by optimizing based on actual usage patterns.

### Using cargo-pgo (Recommended)

```bash
# Install prerequisites
cargo install cargo-pgo
rustup component add llvm-tools-preview

# Build with PGO
cargo pgo build
cargo pgo bench
cargo pgo optimize
```

### Using the PGO Script

```bash
./scripts/pgo-build.sh
```

### Manual PGO

```bash
# Step 1: Build instrumented binary
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" \
    cargo build --release --target=x86_64-unknown-linux-gnu

# Step 2: Run workloads
cargo bench

# Step 3: Merge profile data
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data

# Step 4: Build optimized binary
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" \
    cargo build --release --target=x86_64-unknown-linux-gnu
```

## BOLT Post-Link Optimization (Linux Only)

BOLT provides additional 2-5% performance on top of PGO:

```bash
# Requires llvm-bolt
./scripts/pgo-build.sh --bolt
```

## Benchmarking

Run benchmarks to measure performance:

```bash
# All benchmarks
cargo bench

# Specific benchmark
cargo bench --bench distribution_ops

# Quick benchmark (fewer samples)
cargo bench -- --sample-size 10
```

## Performance Tips

1. **Use release builds** - Debug builds are 10-100x slower
2. **Enable native CPU features** - Use `-C target-cpu=native`
3. **Use LTO** - Already enabled in release profile
4. **Consider PGO** - For production deployments
5. **Profile first** - Use `cargo flamegraph` to find bottlenecks

## Expected Performance

With all optimizations enabled:
- Gradient computation: ~50 Melem/s at 10k samples
- Transform params: 70-150 Melem/s depending on distribution
- SIMD provides ~2-4x speedup for response functions
- PGO adds ~10-15% improvement
- BOLT adds ~2-5% on top of PGO
