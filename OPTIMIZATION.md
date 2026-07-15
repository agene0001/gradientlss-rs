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

### 2. Auto-Vectorization

The hot paths (fused response-function derivatives in `utils.rs`, the batch
transform passes) are written as contiguous single-pass loops that LLVM
auto-vectorizes. To let the compiler use the full instruction set of the
build machine:

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

### 4. Per-Round Train Metric (off by default)

With a validation set driving early stopping, the per-round training-set NLL is
pure reporting — and it costs a full pass over the training set every boosting
round. Skipping it measured ~20% faster end-to-end training (NB + XGBoost,
50k×20, validation set, quiet; see `examples/bench_collect_train_metrics.rs`),
so it is **off by default**. Opt in when you want per-round train curves in
`TrainingResult::train_history`:

```rust
let config = TrainConfig {
    collect_train_metrics: true,
    ..TrainConfig::default()
};
```

It is computed regardless whenever something still needs it: no validation set
(train loss drives early stopping), `verbose: true`, or registered callbacks.

### 5. Memory Optimizations

- Pre-allocated buffers for gradients/hessians in training loops
- In-place response function transformations (`apply_into`)
- Buffer reuse in numerical differentiation

### 6. Cached Constants

Mathematical constants like `ln(2*pi)` are pre-computed in the `constants` module to avoid repeated computation.

### 7. XGBoost Backend: What's Built In vs. What You Tune

The XGBoost backend already applies the wrapper-level optimizations from
rust-xgboost's `docs/SERVING.md` — you get these for free:

- **QuantileDMatrix training** (tree_method=hist, the default): the training
  matrix stores pre-binned values (~1 byte per feature value instead of 4 —
  roughly 4x less training-matrix memory) and skips the sketching pass at the
  start of training. `max_bin` is read from your params so binning always
  matches the booster. Any non-hist tree_method falls back to a plain DMatrix.
- **Allocation-free training loop**: gradients feed straight into
  `Booster::boost` (no callback trampoline, no redundant internal predict per
  round), and per-round predictions reuse one buffer.
- **Inplace prediction**: `predict` skips DMatrix construction entirely when no
  early-stopping truncation applies — this dominates small-batch latency.
- **Post-training cache release** (`booster.reset()`): XGBoost's gradient
  buffers and training prediction caches are freed once training completes.

Two knobs are deliberately left to you because they are workload/machine
decisions:

**a. Native codegen for the bundled libxgboost C++.** All tree building and
prediction time is inside libxgboost, which the `xgb` crate compiles from
source. Two env vars tune that build:

```toml
# .cargo/config.toml (machine-local, NOT committed — a -march=native build
# crashes with an illegal-instruction fault on older CPUs)
[env]
XGB_BUILD_NATIVE = "1"   # -march=native / -mcpu=native for libxgboost
XGB_BUILD_IPO = "1"      # link-time optimization for libxgboost
```

Pin them in `.cargo/config.toml` rather than exporting ad hoc: the build script
watches these vars, so setting them on one cargo command and not the next
silently rebuilds the whole C++ library back to the default configuration.

**b. `nthread` for small-batch serving.** Small-batch prediction latency is
dominated by OpenMP thread dispatch, not tree traversal — rust-xgboost measured
`nthread=1` ~11x faster for 1 row, ~5x for 16 rows, ~2x for 100 rows, with
multithreading winning again above roughly 1000 rows per call. gradientlss does
not set this (training wants all cores, and it can't know your serving batch
size); for latency-sensitive small-batch serving in the same process, set it as
a training param — it stays on the live booster for subsequent predicts:

```rust
params.set("nthread", ParamValue::Int(1)); // serving-oriented models only
```

Note XGBoost does not persist `nthread` through save/load, so a loaded model
predicts with all cores again; there is currently no post-load knob for it.

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
- PGO adds ~10-15% improvement
- BOLT adds ~2-5% on top of PGO
