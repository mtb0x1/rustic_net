# Rustic Net

**🚧 Experimental Machine Learning Framework 🚧**

Crate documentation: [rustic_net](https://mtb0x1.github.io/rustic_net/rustic_net/index.html)

## ⚠️ Disclaimer
This project is intentionally experimental and vibe coded (with variety of tools and models). It's not intended for production use. The goal is to explore, play, and push boundaries within machine learning and artificial intelligence frameworks.

## 🎯 Purpose
Rustic Net is a playground for exploring unconventional approaches to deep learning and neural networks. Expect:
- Unconventional behavior
- Experimental features
- Breaking changes
- Raw, unfiltered exploration

## 🚀 Getting Started

### Prerequisites
- Rust (latest stable version recommended)
- Cargo (Rust's package manager)

### Installation
```bash
git clone https://github.com/yourusername/rustic_net.git
cd rustic_net
cargo build
```

## 🧪 Features
- **Parallel Processing**: Leverage multi-core CPUs for faster tensor operations.
- **SIMD Acceleration**: Utilizes SIMD instructions for enhanced performance on element-wise operations. **Requires a nightly Rust toolchain.**
- Experimental neural network implementations
- Research-focused architecture
- Cutting-edge (and sometimes unstable) features
- Playground for AI/ML research

## 🚀 Performance

Rustic Net includes parallel implementations of key operations to maximize performance on multi-core systems:

- Parallel tensor operations using Rayon for data parallelism
- Automatic thread pool management
- Optimized for both small and large tensors

### Enabling Parallel Processing

Parallel features are enabled by default. You can control the number of threads using the `RUSTIC_NET_NUM_THREADS` environment variable:

```bash
# Use 4 threads for parallel operations
RUSTIC_NET_NUM_THREADS=4 cargo run --example tensor_operations

# Use all available CPU cores
RUSTIC_NET_NUM_THREADS=0 cargo run --example tensor_operations
```

## 📊 Benchmarking

Bench results below are based on `examples/tensor_operations.rs` with 10 threads and Matrix size of 100000 [1000,1000]:

```markdown
| Command | Mean [s] | Min [s] | Max [s] | Relative |
|:---|---:|---:|---:|---:|
| `tensor_operations_default.exe` | 1.648 ± 0.095 | 1.561 | 1.827 | 6.29 ± 2.08 |
| `tensor_operations_simd.exe` | 0.700 ± 0.063 | 0.653 | 0.844 | 2.67 ± 0.90 |
| `tensor_operations_parallel.exe` | 0.895 ± 0.072 | 0.842 | 1.042 | 3.41 ± 1.14 |
| `tensor_operations_simd_parallel.exe` | 0.262 ± 0.085 | 0.191 | 0.480 | 1.00 |
```

> **Note:** These benchmark results are automatically updated. For the most recent results, check [BENCHMARKS.md](BENCHMARKS.md).

## 📂 Project Structure
```
rustic_net/
├── src/           # Core source code
│   ├── dtype.rs   # Data types and operations
│   ├── tensor/    # Tensor operations
│   │   ├── mod.rs
│   │   ├── ops/   # Parallel operations
│   │   └── ...
│   └── parallel/  # Parallel processing utilities
├── examples/      # Example code
├── benches/       # Benchmarks
├── tests/         # Tests
└── Cargo.toml     # Project manifest
```

## 🤝 Contributing
This is an experimental project where we embrace breaking things in the name of learning. If you want to contribute:
1. Fork the repository
2. Create your feature branch (`git checkout -b experiment/your-idea`)
3. Commit your changes (`git commit -am 'Add some experimental feature'`)
4. Push to the branch (`git push origin experiment/your-idea`)
5. Open a Pull Request

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔥 Warning
Expect the unexpected. This project is a sandbox for wild ideas and unconventional approaches. Dive deep. Break things. Learn hard. Have fun.
