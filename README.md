# Rustic Net

**🚧 Experimental Machine Learning Framework 🚧**

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
- **Parallel Processing**: Leverage multi-core CPUs for faster tensor operations
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

Parallel features are enabled by default. You can control the number of threads using the `RAYON_NUM_THREADS` environment variable:

```bash
# Use 4 threads for parallel operations
RAYON_NUM_THREADS=4 cargo run --example tensor_operations

# Use all available CPU cores
RAYON_NUM_THREADS=0 cargo run --example tensor_operations
```

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
