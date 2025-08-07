#![feature(portable_simd)]
//! # Rustic Net
//!
//! A high-performance, ergonomic, and extensible Machine Learning Accelerator (MLA) framework in Rust.
//! Built for both research and production use with a focus on performance and developer experience.
//!
//! ## Key Features
//!
//! - **Tensor Operations**: Efficient multi-dimensional array operations with support for various data types
//! - **Cross-Device Support**: Seamless CPU and GPU execution with a unified API
//! - **Zero-Cost Abstractions**: Leverages Rust's type system and ownership model for optimal performance
//! - **Thread-Safe**: Designed for concurrent execution with minimal synchronization overhead
//! - **Minimal Dependencies**: Core functionality with minimal external dependencies
//! - **FFI Ready**: Easy integration with other languages through C-compatible interfaces
//! - **SIMD Acceleration**: Utilizes CPU vector instructions for maximum performance
//! - **Parallel Processing**: Multi-threaded execution for CPU-bound operations
//!
//! ## Quick Start
//!
//! Add this to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! rustic_net = { version = "0.1", features = ["parallel"] }
//! ```
//!
//! Basic usage:
//! ```rust
//! use rustic_net::tensor::{Tensor, Device};
//! use rustic_net::RusticNetInitTracing;
//!
//! // Initialize logging (optional but recommended for debugging)
//! RusticNetInitTracing();
//!
//! // Create and manipulate tensors
//! let t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], Device::default())?;
//! let t2 = t1.relu()?;
//! assert_eq!(t2.to_vec(), &vec![1.0, 2.0, 3.0]);
//! # Ok::<(), String>(())
//! ```
//!
//! ## Feature Flags
//!
//! The following features can be enabled in your `Cargo.toml`:
//!
//! - `parallel` (enabled by default): Enables multi-threaded execution using Rayon for parallel processing
//! - `cuda`: Enables CUDA support for GPU acceleration (requires CUDA toolkit)
//! - `wasm`: Enables WebAssembly support for running in browsers and other WASM environments
//! - `simd`: Enables SIMD acceleration for CPU operations (enabled by default on supported platforms)
//!
//! ## Performance Considerations
//!
//! - For best performance, enable the `parallel` feature and ensure your tensors are large enough to
//!   benefit from parallel processing.
//! - When working with CUDA, ensure your tensors are large enough to overcome the overhead of
//!   transferring data to and from the GPU.
//! - The library uses 32-bit floating-point numbers (`f32`) by default for optimal performance on most hardware.
//!
//! ## Error Handling
//!
//! Most operations return a `Result<T, String>` where errors are returned as human-readable strings.
//! It's recommended to use the `?` operator for ergonomic error handling.
//!
//! ## Thread Safety
//!
//! All public types in this crate are `Send` and `Sync`, making them safe to use across thread boundaries.
//! The library manages thread pools internally when the `parallel` feature is enabled.

pub mod tensor;
pub(crate) mod tracing;

#[cfg(any(feature = "parallel", feature = "simd_and_parallel"))]
pub use crate::tensor::{current_num_threads, init_thread_pool, recommended_chunk_size};

pub use tracing::init_tracing as RusticNetInitTracing;
pub use tracing::init_tracing_with as RusticNetInitTracingWith;

/// Re-exports for common types
pub use tensor::{DType, Device, Shape, Tensor};
