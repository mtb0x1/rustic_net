//! # Rustic Net
//!
//! A high-performance, ergonomic, and extensible Machine Learning Accelerator (MLA) framework in Rust.
//! Built for both research and production use with a focus on performance and developer experience.
//!
//! ## Key Features
//!
//! - **Tensor Operations**: Efficient multi-dimensional array operations
//! - **Cross-Device Support**: Seamless CPU and GPU execution
//! - **Zero-Cost Abstractions**: Leverages Rust's type system for optimal performance
//! - **Thread-Safe**: Designed for concurrent execution
//! - **Minimal Dependencies**: Core functionality with minimal external dependencies
//! - **FFI Ready**: Easy integration with other languages
//!
//! ## Quick Start
//! ```rust
//! use rustic_net::tensor::{Tensor, Device};
//! use rustic_net::RusticNetInitTracingInit;
//!
//! // Initialize logging
//! RusticNetInitTracingInit();
//!
//! // Create and manipulate tensors
//! let t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], Device::default())?;
//! let t2 = t1.relu()?;
//! assert_eq!(t2.to_vec(), vec![1.0, 2.0, 3.0]);
//! # Ok::<(), String>(())
//! ```
//!
//! ## Feature Flags
//! - `parallel`: Enables multi-threaded execution (enabled by default)
//! - `cuda`: Enables CUDA support (requires CUDA toolkit)
//! - `wasm`: Enables WebAssembly support

#[cfg(feature = "parallel")]
pub mod parallel;

pub mod tensor;
pub(crate) mod tracing;

#[cfg(feature = "parallel")]
pub use parallel::{current_num_threads, init_thread_pool};

pub use tracing::init_tracing as RusticNetInitTracingInit;

/// Re-exports for common types
pub use tensor::{DType, Device, Shape, Tensor};
