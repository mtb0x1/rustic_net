//! # Rustic Net
//!
//! A high-performance, ergonomic, and extensible Machine Learning Accelerator (MLA) framework in Rust.
//!
//! ## Features
//! - **Tensor Operations**: Multi-dimensional array operations with CPU and GPU support
//! - **Device Management**: Seamless tensor movement between different compute devices
//! - **No External Dependencies**: Core tensor operations implemented in pure Rust
//! - **FFI Compatible**: Designed for easy integration with other languages
//! - **Parallel Processing**: Automatic CPU parallelization with configurable thread pool
//!
//! ## Example
//! ```rust
//! use rustic_net::tensor::{Tensor, Device};
//!
//! // Create a tensor on CPU
//! let t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], Device::Cpu(None)).unwrap();
//!
//! // Perform operations
//! let t2 = t1.relu().unwrap();
//!
//! // Convert to vector
//! let result = t2.to_vec();
//! assert_eq!(result, vec![1.0, 2.0, 3.0]);
//! ```

#[cfg(feature = "parallel")]
pub mod parallel;

pub mod tensor;
pub(crate) mod tracing;

#[cfg(feature = "parallel")]
pub use parallel::{current_num_threads, init_thread_pool};

pub use tracing::init_tracing as RusticNetInitTracingInit;

/// Re-exports for common types
pub use tensor::{DType, Device, Shape, Tensor};
