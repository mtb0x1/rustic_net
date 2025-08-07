//! # Tensor Core
//!
//! Core tensor type and operations for Rustic Net, providing efficient
//! multi-dimensional array operations with CPU and GPU support.
//!
//! ## Key Features
//! - **N-dimensional arrays**: Support for tensors of arbitrary rank
//! - **Device-agnostic API**: Seamless CPU/GPU execution with the same interface
//! - **Efficient memory layout**: Row-major order with configurable strides
//! - **Automatic broadcasting**: Operations on tensors of different shapes
//! - **View semantics**: Zero-copy operations like reshape and transpose
//! - **Comprehensive operations**: Linear algebra, element-wise ops, reductions
//! - **Thread-safe**: Designed for concurrent use across threads
//!
//! ## Quick Start
//! ```rust
//! use rustic_net::tensor::{Tensor, Device};
//!
//! // Create tensors
//! let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default())?;
//! let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], Device::default())?;
//!
//! // Perform operations
//! let c = a.matmul(&b)?;  // Matrix multiplication
//! let d = c.relu()?;      // ReLU activation
//! let e = d.sum(Some(0))?; // Sum along first dimension
//! # Ok::<(), String>(())
//! ```
//!
//! ## Design Philosophy
//! - **Performance**: Optimized for both small and large tensors
//! - **Safety**: Compile-time and runtime checks for valid operations
//! - **Ergonomics**: Intuitive API with sensible defaults
//! - **Extensibility**: Easy to add new operations and backends
//!
//! ## Memory Management
//! - Tensors use `Arc` for reference counting
//! - Data is shared between tensor views
//! - Explicit `.clone()` is required for deep copies
//!
//! ## Device Support
//! - **CPU**: Highly optimized for all tensor operations
//! - **CUDA**: GPU acceleration (when `cuda` feature is enabled)
//! - **WebGPU**: Browser-based GPU acceleration (when `wasm` feature is enabled)

// Core tensor modules
pub mod backends;
pub mod device;
pub mod dtype;
pub mod math_operations;
pub mod shape;
#[allow(clippy::module_inception)]
pub mod tensor;

// Re-export common traits
pub use backends::traits::*;

// Re-export debug and tracing utilities
pub use tracing::*;

// Parallel processing utilities
#[cfg(any(feature = "parallel", feature = "simd_and_parallel"))]
pub use crate::tensor::backends::utils::{
    current_num_threads, init_thread_pool, recommended_chunk_size,
};

// Common type aliases and constants
/// A result type for tensor operations
pub type Result<T> = std::result::Result<T, String>;

// Re-export the Shape type for backward compatibility
/// A shape type for tensors, representing the dimensions of a tensor.
pub type Shape = shape::Shape;

// Re-export the tensor module's public API
#[doc(inline)]
pub use tensor::Tensor;

// Re-export the device module's public API
#[doc(inline)]
pub use device::Device;

// Re-export the dtype module's public API
#[doc(inline)]
pub use dtype::DType;
