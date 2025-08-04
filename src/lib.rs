//! # Rustic Net
//!
//! A high-performance, ergonomic, and extensible Machine Learning Accelerator (MLA) framework in Rust.
//!
//! ## Features
//! - **Tensor Operations**: Multi-dimensional array operations with CPU and GPU support
//! - **Device Management**: Seamless tensor movement between different compute devices
//! - **No External Dependencies**: Core tensor operations implemented in pure Rust
//! - **FFI Compatible**: Designed for easy integration with other languages
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

pub mod tensor;
pub(crate) mod tracing;
pub use tracing::init_tracing as RusticNetInitTracingInit;

/// Re-exports for common types
pub use tensor::{DType, Device, Shape, Tensor};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::tensor::{DType, Device, Shape, Tensor};
}
