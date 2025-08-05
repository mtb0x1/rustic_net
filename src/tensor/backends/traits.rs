//! # Backend Operation Traits
//!
//! Defines the core tensor operation interfaces that must be implemented by each backend.
//! These traits provide a unified API across different hardware and execution models.
//!
//! ## Trait Hierarchy
//! - `UnaryOps`: Element-wise operations on single tensors
//! - `BinaryElementwiseOps`: Element-wise operations between two tensors
//! - `MatOps`: Matrix and linear algebra operations
//! - `ReductionOps`: Dimensionality reduction operations

use crate::tensor::Tensor;

/// Defines element-wise operations that operate on a single tensor.
///
/// # Contract
/// Implementations must preserve the input tensor's shape and device placement.
pub trait UnaryOps {
    /// Applies the ReLU activation function element-wise
    fn relu(tensor: &Tensor) -> Result<Tensor, String>;
}

/// Defines element-wise operations between two tensors.
///
/// # Contract
/// - Input tensors must be broadcastable to the same shape
/// - Output tensor shape matches the broadcasted input shapes
/// - Both input tensors must be on the same device
pub trait BinaryElementwiseOps {
    /// Element-wise addition
    fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, String>;

    /// Element-wise subtraction
    fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor, String>;

    /// Element-wise multiplication
    fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, String>;

    /// Element-wise division
    fn div(a: &Tensor, b: &Tensor) -> Result<Tensor, String>;
}

/// Defines matrix and linear algebra operations.
///
/// # Contract
/// - Input tensors must be 2D (matrices)
/// - Inner dimensions must be compatible for matrix multiplication
pub trait MatOps {
    /// Matrix multiplication
    fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String>;
}

/// Defines operations that reduce tensor dimensions.
///
/// # Contract
/// - When `axis` is `None`, reduces all dimensions to a scalar
/// - When `axis` is `Some(dim)`, reduces along the specified dimension
/// - Preserves other dimensions unless reduced
pub trait ReductionOps {
    /// Sum of tensor elements, optionally along an axis
    fn sum(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String>;

    /// Mean of tensor elements, optionally along an axis
    fn mean(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String>;

    /// Maximum value, optionally along an axis
    fn max(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String>;

    /// Minimum value, optionally along an axis
    fn min(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String>;

    /// Index of maximum value, optionally along an axis
    fn argmax(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String>;

    /// Index of minimum value, optionally along an axis
    fn argmin(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String>;
}
