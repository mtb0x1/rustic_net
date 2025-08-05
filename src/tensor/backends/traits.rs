//! Traits defining the interface for tensor operations
//!
//! This module contains traits that define the interface for various tensor operations.
//! Backend implementations must implement these traits to provide the actual functionality.

use crate::tensor::Tensor;

/// Trait for unary operations (operations on a single tensor)
pub trait UnaryOps {
    /// Applies the ReLU activation function element-wise
    fn relu(tensor: &Tensor) -> Result<Tensor, String>;
}

/// Trait for binary element-wise operations (operations between two tensors)
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

/// Trait for matrix operations
pub trait MatOps {
    /// Matrix multiplication
    fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String>;
}

/// Trait for reduction operations
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
