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
//! - `ShapeOps`: Shape manipulation operations

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

/// Defines element-wise operations between a tensor and a scalar.
pub trait ScalarOps {
    /// Element-wise addition with a scalar
    fn add_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String>;

    /// Element-wise subtraction with a scalar
    fn sub_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String>;

    /// Element-wise multiplication with a scalar
    fn mul_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String>;

    /// Element-wise division with a scalar
    fn div_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String>;

    /// Element-wise addition with a scalar (reverse)
    fn r_add_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String>;

    /// Element-wise subtraction with a scalar (reverse)
    fn r_sub_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String>;

    /// Element-wise multiplication with a scalar (reverse)
    fn r_mul_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String>;

    /// Element-wise division with a scalar (reverse)
    fn r_div_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String>;
}

/// Defines tensor creation operations.
pub trait CreationOps {
    /// Creates a tensor with random values in [0, 1).
    fn random(shape: &[usize], device: crate::tensor::Device) -> Result<Tensor, String>;

    /// Creates a 1D tensor with values in the range [start, end).
    fn arange(start: f32, end: f32, device: crate::tensor::Device) -> Result<Tensor, String>;

    /// Creates a tensor filled with zeros.
    fn zeros(shape: &[usize], device: crate::tensor::Device) -> Result<Tensor, String>;

    /// Creates a tensor filled with ones.
    fn ones(shape: &[usize], device: crate::tensor::Device) -> Result<Tensor, String>;

    /// Creates an identity matrix (2D tensor with ones on the diagonal).
    fn identity(size: usize, device: crate::tensor::Device) -> Result<Tensor, String>;

    /// Creates a tensor from a vector with the specified shape and device.
    fn from_vec(
        data: Vec<f32>,
        shape: &[usize],
        device: crate::tensor::Device,
    ) -> Result<Tensor, String>;

    /// Creates a tensor from a slice with the specified shape.
    fn from_slice(
        slice: &[f32],
        shape: &[usize],
        device: crate::tensor::Device,
    ) -> Result<Tensor, String>;
}

/// Defines shape manipulation operations.
pub trait ShapeOps {
    /// Transposes the tensor by reversing its dimensions.
    fn transpose(tensor: &Tensor) -> Result<Tensor, String>;

    /// Transposes the tensor according to the given axes permutation.
    fn transpose_axes(tensor: &Tensor, axes: &[usize]) -> Result<Tensor, String>;
}
