//! Traits defining the interface for tensor operations
//!
//! This module contains traits that define the interface for various tensor operations.
//! Backend implementations must implement these traits to provide the actual functionality.

use super::super::{Device, DType, Shape, Tensor};

/// Trait for tensor creation operations
pub trait CreateTensor {
    /// Creates a new tensor with the given data, shape, device, and data type
    fn create_tensor(data: Vec<f32>, shape: Shape, device: Device, dtype: DType) -> Tensor;
}

/// Trait for unary operations (operations on a single tensor)
pub trait UnaryOps {
    /// Applies the ReLU activation function element-wise
    fn relu(tensor: &Tensor) -> Result<Tensor, String>;
    
    // Add other unary operations here as needed
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

/// Trait for shape manipulation operations
pub trait ShapeOps {
    /// Reshape the tensor to a new shape
    fn reshape(tensor: &Tensor, new_shape: &[usize]) -> Result<Tensor, String>;
    
    /// Transpose the tensor according to the given axes
    fn transpose(tensor: &Tensor, axes: Option<&[usize]>) -> Result<Tensor, String>;
    
    /// Add a new dimension of size 1 at the specified axis
    fn expand_dims(tensor: &Tensor, axis: usize) -> Result<Tensor, String>;
    
    /// Remove dimensions of size 1
    fn squeeze(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String>;
}

/// Trait for random number generation
pub trait RandomOps {
    /// Create a tensor with random values between 0.0 and 1.0
    fn random(shape: &[usize], device: Device) -> Tensor;
}

/// Trait for utility operations
pub trait UtilOps {
    /// Create an identity matrix of the given size
    fn identity(size: usize, device: Device) -> Tensor;
    
    /// Create a 1D tensor with values from start to end (exclusive) with step size 1
    fn arange(start: f32, end: f32, device: Device) -> Tensor;
    
    /// Move the tensor to the specified device
    fn to_device(tensor: &Tensor, device: Device) -> Result<Tensor, String>;
}
