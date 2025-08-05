//! # Tensor Core
//!
//! Core tensor type and operations for Rustic Net, providing efficient
//! multi-dimensional array operations with CPU and GPU support.
//!
//! ## Key Features
//! - N-dimensional tensor operations
//! - Automatic differentiation
//! - Device-agnostic API (CPU/GPU)
//! - Memory-efficient views and in-place operations
//! - Broadcasting and strided operations

use crate::trace_fn;
use std::fmt;
use std::sync::Arc;

pub mod backends;
pub mod creation;
pub mod impl_ops;
pub mod shape;

use backends::traits::{BinaryElementwiseOps, MatOps, ReductionOps, UnaryOps};
pub use creation::*;
pub use shape::Shape;
pub use tracing::debug;

#[cfg(feature = "parallel")]
use backends::cpu_par::CpuParallel;
#[cfg(not(feature = "parallel"))]
use backends::cpu_seq::CpuSequential;

#[cfg(feature = "parallel")]
pub use crate::parallel;

/// Compute device for tensor storage and operations.
///
/// Tensors can be allocated on different devices, with operations
/// automatically dispatched to the appropriate backend.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Device {
    /// CPU device with optional device ID (useful for multi-CPU systems)
    Cpu(Option<usize>),
    /// CUDA device with device ID
    Cuda(usize),
    /// WebGPU device with device ID
    WebGpu(usize),
}

impl Default for Device {
    fn default() -> Self {
        trace_fn!("Device::default");
        debug!("Creating default CPU device with ID 0");
        // Default to CPU device with ID 0
        Device::Cpu(Some(0))
    }
}

/// Numeric type of tensor elements.
///
/// Currently supports 32-bit floating point (f32).
/// Future versions may add support for other numeric types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DType {
    /// 32-bit floating point
    #[default]
    F32,
}

impl DType {
    /// Returns the size in bytes of the data type
    pub fn size_of(&self) -> usize {
        trace_fn!("DType::size_of");
        debug!("Getting size of DType: {:?}", self);
        match self {
            DType::F32 => 4,
        }
    }
}

impl TryFrom<&str> for DType {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        trace_fn!("DType::try_from");
        debug!("Attempting to convert string '{}' to DType", value);
        match value {
            "f32" => Ok(DType::F32),
            _ => Err(format!("Unsupported data type: {value}")),
        }
    }
}

impl TryFrom<DType> for &str {
    type Error = String;

    fn try_from(value: DType) -> Result<Self, Self::Error> {
        match value {
            DType::F32 => Ok("f32"),
        }
    }
}

/// Multi-dimensional array (tensor) for numerical computing.
///
/// The core data structure in Rustic Net, supporting a variety of
/// mathematical operations and linear algebra functions.
#[derive(Clone)]
pub struct Tensor {
    /// The underlying data buffer
    pub(crate) data: Arc<Vec<f32>>,
    /// The shape of the tensor
    pub(crate) shape: Shape,
    /// The device where the tensor data is stored
    pub(crate) device: Device,
    /// The data type of tensor elements
    pub(crate) dtype: DType,
}

impl Tensor {
    // ===== Creation Methods =====

    /// Creates a new tensor from a vector with the given shape and device
    pub fn from_vec<T: Into<Vec<f32>>>(
        data: T,
        shape: &[usize],
        device: Device,
    ) -> Result<Self, String> {
        from_vec(data, shape, device)
    }

    /// Creates a new tensor from a slice with the given shape
    pub fn from_slice(slice: &[f32], shape: &[usize], device: Device) -> Result<Self, String> {
        from_slice(slice, shape, device)
    }

    /// Creates a new tensor filled with zeros
    pub fn zeros(shape: &[usize], device: Device) -> Self {
        zeros(shape, device)
    }

    /// Creates a new tensor filled with ones
    pub fn ones(shape: &[usize], device: Device) -> Self {
        ones(shape, device)
    }

    /// Creates an identity matrix of the given size
    pub fn identity(size: usize, device: Device) -> Self {
        identity(size, device)
    }

    /// Creates a new tensor with random values between 0.0 and 1.0
    pub fn random(shape: &[usize], device: Device) -> Self {
        random(shape, device)
    }

    /// Creates a new 1D tensor with values from start to end (exclusive)
    pub fn arange(start: f32, end: f32, device: Device) -> Self {
        arange(start, end, device)
    }

    // ===== Accessors =====

    /// Returns the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Returns the device where the tensor is stored
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the data type of tensor elements
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the number of dimensions of the tensor
    pub fn rank(&self) -> usize {
        self.shape.ndim()
    }

    /// Returns the total number of elements in the tensor
    pub fn numel(&self) -> usize {
        self.shape.len()
    }

    /// Converts the tensor to a vector
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.as_ref().clone()
    }

    // ===== Computation Operations =====

    /// Applies the ReLU activation function element-wise
    pub fn relu(&self) -> Result<Self, String> {
        trace_fn!("Tensor::relu");
        #[cfg(feature = "parallel")]
        {
            CpuParallel::relu(self)
        }
        #[cfg(not(feature = "parallel"))]
        {
            CpuSequential::relu(self)
        }
    }

    /// Element-wise addition with another tensor
    pub fn add(&self, other: &Self) -> Result<Self, String> {
        trace_fn!("Tensor::add");
        #[cfg(feature = "parallel")]
        {
            CpuParallel::add(self, other)
        }
        #[cfg(not(feature = "parallel"))]
        {
            CpuSequential::add(self, other)
        }
    }

    /// Element-wise subtraction with another tensor
    pub fn sub(&self, other: &Self) -> Result<Self, String> {
        trace_fn!("Tensor::sub");
        #[cfg(feature = "parallel")]
        {
            CpuParallel::sub(self, other)
        }
        #[cfg(not(feature = "parallel"))]
        {
            CpuSequential::sub(self, other)
        }
    }

    /// Element-wise multiplication with another tensor
    pub fn mul(&self, other: &Self) -> Result<Self, String> {
        trace_fn!("Tensor::mul");
        #[cfg(feature = "parallel")]
        {
            CpuParallel::mul(self, other)
        }
        #[cfg(not(feature = "parallel"))]
        {
            CpuSequential::mul(self, other)
        }
    }

    /// Element-wise division by another tensor
    pub fn div(&self, other: &Self) -> Result<Self, String> {
        trace_fn!("Tensor::div");
        #[cfg(feature = "parallel")]
        {
            CpuParallel::div(self, other)
        }
        #[cfg(not(feature = "parallel"))]
        {
            CpuSequential::div(self, other)
        }
    }

    /// Matrix multiplication with another tensor
    pub fn matmul(&self, other: &Self) -> Result<Self, String> {
        trace_fn!("Tensor::matmul");
        #[cfg(feature = "parallel")]
        {
            CpuParallel::matmul(self, other)
        }
        #[cfg(not(feature = "parallel"))]
        {
            CpuSequential::matmul(self, other)
        }
    }

    // ===== Reduction Operations =====

    /// Computes the sum of tensor elements along the specified axis
    pub fn sum(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::sum");
        #[cfg(feature = "parallel")]
        {
            CpuParallel::sum(self, axis)
        }
        #[cfg(not(feature = "parallel"))]
        {
            CpuSequential::sum(self, axis)
        }
    }

    /// Computes the mean of tensor elements along the specified axis
    pub fn mean(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::mean");
        #[cfg(feature = "parallel")]
        {
            CpuParallel::mean(self, axis)
        }
        #[cfg(not(feature = "parallel"))]
        {
            CpuSequential::mean(self, axis)
        }
    }

    /// Finds the maximum value along the specified axis
    pub fn max(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::max");
        #[cfg(feature = "parallel")]
        {
            CpuParallel::max(self, axis)
        }
        #[cfg(not(feature = "parallel"))]
        {
            CpuSequential::max(self, axis)
        }
    }

    /// Finds the minimum value along the specified axis
    pub fn min(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::min");
        #[cfg(feature = "parallel")]
        {
            CpuParallel::min(self, axis)
        }
        #[cfg(not(feature = "parallel"))]
        {
            CpuSequential::min(self, axis)
        }
    }

    /// Finds the index of the maximum value along the specified axis
    pub fn argmax(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::argmax");
        #[cfg(feature = "parallel")]
        {
            CpuParallel::argmax(self, axis)
        }
        #[cfg(not(feature = "parallel"))]
        {
            CpuSequential::argmax(self, axis)
        }
    }

    /// Finds the index of the minimum value along the specified axis
    pub fn argmin(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::argmin");
        #[cfg(feature = "parallel")]
        {
            CpuParallel::argmin(self, axis)
        }
        #[cfg(not(feature = "parallel"))]
        {
            CpuSequential::argmin(self, axis)
        }
    }

    // ===== Shape Manipulation =====

    /// Reshapes the tensor to the given shape without changing the data.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, String> {
        trace_fn!("Tensor::reshape");
        if self.numel() != new_shape.iter().product::<usize>() {
            return Err(format!(
                "Cannot reshape tensor of size {} to shape {:?}",
                self.numel(),
                new_shape
            ));
        }

        Ok(Tensor {
            data: self.data.clone(),
            shape: Shape::new(new_shape),
            device: self.device,
            dtype: self.dtype,
        })
    }

    /// Transposes the tensor by reversing its dimensions.
    /// To transpose by a specific permutation of axes, use `transpose_axes`.
    pub fn transpose(&self) -> Result<Self, String> {
        trace_fn!("Tensor::transpose");
        let axes: Vec<usize> = (0..self.rank()).rev().collect();
        self.transpose_axes(&axes)
    }

    /// Transposes the tensor according to the given axes permutation.
    pub fn transpose_axes(&self, axes: &[usize]) -> Result<Self, String> {
        trace_fn!("Tensor::transpose_axes");
        let rank = self.rank();
        if axes.len() != rank {
            return Err(format!(
                "Axes length {} does not match tensor rank {}",
                axes.len(),
                rank
            ));
        }

        let new_dims: Vec<usize> = axes.iter().map(|&i| self.shape()[i]).collect();
        let new_shape = Shape::new(&new_dims);
        let mut new_data = vec![0.0; self.numel()];

        let old_strides = self.shape.strides();
        let new_strides = new_shape.strides();

        for (i, &val) in self.data.iter().enumerate() {
            let mut old_indices = vec![0; rank];
            let mut temp_index = i;
            for (j, &stride) in old_strides.iter().enumerate() {
                old_indices[j] = temp_index / stride;
                temp_index %= stride;
            }

            let mut new_indices = vec![0; rank];
            for (j, &axis) in axes.iter().enumerate() {
                new_indices[j] = old_indices[axis];
            }

            let mut new_i = 0;
            for (j, &index) in new_indices.iter().enumerate() {
                new_i += index * new_strides[j];
            }
            new_data[new_i] = val;
        }

        Ok(Tensor {
            data: Arc::new(new_data),
            shape: new_shape,
            device: self.device,
            dtype: self.dtype,
        })
    }

    /// Adds a new dimension of size 1 at the specified axis.
    pub fn expand_dims(&self, axis: usize) -> Result<Self, String> {
        trace_fn!("Tensor::expand_dims");
        if axis > self.rank() {
            return Err(format!(
                "Axis {} is out of bounds for tensor of rank {}",
                axis,
                self.rank()
            ));
        }
        let mut new_dims = self.shape().to_vec();
        new_dims.insert(axis, 1);
        self.reshape(&new_dims)
    }

    /// Removes dimensions of size 1. If `axis` is specified, it only removes that dimension.
    pub fn squeeze(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::squeeze");
        let mut new_dims = self.shape().to_vec();
        match axis {
            Some(axis) => {
                if axis >= self.rank() {
                    return Err(format!(
                        "Axis {} is out of bounds for tensor of rank {}",
                        axis,
                        self.rank()
                    ));
                }
                if new_dims[axis] == 1 {
                    new_dims.remove(axis);
                }
            }
            None => {
                new_dims.retain(|&dim| dim != 1);
            }
        }
        if new_dims.is_empty() {
            new_dims.push(1);
        }
        self.reshape(&new_dims)
    }

    // ===== Device Management =====

    /// Moves the tensor to the specified device
    pub fn to_device(&self, device: Device) -> Result<Self, String> {
        trace_fn!("Tensor::to_device");
        if self.device == device {
            return Ok(self.clone());
        }

        // For now, we only support CPU tensors
        // In the future, this will handle transferring data between devices
        match device {
            Device::Cpu(_) => {
                // Create a new tensor with the same data on the CPU
                Ok(Tensor {
                    data: self.data.clone(),
                    shape: self.shape.clone(),
                    device,
                    dtype: self.dtype,
                })
            }
            _ => Err(format!("Device not yet supported: {device:?}")),
        }
    }
}

// Implement display for better debugging
impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("device", &self.device)
            .field("dtype", &self.dtype)
            .field("data", &format!("[{} f32 values]", self.data.len()))
            .finish()
    }
}
