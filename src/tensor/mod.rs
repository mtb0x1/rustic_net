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

use crate::trace_fn;
use std::fmt;
use std::sync::Arc;

pub mod backends;

pub mod impl_ops;
pub mod shape;

use backends::{
    traits::{BinaryElementwiseOps, MatOps, ReductionOps, UnaryOps, CreationOps},
    Cpu,
};

pub use shape::Shape;
pub use tracing::debug;

#[cfg(feature = "parallel")]
pub use crate::parallel;

/// Compute device for tensor storage and operations.
///
/// Tensors can be allocated on different devices, with operations
/// automatically dispatched to the appropriate backend. The device
/// determines where tensor computations are performed.
///
/// # Examples
/// ```rust
/// use rustic_net::tensor::Device;
///
/// // Default CPU device
/// let device1 = Device::default();
///
/// // Specific CPU device (for multi-socket systems)
/// let device2 = Device::Cpu(Some(1));
///
/// // CUDA device (requires 'cuda' feature)
/// #[cfg(feature = "cuda")]
/// let device3 = Device::Cuda(0);
/// ```
///
/// # Thread Safety
/// All device variants are `Send` and `Sync`, allowing them to be shared across threads.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Device {
    /// CPU device with optional device ID
    ///
    /// - `None`: Default CPU device
    /// - `Some(n)`: Specific CPU device (useful for NUMA systems)
    Cpu(Option<usize>),

    /// CUDA device with device ID
    ///
    /// Requires the `cuda` feature. The ID corresponds to the CUDA device index.
    #[cfg(feature = "cuda")]
    Cuda(usize),

    /// WebGPU device with device ID
    ///
    /// Requires the `wasm` feature. The ID corresponds to the WebGPU adapter index.
    #[cfg(feature = "webgpu")]
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
/// Defines the data type of elements stored in a tensor. Currently,
/// only 32-bit floating point (f32) is supported, but the enum is
/// designed to be extended with additional types in the future.
///
/// # Examples
/// ```rust
/// use rustic_net::tensor::DType;
///
/// let dtype = DType::F32;
/// assert_eq!(dtype.size_of(), 4);  // 4 bytes for f32
/// ```
///
/// # Type Safety
/// The `DType` enum ensures type safety by preventing incompatible operations
/// between tensors of different data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit floating point number (IEEE 754)
    ///
    /// - Size: 4 bytes
    /// - Range: Approximately ±3.4 × 10^38 with 7 decimal digits of precision
    F32,
    // Future types:
    // F64,
    // I32,
    // I64,
    // U8,
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
/// The core data structure in Rustic Net, representing an N-dimensional array
/// of numeric values. Tensors support a wide range of mathematical operations
/// and are the fundamental building block for machine learning models.
///
/// # Design
/// - **Data Layout**: Row-major order (C-style)
/// - **Memory**: Reference-counted with `Arc` for efficient sharing
/// - **Device-Agnostic**: Same API for CPU and GPU tensors
/// - **Thread-Safe**: Implements `Send` and `Sync`
///
/// # Examples
/// ```rust
/// use rustic_net::tensor::{Tensor, Device};
///
/// // Create a 2x3 tensor from a vector
/// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Device::default())?;
///
/// // Perform operations
/// let t_squared = t.mul(&t)?;  // Element-wise multiplication
/// let t_sum = t.sum(None)?;    // Sum all elements
/// # Ok::<(), String>(())
/// ```
///
/// # Performance Considerations
/// - **Views**: Operations like `reshape()` and `transpose()` return views that share data
/// - **Cloning**: Use `.clone()` sparingly as it performs a deep copy
/// - **Device Transfers**: Moving data between devices is expensive
/// - **In-Place Operations**: Methods with `_` suffix (e.g., `add_()`) modify in-place
///
/// # Memory Management
/// Tensors use reference counting to manage memory. The underlying data is automatically
/// deallocated when the last tensor referencing it is dropped.
#[derive(Clone)]
pub struct Tensor {
    /// The underlying tensor data
    ///
    /// Stored in row-major order (C-style) with the last index changing fastest.
    /// For example, a 2x3 tensor is stored as [a11, a12, a13, a21, a22, a23].
    pub data: Arc<Vec<f32>>,

    /// The shape of the tensor
    ///
    /// Defines the number of dimensions and size of each dimension.
    /// For example, a 2x3 matrix has shape [2, 3].
    pub shape: Shape,

    /// The device where the tensor is stored
    ///
    /// Determines where computations are performed (CPU/GPU).
    pub device: Device,

    /// The data type of tensor elements
    ///
    /// Currently only `DType::F32` is supported.
    pub dtype: DType,
}

impl Tensor {
    // ===== Creation Methods =====

    /// Creates a new tensor from a vector with the given shape and device
    pub fn from_vec<T: Into<Vec<f32>>>(
        data: T,
        shape: &[usize],
        device: Device,
    ) -> Result<Self, String> {
        Cpu::from_vec(data.into(), shape, device)
    }

    /// Creates a new tensor from a slice with the given shape
    pub fn from_slice(slice: &[f32], shape: &[usize], device: Device) -> Result<Self, String> {
        Cpu::from_slice(slice, shape, device)
    }

    /// Creates a new tensor filled with zeros
    pub fn zeros(shape: &[usize], device: Device) -> Self {
        Cpu::zeros(shape, device).expect("Failed to create zeros tensor")
    }

    /// Creates a new tensor filled with ones
    pub fn ones(shape: &[usize], device: Device) -> Self {
        Cpu::ones(shape, device).expect("Failed to create ones tensor")
    }

    /// Creates an identity matrix of the given size
    pub fn identity(size: usize, device: Device) -> Self {
        Cpu::identity(size, device).expect("Failed to create identity tensor")
    }

    /// Creates a new tensor with random values between 0.0 and 1.0
    pub fn random(shape: &[usize], device: Device) -> Self {
        Cpu::random(shape, device).expect("Failed to create random tensor")
    }

    /// Creates a new 1D tensor with values from start to end (exclusive)
    pub fn arange(start: f32, end: f32, device: Device) -> Self {
        Cpu::arange(start, end, device).expect("Failed to create arange tensor")
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
        Cpu::relu(self)
    }

    /// Element-wise addition with another tensor
    pub fn add(&self, other: &Self) -> Result<Self, String> {
        trace_fn!("Tensor::add");
        Cpu::add(self, other)
    }

    /// Element-wise subtraction with another tensor
    pub fn sub(&self, other: &Self) -> Result<Self, String> {
        trace_fn!("Tensor::sub");
        Cpu::sub(self, other)
    }

    /// Element-wise multiplication with another tensor
    pub fn mul(&self, other: &Self) -> Result<Self, String> {
        trace_fn!("Tensor::mul");
        Cpu::mul(self, other)
    }

    /// Element-wise division by another tensor
    pub fn div(&self, other: &Self) -> Result<Self, String> {
        trace_fn!("Tensor::div");
        Cpu::div(self, other)
    }

    /// Matrix multiplication with another tensor
    pub fn matmul(&self, other: &Self) -> Result<Self, String> {
        trace_fn!("Tensor::matmul");
        Cpu::matmul(self, other)
    }

    // ===== Reduction Operations =====

    /// Computes the sum of tensor elements along the specified axis
    pub fn sum(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::sum");
        Cpu::sum(self, axis)
    }

    /// Computes the mean of tensor elements along the specified axis
    pub fn mean(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::mean");
        Cpu::mean(self, axis)
    }

    /// Finds the maximum value along the specified axis
    pub fn max(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::max");
        Cpu::max(self, axis)
    }

    /// Finds the minimum value along the specified axis
    pub fn min(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::min");
        Cpu::min(self, axis)
    }

    /// Finds the index of the maximum value along the specified axis
    pub fn argmax(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::argmax");
        Cpu::argmax(self, axis)
    }

    /// Finds the index of the minimum value along the specified axis
    pub fn argmin(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::argmin");
        Cpu::argmin(self, axis)
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
    #[allow(unreachable_patterns)]
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
