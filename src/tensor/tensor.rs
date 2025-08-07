use crate::trace_fn;
use crate::DType;
use crate::Device;
use crate::Shape;
use std::sync::Arc;

use std::fmt;

use crate::tensor::backends::traits::{
    BinaryElementwiseOps, CreationOps, MatOps, ReductionOps, ShapeOps, UnaryOps,
};
use crate::tensor::backends::Cpu;

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
    pub fn to_vec(&self) -> &Vec<f32> {
        self.data.as_ref()
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
        Cpu::transpose(self)
    }

    /// Transposes the tensor according to the given axes permutation.
    pub fn transpose_axes(&self, axes: &[usize]) -> Result<Self, String> {
        trace_fn!("Tensor::transpose_axes");
        Cpu::transpose_axes(self, axes)
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
