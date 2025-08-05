//! # Tensor Module
//! 
//! This module provides the core tensor type and operations for the Rustic Net library.

use crate::trace_fn;
use std::fmt;
use std::sync::Arc;
use std::ops::{Add, Sub, Mul, Div};

pub mod backends;
pub mod creation;
pub mod shape;
pub mod impl_ops;
pub mod shape_ops;

use backends::*;
pub use creation::*;
pub use shape::Shape;
pub use shape_ops::*;
pub use tracing::debug;

#[cfg(feature = "parallel")]
pub use crate::parallel;

/// Represents the device where tensor data is stored
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

/// Represents the data type of tensor elements
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
            _ => Err(format!("Unsupported data type: {}", value)),
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

/// Represents a multi-dimensional array (tensor)
pub struct Tensor {
    /// The underlying data buffer
    data: Arc<Vec<f32>>,
    /// The shape of the tensor
    shape: Shape,
    /// The device where the tensor data is stored
    device: Device,
    /// The data type of tensor elements
    dtype: DType,
}

impl Tensor {
    // ===== Creation Methods =====

    /// Creates a new tensor from a vector with the given shape and device
    pub fn from_vec<T: Into<Vec<f32>>>(data: T, shape: &[usize], device: Device) -> Result<Self, String> {
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

    /// Converts the tensor to a vector
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.as_ref().clone()
    }

    // ===== Operations =====

    /// Applies the ReLU activation function element-wise
    pub fn relu(&self) -> Result<Self, String> {
        trace_fn!("Tensor::relu");
        match self.device {
            Device::Cpu(_) => {
                #[cfg(feature = "parallel")]
                {
                    backends::cpu_par::CpuParallel::relu(self)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    backends::cpu_seq::CpuSequential::relu(self)
                }
            }
            _ => Err(format!("Device not yet supported: {:?}", self.device)),
        }
    }

    /// Element-wise addition with another tensor
    pub fn add_tensor(&self, other: &Self) -> Result<Self, String> {
        trace_fn!("Tensor::add_tensor");
        match (self.device, other.device) {
            (Device::Cpu(_), Device::Cpu(_)) => {
                #[cfg(feature = "parallel")]
                {
                    backends::cpu_par::CpuParallel::add(self, other)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    backends::cpu_seq::CpuSequential::add(self, other)
                }
            }
            _ => Err(format!(
                "Operation between tensors on different devices not supported: {:?} vs {:?}",
                self.device, other.device
            )),
        }
    }

    /// Element-wise subtraction with another tensor
    pub fn sub_tensor(&self, other: &Self) -> Result<Self, String> {
        trace_fn!("Tensor::sub_tensor");
        match (self.device, other.device) {
            (Device::Cpu(_), Device::Cpu(_)) => {
                #[cfg(feature = "parallel")]
                {
                    backends::cpu_par::CpuParallel::sub(self, other)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    backends::cpu_seq::CpuSequential::sub(self, other)
                }
            }
            _ => Err(format!(
                "Operation between tensors on different devices not supported: {:?} vs {:?}",
                self.device, other.device
            )),
        }
    }

    /// Element-wise multiplication with another tensor
    pub fn mul_tensor(&self, other: &Self) -> Result<Self, String> {
        trace_fn!("Tensor::mul_tensor");
        match (self.device, other.device) {
            (Device::Cpu(_), Device::Cpu(_)) => {
                #[cfg(feature = "parallel")]
                {
                    backends::cpu_par::CpuParallel::mul(self, other)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    backends::cpu_seq::CpuSequential::mul(self, other)
                }
            }
            _ => Err(format!(
                "Operation between tensors on different devices not supported: {:?} vs {:?}",
                self.device, other.device
            )),
        }
    }

    /// Element-wise division by another tensor
    pub fn div_tensor(&self, other: &Self) -> Result<Self, String> {
        trace_fn!("Tensor::div_tensor");
        match (self.device, other.device) {
            (Device::Cpu(_), Device::Cpu(_)) => {
                #[cfg(feature = "parallel")]
                {
                    backends::cpu_par::CpuParallel::div(self, other)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    backends::cpu_seq::CpuSequential::div(self, other)
                }
            }
            _ => Err(format!(
                "Operation between tensors on different devices not supported: {:?} vs {:?}",
                self.device, other.device
            )),
        }
    }

    /// Matrix multiplication with another tensor
    pub fn matmul(&self, other: &Self) -> Result<Self, String> {
        trace_fn!("Tensor::matmul");
        match (self.device, other.device) {
            (Device::Cpu(_), Device::Cpu(_)) => {
                #[cfg(feature = "parallel")]
                {
                    backends::cpu_par::CpuParallel::matmul(self, other)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    backends::cpu_seq::CpuSequential::matmul(self, other)
                }
            }
            _ => Err(format!(
                "Operation between tensors on different devices not supported: {:?} vs {:?}",
                self.device, other.device
            )),
        }
    }

    // ===== Reduction Operations =====

    /// Computes the sum of tensor elements along the specified axis
    /// If axis is None, sums all elements
    pub fn sum(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::sum");
        match self.device {
            Device::Cpu(_) => {
                #[cfg(feature = "parallel")]
                {
                    backends::cpu_par::CpuParallel::sum(self, axis)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    backends::cpu_seq::CpuSequential::sum(self, axis)
                }
            }
            _ => Err(format!("Device not yet supported: {:?}", self.device)),
        }
    }

    /// Computes the mean of tensor elements along the specified axis
    /// If axis is None, computes mean of all elements
    pub fn mean(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::mean");
        match self.device {
            Device::Cpu(_) => {
                #[cfg(feature = "parallel")]
                {
                    backends::cpu_par::CpuParallel::mean(self, axis)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    backends::cpu_seq::CpuSequential::mean(self, axis)
                }
            }
            _ => Err(format!("Device not yet supported: {:?}", self.device)),
        }
    }

    /// Finds the maximum value along the specified axis
    /// If axis is None, finds the global maximum
    pub fn max(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::max");
        match self.device {
            Device::Cpu(_) => {
                #[cfg(feature = "parallel")]
                {
                    backends::cpu_par::CpuParallel::max(self, axis)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    backends::cpu_seq::CpuSequential::max(self, axis)
                }
            }
            _ => Err(format!("Device not yet supported: {:?}", self.device)),
        }
    }

    /// Finds the minimum value along the specified axis
    /// If axis is None, finds the global minimum
    pub fn min(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::min");
        match self.device {
            Device::Cpu(_) => {
                #[cfg(feature = "parallel")]
                {
                    backends::cpu_par::CpuParallel::min(self, axis)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    backends::cpu_seq::CpuSequential::min(self, axis)
                }
            }
            _ => Err(format!("Device not yet supported: {:?}", self.device)),
        }
    }

    /// Finds the index of the maximum value along the specified axis
    /// If axis is None, finds the index of the global maximum
    pub fn argmax(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::argmax");
        match self.device {
            Device::Cpu(_) => {
                #[cfg(feature = "parallel")]
                {
                    backends::cpu_par::CpuParallel::argmax(self, axis)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    backends::cpu_seq::CpuSequential::argmax(self, axis)
                }
            }
            _ => Err(format!("Device not yet supported: {:?}", self.device)),
        }
    }

    /// Finds the index of the minimum value along the specified axis
    /// If axis is None, finds the index of the global minimum
    pub fn argmin(&self, axis: Option<usize>) -> Result<Self, String> {
        trace_fn!("Tensor::argmin");
        match self.device {
            Device::Cpu(_) => {
                #[cfg(feature = "parallel")]
                {
                    backends::cpu_par::CpuParallel::argmin(self, axis)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    backends::cpu_seq::CpuSequential::argmin(self, axis)
                }
            }
            _ => Err(format!("Device not yet supported: {:?}", self.device)),
        }
    }

    // ===== Shape Manipulation =====

    /// Reshapes the tensor to the given shape
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, String> {
        trace_fn!("Tensor::reshape");
        // Calculate the total number of elements in the new shape
        let new_size: usize = new_shape.iter().product();
        
        // Ensure the new shape has the same number of elements as the original
        if new_size != self.data.len() {
            return Err(format!(
                "Cannot reshape tensor of size {} to shape {:?}",
                self.data.len(),
                new_shape
            ));
        }

        // Create a new tensor with the same data but new shape
        Ok(Tensor {
            data: self.data.clone(),
            shape: Shape::new(new_shape),
            device: self.device,
            dtype: self.dtype,
        })
    }

    /// Transposes the tensor according to the given axes
    /// If axes is None, reverses the dimensions
    pub fn transpose(&self, axes: Option<&[usize]>) -> Result<Self, String> {
        trace_fn!("Tensor::transpose");
        let ndim = self.shape().len();
        
        // Determine the new axes order
        let axes = match axes {
            Some(axes) => {
                if axes.len() != ndim {
                    return Err(format!(
                        "Axes length {} does not match tensor rank {}",
                        axes.len(),
                        ndim
                    ));
                }
                axes.to_vec()
            }
            None => (0..ndim).rev().collect::<Vec<_>>(),
        };

        // Calculate the new shape and strides
        let new_dims: Vec<usize> = axes.iter().map(|&i| self.shape()[i]).collect();
        let new_strides: Vec<usize> = axes.iter().map(|&i| self.shape.strides()[i]).collect();
        
        // Create a new tensor with the transposed shape
        let mut result = Tensor {
            data: Arc::new(vec![0.0; self.data.len()]),
            shape: Shape {
                dims: new_dims,
                strides: new_strides,
                size: self.data.len(),
            },
            device: self.device,
            dtype: self.dtype,
        };

        // Perform the transposition
        let result_data = Arc::make_mut(&mut result.data);
        
        // For each element in the original tensor, calculate its new position
        for (i, &val) in self.data.iter().enumerate() {
            let mut new_idx = 0;
            let mut remaining = i;
            
            // Calculate the new index using the original strides and new axes order
            for (&dim, &stride) in self.shape.dims().iter().zip(self.shape.strides().iter()) {
                let pos = remaining / stride;
                remaining %= stride;
                
                // Find the position of this dimension in the new axes order
                let new_axis = axes.iter().position(|&x| x == 0).unwrap();
                new_idx += pos * result.shape.strides()[new_axis];
            }
            
            result_data[new_idx] = val;
        }

        Ok(result)
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
            _ => Err(format!("Device not yet supported: {:?}", device)),
        }
    }
}

// Implementation of Tensor methods will be added in subsequent steps

// Implement Clone for Tensor
impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            device: self.device,
            dtype: self.dtype,
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

// Implement basic arithmetic operations with scalar values
impl Add<f32> for Tensor {
    type Output = Tensor;

    fn add(mut self, rhs: f32) -> Self::Output {
        trace_fn!("Tensor::add_scalar");
        let data = self.data.par_iter().map(|&x| x + rhs).collect();
        
        Tensor {
            data: Arc::new(data),
            shape: self.shape,
            device: self.device,
            dtype: self.dtype,
        }
    }
}

impl Sub<f32> for Tensor {
    type Output = Tensor;

    fn sub(mut self, rhs: f32) -> Self::Output {
        trace_fn!("Tensor::sub_scalar");
        let data = self.data.par_iter().map(|&x| x - rhs).collect();
        
        Tensor {
            data: Arc::new(data),
            shape: self.shape,
            device: self.device,
            dtype: self.dtype,
        }
    }
}

impl Mul<f32> for Tensor {
    type Output = Tensor;

    fn mul(mut self, rhs: f32) -> Self::Output {
        trace_fn!("Tensor::mul_scalar");
        let data = self.data.par_iter().map(|&x| x * rhs).collect();
        
        Tensor {
            data: Arc::new(data),
            shape: self.shape,
            device: self.device,
            dtype: self.dtype,
        }
    }
}

impl Div<f32> for Tensor {
    type Output = Tensor;

    fn div(mut self, rhs: f32) -> Self::Output {
        trace_fn!("Tensor::div_scalar");
        if rhs == 0.0 {
            panic!("Division by zero");
        }
        
        let data = self.data.par_iter().map(|&x| x / rhs).collect();
        
        Tensor {
            data: Arc::new(data),
            shape: self.shape,
            device: self.device,
            dtype: self.dtype,
        }
    }
}

// Import the tests module
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracing::init_tracing;

    #[test]
    fn test_device_default() {
        let _ = init_tracing();
        let device = Device::default();
        assert_eq!(device, Device::Cpu(Some(0)));
    }

    #[test]
    fn test_dtype_conversion() {
        let _ = init_tracing();
        let dtype: DType = "f32".try_into().unwrap();
        assert_eq!(dtype, DType::F32);
        
        let dtype_str: &str = DType::F32.try_into().unwrap();
        assert_eq!(dtype_str, "f32");
        
        assert_eq!(DType::F32.size_of(), 4);
    }
    
    #[test]
    fn test_tensor_creation() {
        let _ = init_tracing();
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], Device::default()).unwrap();
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0]);
    }
    
    #[test]
    fn test_tensor_operations() {
        let _ = init_tracing();
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], Device::default()).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3], Device::default()).unwrap();
        
        // Test element-wise addition
        let c = a.add_tensor(&b).unwrap();
        assert_eq!(c.to_vec(), vec![5.0, 7.0, 9.0]);
        
        // Test scalar operations
        let d = c + 1.0;
        assert_eq!(d.to_vec(), vec![6.0, 8.0, 10.0]);
        
        // Test reduction
        let e = d.sum(None).unwrap();
        assert_eq!(e.to_vec(), vec![24.0]);
    }
    
    #[test]
    fn test_reshape() {
        let _ = init_tracing();
        let t = Tensor::arange(0.0, 6.0, Device::default());
        let t = t.reshape(&[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }
    
    #[test]
    fn test_transpose() {
        let _ = init_tracing();
        let t = Tensor::arange(0.0, 6.0, Device::default());
        let t = t.reshape(&[2, 3]).unwrap();
        let t = t.transpose(None).unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.to_vec(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }
}
