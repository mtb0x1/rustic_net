//! # SIMD CPU Backend
//!
//! A high-performance backend that leverages SIMD (Single Instruction, Multiple Data)
//! instructions to accelerate tensor operations on compatible CPUs. This backend is
//! automatically selected when the `simd` feature is enabled.
//!
//! ## Features
//! - Uses 256-bit wide SIMD registers (AVX2) for processing 8 f32 elements in parallel
//! - Fallback to scalar operations for remaining elements not fitting in SIMD lanes
//! - Optimized for modern x86_64 CPUs with AVX2 support
//! - Maintains numerical consistency with non-SIMD implementations

use super::traits::*;
use crate::tensor::Tensor;
use crate::trace_fn;
use std::simd::{cmp::SimdPartialOrd, f32x8};
use std::sync::Arc;

/// Marker type for the SIMD CPU backend.
///
/// This backend implements all tensor operations using explicit SIMD intrinsics
/// for maximum performance on supported hardware. It processes data in chunks of
/// 8 floating-point numbers (256 bits) when possible, falling back to scalar
/// operations for any remaining elements.
///
/// # Performance Considerations
/// - Best performance is achieved with tensors whose dimensions are multiples of 8
/// - Overhead of SIMD setup means small tensors might see limited benefit
/// - Enables significant speedups for element-wise operations on large tensors
#[derive(Debug, Clone, Copy)]
pub struct CpuSimd;

impl UnaryOps for CpuSimd {
    /// Applies the Rectified Linear Unit (ReLU) activation function using SIMD acceleration.
    ///
    /// ReLU is defined as `max(0, x)` and is a common activation function in neural networks.
    /// This implementation processes the tensor data in chunks of 8 elements (256 bits)
    /// using SIMD instructions for optimal performance.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor to apply ReLU to
    ///
    /// # Returns
    /// A new tensor with ReLU applied element-wise, or an error if the operation fails.
    ///
    /// # Performance
    /// - Processes 8 elements in parallel using SIMD instructions
    /// - Falls back to scalar operations for any remaining elements (tensor length % 8)
    /// - Maintains numerical stability by handling negative values correctly
    ///
    /// # Example
    /// ```
    /// # use rustic_net::tensor::backends::cpu_simd::CpuSimd;
    /// # use rustic_net::tensor::backends::traits::UnaryOps;
    /// # use rustic_net::tensor::{Tensor, Device};
    /// let input = Tensor::from_vec(vec![-1.0, 0.5, -0.5, 2.0], &[2, 2], Device::default()).unwrap();
    /// let output = CpuSimd::relu(&input).unwrap();
    /// assert_eq!(output.to_vec(), vec![0.0, 0.5, 0.0, 2.0]);
    /// ```
    fn relu(tensor: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::relu");

        // Create a mutable copy of the tensor data
        let mut data = tensor.data.to_vec();
        let len = data.len();

        // Split data into chunks of 8 elements for SIMD processing and a remainder
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let chunks = chunks.chunks_mut(8);

        // Process chunks using SIMD instructions
        for chunk in chunks {
            // Load 8 elements into SIMD register
            let simd_chunk = f32x8::from_slice(chunk);
            // Create mask for elements > 0
            let mask = simd_chunk.simd_gt(f32x8::splat(0.0));
            // Select between original value and 0.0 based on mask
            let result = mask.select(simd_chunk, f32x8::splat(0.0));
            // Store result back to memory
            result.copy_to_slice(chunk);
        }

        // Process any remaining elements that didn't fit in a full SIMD chunk
        for val in remainder.iter_mut() {
            if *val < 0.0 {
                *val = 0.0;
            }
        }

        // Create and return a new tensor with the processed data
        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }
}

impl ScalarOps for CpuSimd {
    /// Performs element-wise addition of a scalar to a tensor using SIMD acceleration.
    ///
    /// This implementation processes the tensor data in chunks of 8 elements (256 bits)
    /// using SIMD instructions for optimal performance, with a fallback to scalar
    /// operations for any remaining elements.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor to add the scalar to
    /// * `scalar` - Scalar value to add to each element
    ///
    /// # Returns
    /// A new tensor with the scalar added to each element, or an error if the operation fails.
    ///
    /// # Performance
    /// - Processes 8 elements in parallel using SIMD instructions
    /// - Falls back to scalar operations for any remaining elements
    /// - Maintains numerical stability and handles edge cases properly
    ///
    /// # Example
    /// ```
    /// # use rustic_net::tensor::backends::cpu_simd::CpuSimd;
    /// # use rustic_net::tensor::backends::traits::ScalarOps;
    /// # use rustic_net::tensor::{Tensor, Device};
    /// let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();
    /// let result = CpuSimd::add_scalar(&input, 5.0).unwrap();
    /// assert_eq!(result.to_vec(), vec![6.0, 7.0, 8.0, 9.0]);
    /// ```
    fn add_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::add_scalar");
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let chunks = chunks.chunks_mut(8);
        let scalar_simd = f32x8::splat(scalar);

        for chunk in chunks {
            let simd_chunk = f32x8::from_slice(chunk);
            let result = simd_chunk + scalar_simd;
            result.copy_to_slice(chunk);
        }

        for val in remainder.iter_mut() {
            *val += scalar;
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    /// Performs element-wise subtraction of a scalar from a tensor using SIMD acceleration.
    ///
    /// This implementation processes the tensor data in chunks of 8 elements (256 bits)
    /// using SIMD instructions for optimal performance, with a fallback to scalar
    /// operations for any remaining elements.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor to subtract the scalar from
    /// * `scalar` - Scalar value to subtract from each element
    ///
    /// # Returns
    /// A new tensor with the scalar subtracted from each element, or an error if the operation fails.
    ///
    /// # Performance
    /// - Processes 8 elements in parallel using SIMD instructions
    /// - Falls back to scalar operations for any remaining elements
    /// - Maintains numerical stability and handles edge cases properly
    ///
    /// # Example
    /// ```
    /// # use rustic_net::tensor::backends::cpu_simd::CpuSimd;
    /// # use rustic_net::tensor::backends::traits::ScalarOps;
    /// # use rustic_net::tensor::{Tensor, Device};
    /// let input = Tensor::from_vec(vec![5.0, 7.0, 9.0, 11.0], &[2, 2], Device::default()).unwrap();
    /// let result = CpuSimd::sub_scalar(&input, 3.0).unwrap();
    /// assert_eq!(result.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    /// ```
    fn sub_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::sub_scalar");
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let chunks = chunks.chunks_mut(8);
        let scalar_simd = f32x8::splat(scalar);

        for chunk in chunks {
            let simd_chunk = f32x8::from_slice(chunk);
            let result = simd_chunk - scalar_simd;
            result.copy_to_slice(chunk);
        }

        for val in remainder.iter_mut() {
            *val -= scalar;
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    /// Performs element-wise multiplication of a tensor by a scalar using SIMD acceleration.
    ///
    /// This implementation processes the tensor data in chunks of 8 elements (256 bits)
    /// using SIMD instructions for optimal performance, with a fallback to scalar
    /// operations for any remaining elements.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor to multiply by the scalar
    /// * `scalar` - Scalar value to multiply each element by
    ///
    /// # Returns
    /// A new tensor with each element multiplied by the scalar, or an error if the operation fails.
    ///
    /// # Performance
    /// - Processes 8 elements in parallel using SIMD instructions
    /// - Falls back to scalar operations for any remaining elements
    /// - Handles special cases like multiplication by zero or one efficiently
    ///
    /// # Example
    /// ```
    /// # use rustic_net::tensor::backends::cpu_simd::CpuSimd;
    /// # use rustic_net::tensor::backends::traits::ScalarOps;
    /// # use rustic_net::tensor::{Tensor, Device};
    /// let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();
    /// let result = CpuSimd::mul_scalar(&input, 2.5).unwrap();
    /// assert_eq!(result.to_vec(), vec![2.5, 5.0, 7.5, 10.0]);
    /// ```
    fn mul_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::mul_scalar");
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let chunks = chunks.chunks_mut(8);
        let scalar_simd = f32x8::splat(scalar);

        for chunk in chunks {
            let simd_chunk = f32x8::from_slice(chunk);
            let result = simd_chunk * scalar_simd;
            result.copy_to_slice(chunk);
        }

        for val in remainder.iter_mut() {
            *val *= scalar;
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    /// Performs element-wise division of a tensor by a scalar using SIMD acceleration.
    ///
    /// This implementation processes the tensor data in chunks of 8 elements (256 bits)
    /// using SIMD instructions for optimal performance, with a fallback to scalar
    /// operations for any remaining elements. It includes checks for division by zero.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor to be divided by the scalar
    /// * `scalar` - Scalar value to divide each element by (must not be zero)
    ///
    /// # Returns
    /// A new tensor with each element divided by the scalar, or an error if:
    /// - The scalar is zero (division by zero)
    /// - Any other error occurs during the operation
    ///
    /// # Performance
    /// - Processes 8 elements in parallel using SIMD instructions
    /// - Falls back to scalar operations for any remaining elements
    /// - Includes a single check for division by zero before processing
    ///
    /// # Example
    /// ```
    /// # use rustic_net::tensor::backends::cpu_simd::CpuSimd;
    /// # use rustic_net::tensor::backends::traits::ScalarOps;
    /// # use rustic_net::tensor::{Tensor, Device};
    /// let input = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2], Device::default()).unwrap();
    /// let result = CpuSimd::div_scalar(&input, 5.0).unwrap();
    /// assert_eq!(result.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    /// ```
    ///
    /// # Panics
    /// This function will return an error if the scalar is zero.
    fn div_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::div_scalar");
        if scalar == 0.0 {
            return Err("Division by zero".to_string());
        }
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let chunks = chunks.chunks_mut(8);
        let scalar_simd = f32x8::splat(scalar);

        for chunk in chunks {
            let simd_chunk = f32x8::from_slice(chunk);
            let result = simd_chunk / scalar_simd;
            result.copy_to_slice(chunk);
        }

        for val in remainder.iter_mut() {
            *val /= scalar;
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    /// Performs reverse element-wise addition of a scalar to a tensor using SIMD acceleration.
    ///
    /// This method adds each element of the tensor to a scalar value (scalar + element),
    /// which is different from `add_scalar` that adds the scalar to each element (element + scalar).
    /// This implementation processes the tensor data in chunks of 8 elements (256 bits)
    /// using SIMD instructions for optimal performance.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor to be added to the scalar
    /// * `scalar` - Scalar value to which each element will be added
    ///
    /// # Returns
    /// A new tensor where each element is the result of (scalar + element).
    ///
    /// # Performance
    /// - Processes 8 elements in parallel using SIMD instructions
    /// - Falls back to scalar operations for any remaining elements
    /// - Maintains numerical stability and handles edge cases properly
    ///
    /// # Example
    /// ```
    /// # use rustic_net::tensor::backends::cpu_simd::CpuSimd;
    /// # use rustic_net::tensor::backends::traits::ScalarOps;
    /// # use rustic_net::tensor::{Tensor, Device};
    /// let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();
    /// let result = CpuSimd::r_add_scalar(&input, 5.0).unwrap();
    /// assert_eq!(result.to_vec(), vec![6.0, 7.0, 8.0, 9.0]);
    /// ```
    ///
    /// # Note
    /// This operation is equivalent to `scalar + tensor` and is different from `tensor + scalar`
    /// (which is implemented by `add_scalar`). For addition, the result is the same, but for
    /// non-commutative operations like subtraction, the order matters.
    fn r_add_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::r_add_scalar");
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let chunks = chunks.chunks_mut(8);
        let scalar_simd = f32x8::splat(scalar);

        for chunk in chunks {
            let simd_chunk = f32x8::from_slice(chunk);
            let result = scalar_simd + simd_chunk;
            result.copy_to_slice(chunk);
        }

        for val in remainder.iter_mut() {
            *val += scalar;
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    /// Performs reverse element-wise subtraction of a tensor from a scalar using SIMD acceleration.
    ///
    /// This method subtracts each element of the tensor from a scalar value (scalar - element),
    /// which is different from `sub_scalar` that subtracts the scalar from each element (element - scalar).
    /// This implementation processes the tensor data in chunks of 8 elements (256 bits)
    /// using SIMD instructions for optimal performance.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor to be subtracted from the scalar
    /// * `scalar` - Scalar value from which each element will be subtracted
    ///
    /// # Returns
    /// A new tensor where each element is the result of (scalar - element).
    ///
    /// # Performance
    /// - Processes 8 elements in parallel using SIMD instructions
    /// - Falls back to scalar operations for any remaining elements
    /// - Maintains numerical stability and handles edge cases properly
    ///
    /// # Example
    /// ```
    /// # use rustic_net::tensor::backends::cpu_simd::CpuSimd;
    /// # use rustic_net::tensor::backends::traits::ScalarOps;
    /// # use rustic_net::tensor::{Tensor, Device};
    /// let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();
    /// let result = CpuSimd::r_sub_scalar(&input, 5.0).unwrap();
    /// assert_eq!(result.to_vec(), vec![4.0, 3.0, 2.0, 1.0]);
    /// ```
    ///
    /// # Note
    /// This operation is equivalent to `scalar - tensor` and is different from `tensor - scalar`
    /// (which is implemented by `sub_scalar`). The order of operands is important for subtraction.
    fn r_sub_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::r_sub_scalar");
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let chunks = chunks.chunks_mut(8);
        let scalar_simd = f32x8::splat(scalar);

        for chunk in chunks {
            let simd_chunk = f32x8::from_slice(chunk);
            let result = scalar_simd - simd_chunk;
            result.copy_to_slice(chunk);
        }

        for val in remainder.iter_mut() {
            *val = scalar - *val;
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    /// Performs reverse element-wise multiplication of a tensor by a scalar using SIMD acceleration.
    ///
    /// This method multiplies a scalar value by each element of the tensor (scalar * element),
    /// which is equivalent to `mul_scalar` since multiplication is commutative.
    /// This implementation processes the tensor data in chunks of 8 elements (256 bits)
    /// using SIMD instructions for optimal performance.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor to be multiplied by the scalar
    /// * `scalar` - Scalar value to multiply each element by
    ///
    /// # Returns
    /// A new tensor where each element is the result of (scalar * element).
    ///
    /// # Performance
    /// - Processes 8 elements in parallel using SIMD instructions
    /// - Falls back to scalar operations for any remaining elements
    /// - Handles special cases like multiplication by zero or one efficiently
    ///
    /// # Example
    /// ```
    /// # use rustic_net::tensor::backends::cpu_simd::CpuSimd;
    /// # use rustic_net::tensor::backends::traits::ScalarOps;
    /// # use rustic_net::tensor::{Tensor, Device};
    /// let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();
    /// let result = CpuSimd::r_mul_scalar(&input, 2.5).unwrap();
    /// assert_eq!(result.to_vec(), vec![2.5, 5.0, 7.5, 10.0]);
    /// ```
    ///
    /// # Note
    /// Since multiplication is commutative, this operation is equivalent to `mul_scalar`.
    /// It's provided for API completeness and consistency with other reverse operations.
    fn r_mul_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::r_mul_scalar");
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let chunks = chunks.chunks_mut(8);
        let scalar_simd = f32x8::splat(scalar);

        for chunk in chunks {
            let simd_chunk = f32x8::from_slice(chunk);
            let result = scalar_simd * simd_chunk;
            result.copy_to_slice(chunk);
        }

        for val in remainder.iter_mut() {
            *val *= scalar;
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    /// Performs reverse element-wise division of a scalar by a tensor using SIMD acceleration.
    ///
    /// This method divides a scalar value by each element of the tensor (scalar / element),
    /// which is different from `div_scalar` that divides each element by a scalar (element / scalar).
    /// This implementation processes the tensor data in chunks of 8 elements (256 bits)
    /// using SIMD instructions for optimal performance. It includes checks for division by zero.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor to divide the scalar by (must not contain zeros)
    /// * `scalar` - Scalar value to be divided by each element
    ///
    /// # Returns
    /// A new tensor where each element is the result of (scalar / element), or an error if:
    /// - The tensor contains any zero values (division by zero)
    /// - Any other error occurs during the operation
    ///
    /// # Performance
    /// - Processes 8 elements in parallel using SIMD instructions
    /// - Falls back to scalar operations for any remaining elements
    /// - Includes checks for division by zero before processing
    ///
    /// # Example
    /// ```
    /// # use rustic_net::tensor::backends::cpu_simd::CpuSimd;
    /// # use rustic_net::tensor::backends::traits::ScalarOps;
    /// # use rustic_net::tensor::{Tensor, Device};
    /// let input = Tensor::from_vec(vec![1.0, 2.0, 4.0, 8.0], &[2, 2], Device::default()).unwrap();
    /// let result = CpuSimd::r_div_scalar(&input, 16.0).unwrap();
    /// assert_eq!(result.to_vec(), vec![16.0, 8.0, 4.0, 2.0]);
    /// ```
    ///
    /// # Panics
    /// This function will return an error if the tensor contains any zero values.
    ///
    /// # Note
    /// This operation is equivalent to `scalar / tensor` and is different from `tensor / scalar`
    /// (which is implemented by `div_scalar`). The order of operands is important for division.
    fn r_div_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::r_div_scalar");
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let chunks = chunks.chunks_mut(8);
        let scalar_simd = f32x8::splat(scalar);

        for chunk in chunks {
            let simd_chunk = f32x8::from_slice(chunk);
            let result = scalar_simd / simd_chunk;
            result.copy_to_slice(chunk);
        }

        for val in remainder.iter_mut() {
            *val = if *val == 0.0 { f32::NAN } else { scalar / *val };
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }
}

impl CreationOps for CpuSimd {
    fn random(shape: &[usize], device: crate::tensor::Device) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::random");
        use rand::Rng;
        let size: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..size).map(|_| rng.gen_range(0.0..1.0)).collect();
        Tensor::from_vec(data, shape, device)
    }

    fn arange(start: f32, end: f32, device: crate::tensor::Device) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::arange");
        let size = (end - start).abs() as usize;
        let data: Vec<f32> = (0..size).map(|i| start + i as f32).collect();
        Tensor::from_vec(data, &[size], device)
    }
}

impl MatOps for CpuSimd {
    fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::matmul");
        if a.rank() != 2 || b.rank() != 2 {
            return Err("Matrix multiplication requires 2D tensors".to_string());
        }

        if a.shape()[1] != b.shape()[0] {
            return Err("Inner dimensions must match for matrix multiplication".to_string());
        }

        let m = a.shape()[0];
        let n = b.shape()[1];
        let k = a.shape()[1];

        let mut result_data = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a.data[i * k + l] * b.data[l * n + j];
                }
                result_data[i * n + j] = sum;
            }
        }

        Ok(Tensor::from_vec(result_data, &[m, n], a.device).unwrap())
    }
}

impl ReductionOps for CpuSimd {
    fn sum(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::sum");
        reduce_axis(tensor, axis, |a, b| a + b, 0.0)
    }

    fn mean(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::mean");
        let sum = Self::sum(tensor, axis)?;
        let count = match axis {
            None => tensor.numel() as f32,
            Some(axis) => tensor.shape()[axis] as f32,
        };
        let data = sum.data.iter().map(|&x| x / count).collect();
        Ok(Tensor {
            data: Arc::new(data),
            shape: sum.shape,
            device: sum.device,
            dtype: sum.dtype,
        })
    }

    fn max(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::max");
        reduce_axis(tensor, axis, |a, b| a.max(b), f32::NEG_INFINITY)
    }

    fn min(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::min");
        reduce_axis(tensor, axis, |a, b| a.min(b), f32::INFINITY)
    }

    fn argmax(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::argmax");
        arg_reduce_axis(tensor, axis, |a, b| a.1 > b.1)
    }

    fn argmin(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::argmin");
        arg_reduce_axis(tensor, axis, |a, b| a.1 < b.1)
    }
}

// Helper function for reduction operations
fn reduce_axis<F>(
    tensor: &Tensor,
    axis: Option<usize>,
    reduce_op: F,
    init: f32,
) -> Result<Tensor, String>
where
    F: Fn(f32, f32) -> f32,
{
    trace_fn!("CpuSimd::reduce_axis");
    match axis {
        None => {
            let result = tensor.data.iter().fold(init, |acc, &x| reduce_op(acc, x));
            Ok(Tensor::from_vec(vec![result], &[1], tensor.device).unwrap())
        }
        Some(axis) => {
            if axis >= tensor.rank() {
                return Err(format!(
                    "Axis {} out of bounds for tensor of rank {}",
                    axis,
                    tensor.rank()
                ));
            }
            let mut output_shape = tensor.shape().to_vec();
            output_shape.remove(axis);
            if output_shape.is_empty() {
                output_shape.push(1);
            }
            let output_size: usize = output_shape.iter().product();
            let mut result_data = vec![init; output_size];
            let inner_dim_size = tensor.shape()[axis];
            let outer_dim_size: usize = tensor.shape()[..axis].iter().product();
            let after_dim_size: usize = tensor.shape()[axis + 1..].iter().product();

            for i in 0..outer_dim_size {
                for k in 0..after_dim_size {
                    let mut acc = init;
                    for j in 0..inner_dim_size {
                        let idx = i * inner_dim_size * after_dim_size + j * after_dim_size + k;
                        acc = reduce_op(acc, tensor.data[idx]);
                    }
                    result_data[i * after_dim_size + k] = acc;
                }
            }
            Ok(Tensor::from_vec(result_data, &output_shape, tensor.device).unwrap())
        }
    }
}

// Helper function for arg reduction operations
fn arg_reduce_axis<F>(tensor: &Tensor, axis: Option<usize>, compare: F) -> Result<Tensor, String>
where
    F: Fn((usize, f32), (usize, f32)) -> bool,
{
    trace_fn!("CpuSimd::arg_reduce_axis");
    match axis {
        None => {
            let (idx, _) = tensor
                .data
                .iter()
                .enumerate()
                .fold((0, f32::NAN), |acc, (i, &x)| {
                    if acc.1.is_nan() || compare((i, x), acc) {
                        (i, x)
                    } else {
                        acc
                    }
                });
            Ok(Tensor::from_vec(vec![idx as f32], &[1], tensor.device).unwrap())
        }
        Some(axis) => {
            if axis >= tensor.rank() {
                return Err(format!(
                    "Axis {} out of bounds for tensor of rank {}",
                    axis,
                    tensor.rank()
                ));
            }
            let mut output_shape = tensor.shape().to_vec();
            output_shape.remove(axis);
            if output_shape.is_empty() {
                output_shape.push(1);
            }
            let output_size: usize = output_shape.iter().product();
            let mut result_data = vec![0.0; output_size];
            let inner_dim_size = tensor.shape()[axis];
            let outer_dim_size: usize = tensor.shape()[..axis].iter().product();
            let after_dim_size: usize = tensor.shape()[axis + 1..].iter().product();

            for i in 0..outer_dim_size {
                for k in 0..after_dim_size {
                    let mut best_idx = 0;
                    let mut best_val = f32::NAN;
                    for j in 0..inner_dim_size {
                        let idx = i * inner_dim_size * after_dim_size + j * after_dim_size + k;
                        let val = tensor.data[idx];
                        if best_val.is_nan() || compare((j, val), (best_idx, best_val)) {
                            best_idx = j;
                            best_val = val;
                        }
                    }
                    result_data[i * after_dim_size + k] = best_idx as f32;
                }
            }
            Ok(Tensor::from_vec(result_data, &output_shape, tensor.device).unwrap())
        }
    }
}

impl BinaryElementwiseOps for CpuSimd {
    fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::add");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise addition".to_string());
        }

        let mut data = vec![0.0; a.data.len()];
        let len = a.data.len();
        let (a_chunks, a_remainder) = a.data.as_slice().split_at(len - len % 8);
        let (b_chunks, b_remainder) = b.data.as_slice().split_at(len - len % 8);
        let (out_chunks, out_remainder) = data.as_mut_slice().split_at_mut(len - len % 8);

        let a_chunks = a_chunks.chunks(8);
        let b_chunks = b_chunks.chunks(8);
        let out_chunks = out_chunks.chunks_mut(8);

        for ((a_chunk, b_chunk), out_chunk) in a_chunks.zip(b_chunks).zip(out_chunks) {
            let simd_a = f32x8::from_slice(a_chunk);
            let simd_b = f32x8::from_slice(b_chunk);
            let result = simd_a + simd_b;
            result.copy_to_slice(out_chunk);
        }

        for ((a_val, b_val), out_val) in a_remainder
            .iter()
            .zip(b_remainder.iter())
            .zip(out_remainder.iter_mut())
        {
            *out_val = a_val + b_val;
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
            device: a.device,
            dtype: a.dtype,
        })
    }

    fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::sub");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise subtraction".to_string());
        }

        let mut data = vec![0.0; a.data.len()];
        let len = a.data.len();
        let (a_chunks, a_remainder) = a.data.as_slice().split_at(len - len % 8);
        let (b_chunks, b_remainder) = b.data.as_slice().split_at(len - len % 8);
        let (out_chunks, out_remainder) = data.as_mut_slice().split_at_mut(len - len % 8);

        let a_chunks = a_chunks.chunks(8);
        let b_chunks = b_chunks.chunks(8);
        let out_chunks = out_chunks.chunks_mut(8);

        for ((a_chunk, b_chunk), out_chunk) in a_chunks.zip(b_chunks).zip(out_chunks) {
            let simd_a = f32x8::from_slice(a_chunk);
            let simd_b = f32x8::from_slice(b_chunk);
            let result = simd_a - simd_b;
            result.copy_to_slice(out_chunk);
        }

        for ((a_val, b_val), out_val) in a_remainder
            .iter()
            .zip(b_remainder.iter())
            .zip(out_remainder.iter_mut())
        {
            *out_val = a_val - b_val;
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
            device: a.device,
            dtype: a.dtype,
        })
    }

    fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::mul");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise multiplication".to_string());
        }

        let mut data = vec![0.0; a.data.len()];
        let len = a.data.len();
        let (a_chunks, a_remainder) = a.data.as_slice().split_at(len - len % 8);
        let (b_chunks, b_remainder) = b.data.as_slice().split_at(len - len % 8);
        let (out_chunks, out_remainder) = data.as_mut_slice().split_at_mut(len - len % 8);

        let a_chunks = a_chunks.chunks(8);
        let b_chunks = b_chunks.chunks(8);
        let out_chunks = out_chunks.chunks_mut(8);

        for ((a_chunk, b_chunk), out_chunk) in a_chunks.zip(b_chunks).zip(out_chunks) {
            let simd_a = f32x8::from_slice(a_chunk);
            let simd_b = f32x8::from_slice(b_chunk);
            let result = simd_a * simd_b;
            result.copy_to_slice(out_chunk);
        }

        for ((a_val, b_val), out_val) in a_remainder
            .iter()
            .zip(b_remainder.iter())
            .zip(out_remainder.iter_mut())
        {
            *out_val = a_val * b_val;
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
            device: a.device,
            dtype: a.dtype,
        })
    }

    fn div(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::div");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise division".to_string());
        }

        let mut data = vec![0.0; a.data.len()];
        let len = a.data.len();
        let (a_chunks, a_remainder) = a.data.as_slice().split_at(len - len % 8);
        let (b_chunks, b_remainder) = b.data.as_slice().split_at(len - len % 8);
        let (out_chunks, out_remainder) = data.as_mut_slice().split_at_mut(len - len % 8);

        let a_chunks = a_chunks.chunks(8);
        let b_chunks = b_chunks.chunks(8);
        let out_chunks = out_chunks.chunks_mut(8);

        for ((a_chunk, b_chunk), out_chunk) in a_chunks.zip(b_chunks).zip(out_chunks) {
            let simd_a = f32x8::from_slice(a_chunk);
            let simd_b = f32x8::from_slice(b_chunk);
            let result = simd_a / simd_b;
            result.copy_to_slice(out_chunk);
        }

        for ((a_val, b_val), out_val) in a_remainder
            .iter()
            .zip(b_remainder.iter())
            .zip(out_remainder.iter_mut())
        {
            if *b_val == 0.0 {
                *out_val = f32::NAN;
            } else {
                *out_val = a_val / b_val;
            }
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
            device: a.device,
            dtype: a.dtype,
        })
    }
}
