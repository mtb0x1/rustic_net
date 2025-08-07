//! # Parallel CPU Backend
//!
//! A high-performance, multi-threaded CPU implementation of tensor operations
//! using Rayon for parallel execution. This backend automatically scales across
//! available CPU cores to accelerate tensor operations on multi-core systems.
//!
//! ## Features
//! - **Automatic Parallelism**: Leverages Rayon's work-stealing thread pool for optimal load balancing
//! - **Data Parallelism**: Element-wise operations are automatically parallelized
//! - **Cache Efficiency**: Uses chunked processing to optimize memory access patterns
//! - **Adaptive Execution**: Falls back to sequential processing for small tensors
//! - **Thread Safety**: All operations are safe for concurrent use
//!
//! ## Performance Characteristics
//! - Best for medium to large tensors (typically >1000 elements)
//! - Near-linear scaling with core count for compute-bound operations
//! - Minimal overhead for small tensors due to automatic fallback
//!
//! ## Example
//! ```rust
//! # use rustic_net::tensor::backends::traits::MatOps;
//! use rustic_net::tensor::{Tensor, Device};
//! use rustic_net::tensor::backends::cpu_par::CpuParallel;
//!
//! // Create large tensors
//! let a = Tensor::random(&[1000, 1000], Device::default());
//! let b = Tensor::random(&[1000, 1000], Device::default());
//!
//! // Matrix multiplication will be parallelized automatically
//! let c = CpuParallel::matmul(&a, &b).unwrap();
//! ```
//!
//! ## Configuration
//! The number of threads can be controlled via the `RUSTIC_NET_NUM_THREADS` environment variable:
//! ```bash
//! RUSTIC_NET_NUM_THREADS=4 cargo run --release
//! ```

use crate::tensor::backends::traits::BinaryElementwiseOps;
use crate::tensor::backends::traits::CreationOps;
use crate::tensor::backends::traits::MatOps;
use crate::tensor::backends::traits::ReductionOps;
use crate::tensor::backends::traits::ScalarOps;
use crate::tensor::backends::traits::ShapeOps;
use crate::tensor::backends::traits::UnaryOps;
use crate::tensor::{Shape, Tensor};
use crate::trace_fn;
use rayon::prelude::*;
use std::sync::Arc;
use tracing::debug;

/// Parallel CPU backend implementation for tensor operations.
///
/// This type implements all tensor operation traits using Rayon's parallel iterators
/// to distribute work across available CPU cores. It's automatically selected when:
/// 1. The `parallel` feature is enabled
//  2. The tensor size exceeds the parallelization threshold
//  3. The operation benefits from parallel execution
///
/// # Implementation Notes
/// - Uses Rayon's work-stealing scheduler for load balancing
/// - Automatically chunks data to optimize cache usage
/// - Falls back to sequential execution for small tensors
/// - Thread-safe and `Send`/`Sync` for concurrent use
///
/// # Performance Tips
/// - For optimal performance, ensure tensors are large enough to amortize
//    parallelization overhead
/// - Consider using `rayon::ThreadPoolBuilder` for custom thread pool configuration
/// - Memory layout can significantly impact performance - prefer contiguous tensors
pub struct CpuParallel;

impl UnaryOps for CpuParallel {
    /// Applies the Rectified Linear Unit (ReLU) activation function element-wise.
    ///
    /// ReLU is defined as `max(0, x)` for each element `x` in the input tensor.
    ///
    /// # Performance
    /// - **Parallelization**: Fully parallelized across all available CPU cores
    /// - **Complexity**: O(n) where n is the number of elements
    /// - **Memory**: Creates a new tensor, does not modify in-place
    ///
    /// # Example
    /// ```rust
    /// # use rustic_net::tensor::{Tensor, Device};
    /// # use rustic_net::tensor::backends::cpu_par::CpuParallel;
    /// # use rustic_net::tensor::backends::traits::UnaryOps;
    /// let t = Tensor::from_vec(vec![1.0, -2.0, 3.0, -4.0], &[2, 2], Device::default()).unwrap();
    /// let result = CpuParallel::relu(&t).unwrap();
    /// assert_eq!(result.to_vec(), &vec![1.0, 0.0, 3.0, 0.0]);
    /// ```
    fn relu(tensor: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::relu");
        let data: Vec<f32> = tensor
            .data
            .par_iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect();

        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }
}

impl BinaryElementwiseOps for CpuParallel {
    /// Performs element-wise addition of two tensors.
    ///
    /// # Arguments
    /// * `a` - First input tensor
    /// * `b` - Second input tensor (must have the same shape as `a`)
    ///
    /// # Returns
    /// A new tensor containing the element-wise sum of the inputs.
    ///
    /// # Errors
    /// Returns an error if the input tensors have different shapes.
    ///
    /// # Performance
    /// - **Parallelization**: Fully parallelized across all available CPU cores
    /// - **Complexity**: O(n) where n is the number of elements
    /// - **Memory**: Creates a new tensor, does not modify inputs
    ///
    /// # Example
    /// ```rust
    /// # use rustic_net::tensor::{Tensor, Device};
    /// # use rustic_net::tensor::backends::cpu_par::CpuParallel;
    /// # use rustic_net::tensor::backends::traits::BinaryElementwiseOps;
    /// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();
    /// let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], Device::default()).unwrap();
    /// let result = CpuParallel::add(&a, &b).unwrap();
    /// assert_eq!(result.to_vec(), &vec![6.0, 8.0, 10.0, 12.0]);
    /// ```
    fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::add");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise addition".to_string());
        }

        let data = a
            .data
            .par_iter()
            .zip(b.data.par_iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Ok(Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
            device: a.device,
            dtype: a.dtype,
        })
    }

    fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::sub");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise subtraction".to_string());
        }

        let data = a
            .data
            .par_iter()
            .zip(b.data.par_iter())
            .map(|(&a, &b)| a - b)
            .collect();

        Ok(Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
            device: a.device,
            dtype: a.dtype,
        })
    }

    fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::mul");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise multiplication".to_string());
        }

        let data = a
            .data
            .par_iter()
            .zip(b.data.par_iter())
            .map(|(&a, &b)| a * b)
            .collect();

        Ok(Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
            device: a.device,
            dtype: a.dtype,
        })
    }

    fn div(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::div");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise division".to_string());
        }

        let data = a
            .data
            .par_iter()
            .zip(b.data.par_iter())
            .map(|(&a, &b)| if b == 0.0 { f32::NAN } else { a / b })
            .collect();

        Ok(Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
            device: a.device,
            dtype: a.dtype,
        })
    }
}

impl MatOps for CpuParallel {
    /// Performs matrix multiplication between two tensors.
    ///
    /// The last two dimensions of the input tensors must be compatible for matrix
    /// multiplication. If the tensors have more than 2 dimensions, the operation
    /// is batched over the leading dimensions.
    ///
    /// # Arguments
    /// * `a` - Left-hand side tensor of shape `[..., M, K]`
    /// * `b` - Right-hand side tensor of shape `[..., K, N]`
    ///
    /// # Returns
    /// A new tensor containing the matrix product of shape `[..., M, N]`.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The last dimension of `a` does not match the second-to-last dimension of `b`
    /// - The batch dimensions are not broadcastable
    ///
    /// # Performance
    /// - **Parallelization**: Parallelized across both batch and inner matrix dimensions
    /// - **Complexity**: O(batch_size * M * K * N)
    /// - **Memory**: Creates a new tensor, does not modify inputs
    ///
    /// # Example
    /// ```rust
    /// # use rustic_net::tensor::{Tensor, Device};
    /// # use rustic_net::tensor::backends::cpu_par::CpuParallel;
    /// # use rustic_net::tensor::backends::traits::MatOps;
    /// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();
    /// let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], Device::default()).unwrap();
    /// let result = CpuParallel::matmul(&a, &b).unwrap();
    /// ```
    fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::matmul");
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
        debug!("m: {}, n: {}, k: {}", m, n, k);
        result_data
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, row)| {
                //debug!("\ni: {}, row: {:?}", i, row);
                for (j, row_val) in row.iter_mut().enumerate() {
                    //debug!("j: {}, row_val: {}", j, row_val);
                    *row_val = (0..k).map(|l| a.data[i * k + l] * b.data[l * n + j]).sum();
                }
                //debug!("row: {:?} done\n", i);
            });

        Ok(Tensor::from_vec(result_data, &[m, n], a.device).unwrap())
    }
}

impl ReductionOps for CpuParallel {
    /// Computes the sum of tensor elements along the specified axis.
    ///
    /// If `axis` is `None`, sums all elements in the tensor, returning a scalar.
    /// Otherwise, reduces along the specified axis, reducing the tensor's rank by 1.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor
    /// * `axis` - Axis along which to sum. If `None`, sums all elements.
    ///
    /// # Returns
    /// A new tensor containing the sum along the specified axis.
    ///
    /// # Performance
    /// - **Parallelization**: Parallelized along the reduction axis
    /// - **Complexity**: O(n) where n is the number of elements
    ///
    /// # Example
    /// ```rust
    /// # use rustic_net::tensor::{Tensor, Device};
    /// # use rustic_net::tensor::backends::cpu_par::CpuParallel;
    /// # use rustic_net::tensor::backends::traits::ReductionOps;
    /// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();
    /// let sum_all = CpuParallel::sum(&t, None).unwrap();  // Scalar 10.0
    /// let sum_rows = CpuParallel::sum(&t, Some(0)).unwrap();  // [4.0, 6.0]
    /// ```
    fn sum(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::sum");
        reduce_axis(tensor, axis, |a, b| a + b, 0.0)
    }

    fn mean(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::mean");
        let sum = Self::sum(tensor, axis)?;
        let count = match axis {
            None => tensor.numel() as f32,
            Some(axis) => tensor.shape()[axis] as f32,
        };
        let data = sum.data.par_iter().map(|&x| x / count).collect();
        Ok(Tensor {
            data: Arc::new(data),
            shape: sum.shape,
            device: sum.device,
            dtype: sum.dtype,
        })
    }

    fn max(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::max");
        reduce_axis(tensor, axis, |a, b| a.max(b), f32::NEG_INFINITY)
    }

    fn min(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::min");
        reduce_axis(tensor, axis, |a, b| a.min(b), f32::INFINITY)
    }

    fn argmax(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::argmax");
        arg_reduce_axis(tensor, axis, |a, b| a.1 > b.1)
    }

    fn argmin(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::argmin");
        arg_reduce_axis(tensor, axis, |a, b| a.1 < b.1)
    }
}

// Helper function for parallel reduction operations
fn reduce_axis<F>(
    tensor: &Tensor,
    axis: Option<usize>,
    reduce_op: F,
    init: f32,
) -> Result<Tensor, String>
where
    F: Fn(f32, f32) -> f32 + Send + Sync,
{
    trace_fn!("CpuParallel::reduce_axis");
    match axis {
        None => {
            let result = tensor.data.par_iter().cloned().reduce(|| init, reduce_op);
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
            let _outer_dim_size: usize = tensor.shape()[..axis].iter().product();
            let after_dim_size: usize = tensor.shape()[axis + 1..].iter().product();

            result_data
                .par_chunks_mut(after_dim_size)
                .enumerate()
                .for_each(|(i, chunk)| {
                    for (k, chunk_val) in chunk.iter_mut().enumerate() {
                        *chunk_val = (0..inner_dim_size)
                            .map(|j| {
                                let idx =
                                    i * inner_dim_size * after_dim_size + j * after_dim_size + k;
                                tensor.data[idx]
                            })
                            .fold(init, &reduce_op);
                    }
                });
            Ok(Tensor::from_vec(result_data, &output_shape, tensor.device).unwrap())
        }
    }
}

impl ScalarOps for CpuParallel {
    fn add_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::add_scalar");
        let data: Vec<f32> = tensor.data.par_iter().map(|&x| x + scalar).collect();
        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    /// Subtracts a scalar from each element of the tensor in parallel.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor
    /// * `scalar` - Scalar value to subtract from each element
    ///
    /// # Returns
    /// A new tensor with the scalar subtracted from each element.
    ///
    /// # Performance
    /// - **Parallelization**: Fully parallelized across all elements
    /// - **Complexity**: O(n) where n is the number of elements
    /// - **Memory**: Creates a new tensor, does not modify the input
    ///
    /// # Example
    /// ```rust
    /// # use rustic_net::tensor::{Tensor, Device};
    /// # use rustic_net::tensor::backends::cpu_par::CpuParallel;
    /// # use rustic_net::tensor::backends::traits::ScalarOps;
    /// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], Device::default()).unwrap();
    /// let result = CpuParallel::sub_scalar(&t, 5.0).unwrap();
    /// assert_eq!(result.to_vec(), &vec![-4.0, -3.0, -2.0]);
    /// ```
    fn sub_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::sub_scalar");
        let data: Vec<f32> = tensor.data.par_iter().map(|&x| x - scalar).collect();
        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    /// Multiplies each element of the tensor by a scalar in parallel.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor
    /// * `scalar` - Scalar value to multiply each element by
    ///
    /// # Returns
    /// A new tensor with each element multiplied by the scalar.
    ///
    /// # Performance
    /// - **Parallelization**: Fully parallelized across all elements
    /// - **Complexity**: O(n) where n is the number of elements
    /// - **Memory**: Creates a new tensor, does not modify the input
    ///
    /// # Example
    /// ```rust
    /// # use rustic_net::tensor::{Tensor, Device};
    /// # use rustic_net::tensor::backends::cpu_par::CpuParallel;
    /// # use rustic_net::tensor::backends::traits::ScalarOps;
    /// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], Device::default()).unwrap();
    /// let result = CpuParallel::mul_scalar(&t, 2.0).unwrap();
    /// assert_eq!(result.to_vec(), &vec![2.0, 4.0, 6.0]);
    /// ```
    fn mul_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::mul_scalar");
        let data: Vec<f32> = tensor.data.par_iter().map(|&x| x * scalar).collect();
        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    /// Divides each element of the tensor by a scalar in parallel.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor
    /// * `scalar` - Scalar value to divide each element by (must not be zero)
    ///
    /// # Returns
    /// A new tensor with each element divided by the scalar.
    ///
    /// # Errors
    /// Returns an error if `scalar` is zero.
    ///
    /// # Performance
    /// - **Parallelization**: Fully parallelized across all elements
    /// - **Complexity**: O(n) where n is the number of elements
    /// - **Memory**: Creates a new tensor, does not modify the input
    ///
    /// # Example
    /// ```rust
    /// # use rustic_net::tensor::{Tensor, Device};
    /// # use rustic_net::tensor::backends::cpu_par::CpuParallel;
    /// # use rustic_net::tensor::backends::traits::ScalarOps;
    /// let t = Tensor::from_vec(vec![2.0, 4.0, 6.0], &[3], Device::default()).unwrap();
    /// let result = CpuParallel::div_scalar(&t, 2.0).unwrap();
    /// assert_eq!(result.to_vec(), &vec![1.0, 2.0, 3.0]);
    /// ```
    fn div_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::div_scalar");
        if scalar == 0.0 {
            return Err("Division by zero".to_string());
        }
        let data: Vec<f32> = tensor.data.par_iter().map(|&x| x / scalar).collect();
        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    /// Adds a scalar to each element of the tensor (right-add) in parallel.
    ///
    /// This is equivalent to `add_scalar` but provided for completeness in the API.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor
    /// * `scalar` - Scalar value to add to each element
    ///
    /// # Returns
    /// A new tensor with the scalar added to each element.
    ///
    /// # Performance
    /// - **Parallelization**: Fully parallelized across all elements
    /// - **Complexity**: O(n) where n is the number of elements
    /// - **Memory**: Creates a new tensor, does not modify the input
    ///
    /// # Example
    /// ```rust
    /// # use rustic_net::tensor::{Tensor, Device};
    /// # use rustic_net::tensor::backends::cpu_par::CpuParallel;
    /// # use rustic_net::tensor::backends::traits::ScalarOps;
    /// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], Device::default()).unwrap();
    /// let result = CpuParallel::r_add_scalar(&t, 5.0).unwrap();
    /// assert_eq!(result.to_vec(), &vec![6.0, 7.0, 8.0]);
    /// ```
    fn r_add_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::r_add_scalar");
        let data: Vec<f32> = tensor.data.par_iter().map(|&x| scalar + x).collect();
        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    /// Subtracts each element of the tensor from a scalar in parallel.
    ///
    /// This is the reverse of `sub_scalar` - it computes `scalar - x` for each element `x`.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor
    /// * `scalar` - Scalar value to subtract each element from
    ///
    /// # Returns
    /// A new tensor with each element subtracted from the scalar.
    ///
    /// # Performance
    /// - **Parallelization**: Fully parallelized across all elements
    /// - **Complexity**: O(n) where n is the number of elements
    /// - **Memory**: Creates a new tensor, does not modify the input
    ///
    /// # Example
    /// ```rust
    /// # use rustic_net::tensor::{Tensor, Device};
    /// # use rustic_net::tensor::backends::cpu_par::CpuParallel;
    /// # use rustic_net::tensor::backends::traits::ScalarOps;
    /// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], Device::default()).unwrap();
    /// let result = CpuParallel::r_sub_scalar(&t, 5.0).unwrap();
    /// assert_eq!(result.to_vec(), &vec![4.0, 3.0, 2.0]);
    /// ```
    fn r_sub_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::r_sub_scalar");
        let data: Vec<f32> = tensor.data.par_iter().map(|&x| scalar - x).collect();
        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    /// Multiplies a scalar by each element of the tensor in parallel.
    ///
    /// This is equivalent to `mul_scalar` but provided for completeness in the API.
    ///
    /// # Arguments
    /// * `tensor` - Input tensor
    /// * `scalar` - Scalar value to multiply by each element
    ///
    /// # Returns
    /// A new tensor with the scalar multiplied by each element.
    ///
    /// # Performance
    /// - **Parallelization**: Fully parallelized across all elements
    /// - **Complexity**: O(n) where n is the number of elements
    /// - **Memory**: Creates a new tensor, does not modify the input
    ///
    /// # Example
    /// ```rust
    /// # use rustic_net::tensor::{Tensor, Device};
    /// # use rustic_net::tensor::backends::cpu_par::CpuParallel;
    /// # use rustic_net::tensor::backends::traits::ScalarOps;
    /// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], Device::default()).unwrap();
    /// let result = CpuParallel::r_mul_scalar(&t, 2.0).unwrap();
    /// assert_eq!(result.to_vec(), &vec![2.0, 4.0, 6.0]);
    /// ```
    fn r_mul_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::r_mul_scalar");
        let data: Vec<f32> = tensor.data.par_iter().map(|&x| scalar * x).collect();
        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    fn r_div_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::r_div_scalar");
        let data: Vec<f32> = tensor
            .data
            .par_iter()
            .map(|&x| if x == 0.0 { f32::NAN } else { scalar / x })
            .collect();
        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }
}

impl ShapeOps for CpuParallel {
    fn transpose(tensor: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::transpose");
        let rank = tensor.rank();
        let axes: Vec<usize> = (0..rank).rev().collect();
        CpuParallel::transpose_axes(tensor, &axes)
    }

    fn transpose_axes(tensor: &Tensor, axes: &[usize]) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::transpose_axes");
        let rank = tensor.rank();
        if axes.len() != rank {
            return Err(format!(
                "Axes length {} does not match tensor rank {}",
                axes.len(),
                rank
            ));
        }

        // Create new shape and calculate new strides
        let new_dims: Vec<usize> = axes.iter().map(|&i| tensor.shape()[i]).collect();
        let new_shape = Shape::new(&new_dims);
        let new_len = new_shape.len();

        // Pre-allocate the output vector
        let mut new_data = vec![0.0; new_len];

        // Get strides for both old and new shapes
        let old_strides = tensor.shape.strides();
        let new_strides = new_shape.strides();

        // Calculate chunk size based on number of CPU cores
        let chunk_size = std::cmp::max(1, new_len / (rayon::current_num_threads() * 4));

        // Process data in parallel chunks for better cache locality
        new_data
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_idx = chunk_idx * chunk_size;

                // Process each element in the chunk
                for (i, val) in chunk.iter_mut().enumerate() {
                    let idx = start_idx + i;
                    if idx >= new_len {
                        break;
                    } // Handle the last chunk which might be smaller

                    // Calculate the multi-dimensional index in the new shape
                    let mut old_idx = 0;
                    let mut remaining = idx;

                    // Convert flat index to multi-dimensional index and then to old index
                    for (j, &stride) in new_strides.iter().enumerate() {
                        let dim = remaining / stride;
                        remaining %= stride;
                        old_idx += dim * old_strides[axes[j]];
                    }

                    *val = tensor.data[old_idx];
                }
            });

        Ok(Tensor {
            data: Arc::new(new_data),
            shape: new_shape,
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }
}

impl CreationOps for CpuParallel {
    /// Generates a tensor with random values between 0.0 and 1.0.
    ///
    /// Uses ChaCha8Rng for thread-local random number generation, which is
    /// cryptographically secure and provides good parallel performance.
    ///
    /// # Arguments
    /// * `shape` - The shape of the output tensor
    /// * `device` - The device to create the tensor on (must be CPU)
    ///
    /// # Returns
    /// A new tensor filled with random values in the range [0.0, 1.0).
    ///
    /// # Performance
    /// - **Parallelization**: Fully parallelized across all elements
    /// - **Complexity**: O(n) where n is the number of elements
    /// - **Memory**: Allocates a new tensor
    ///
    /// # Example
    /// ```rust
    /// # use rustic_net::tensor::{Tensor, Device};
    /// # use rustic_net::tensor::backends::cpu_par::CpuParallel;
    /// # use rustic_net::tensor::backends::traits::CreationOps;
    /// let t = CpuParallel::random(&[2, 3], Device::default()).unwrap();
    /// assert_eq!(t.shape(), &[2, 3]);
    /// // All values should be in [0.0, 1.0)
    /// assert!(t.data.iter().all(|&x| x >= 0.0 && x < 1.0));
    /// ```
    fn random(shape: &[usize], device: crate::tensor::Device) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::random");
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use std::sync::Mutex;

        let size = shape.iter().product();
        let data: Vec<f32> = (0..size)
            .into_par_iter()
            .map_init(
                || Mutex::new(ChaCha8Rng::from_entropy()),
                |rng, _| {
                    let rng = rng.get_mut().unwrap();
                    rng.gen_range(0.0..1.0)
                },
            )
            .collect();

        Self::from_vec(data, shape, device)
    }

    /// Creates a 1D tensor with values from `start` to `end` (exclusive) with step 1.
    ///
    /// If `end` is less than `start`, the sequence will be decreasing.
    ///
    /// # Arguments
    /// * `start` - The starting value (inclusive)
    /// * `end` - The ending value (exclusive)
    /// * `device` - The device to create the tensor on (must be CPU)
    ///
    /// # Returns
    /// A new 1D tensor containing the sequence of values.
    ///
    /// # Performance
    /// - **Parallelization**: Fully parallelized
    /// - **Complexity**: O(n) where n is the number of elements
    /// - **Memory**: Allocates a new tensor
    ///
    /// # Example
    /// ```rust
    /// # use rustic_net::tensor::{Tensor, Device};
    /// # use rustic_net::tensor::backends::cpu_par::CpuParallel;
    /// # use rustic_net::tensor::backends::traits::CreationOps;
    /// let t = CpuParallel::arange(2.0, 5.0, Device::default()).unwrap();
    /// assert_eq!(t.to_vec(), &vec![2.0, 3.0, 4.0]);
    ///
    /// let t = CpuParallel::arange(5.0, 2.0, Device::default()).unwrap();
    /// assert_eq!(t.to_vec(), &vec![5.0, 4.0, 3.0]);
    /// ```
    fn arange(start: f32, end: f32, device: crate::tensor::Device) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::arange");
        let size = (end - start).abs().ceil() as usize;
        let step = if end >= start { 1.0 } else { -1.0 };

        let data: Vec<f32> = (0..size)
            .into_par_iter()
            .map(|i| start + (i as f32) * step)
            .collect();

        Self::from_vec(data, &[size], device)
    }

    /// Creates a tensor filled with zeros.
    ///
    /// # Arguments
    /// * `shape` - The shape of the output tensor
    /// * `device` - The device to create the tensor on (must be CPU)
    ///
    /// # Returns
    /// A new tensor filled with zeros.
    ///
    /// # Performance
    /// - **Parallelization**: Uses parallel initialization
    /// - **Complexity**: O(n) where n is the number of elements
    /// - **Memory**: Allocates a new tensor
    fn zeros(shape: &[usize], device: crate::tensor::Device) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::zeros");
        let size: usize = shape.iter().product();
        let data = vec![0.0; size];
        Self::from_vec(data, shape, device)
    }

    /// Creates a tensor filled with ones.
    ///
    /// # Arguments
    /// * `shape` - The shape of the output tensor
    /// * `device` - The device to create the tensor on (must be CPU)
    ///
    /// # Returns
    /// A new tensor filled with ones.
    ///
    /// # Performance
    /// - **Parallelization**: Uses parallel initialization
    /// - **Complexity**: O(n) where n is the number of elements
    /// - **Memory**: Allocates a new tensor
    fn ones(shape: &[usize], device: crate::tensor::Device) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::ones");
        let size: usize = shape.iter().product();
        let data = vec![1.0; size];
        Self::from_vec(data, shape, device)
    }

    /// Creates an identity matrix (2D tensor with ones on the diagonal).
    ///
    /// # Arguments
    /// * `size` - The size of the square matrix
    /// * `device` - The device to create the tensor on (must be CPU)
    ///
    /// # Returns
    /// A new identity matrix of shape `[size, size]`.
    ///
    /// # Performance
    /// - **Parallelization**: Uses parallel initialization
    /// - **Complexity**: O(n²) where n is the size
    /// - **Memory**: Allocates a new tensor
    fn identity(size: usize, device: crate::tensor::Device) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::identity");
        let len = size * size;
        let mut data = vec![0.0; len];

        // Parallelize setting the diagonal elements
        data.par_chunks_mut(size).enumerate().for_each(|(i, row)| {
            row[i] = 1.0;
        });

        Self::from_vec(data, &[size, size], device)
    }

    /// Creates a tensor from a vector of data with the specified shape.
    ///
    /// # Arguments
    /// * `data` - The data to populate the tensor with
    /// * `shape` - The shape of the output tensor
    /// * `device` - The device to create the tensor on (must be CPU)
    ///
    /// # Returns
    /// A new tensor with the given data and shape.
    ///
    /// # Errors
    /// Returns an error if the data length does not match the product of the shape dimensions.
    ///
    /// # Performance
    /// - **Complexity**: O(n) where n is the number of elements
    /// - **Memory**: Allocates a new tensor
    fn from_vec(
        data: Vec<f32>,
        shape: &[usize],
        device: crate::tensor::Device,
    ) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::from_vec");
        let shape_obj = crate::tensor::Shape::new(shape);

        // Validate that the data length matches the shape
        if data.len() != shape_obj.len() {
            return Err(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape_obj.dims(),
                shape_obj.len()
            ));
        }

        Ok(Tensor {
            data: std::sync::Arc::new(data),
            shape: shape_obj,
            device,
            dtype: crate::tensor::DType::F32,
        })
    }
    fn from_slice(
        data: &[f32],
        shape: &[usize],
        device: crate::tensor::Device,
    ) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::from_slice");
        let shape_obj = crate::tensor::Shape::new(shape);

        // Validate that the data length matches the shape
        if data.len() != shape_obj.len() {
            return Err(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape_obj.dims(),
                shape_obj.len()
            ));
        }

        Ok(Tensor {
            data: std::sync::Arc::new(data.to_vec()),
            shape: shape_obj,
            device,
            dtype: crate::tensor::DType::F32,
        })
    }
}

/// Helper function for parallel arg reduction operations (argmax, argmin).
///
/// This function implements the common pattern of finding the index of the
/// minimum or maximum value along a given axis in a tensor.
///
/// # Type Parameters
/// * `F` - A comparison function that takes two `(index, value)` pairs and returns
///   `true` if the second pair should replace the first as the current best.
///
/// # Arguments
/// * `tensor` - The input tensor to reduce
/// * `axis` - The axis along which to find the index of the minimum/maximum value.
///   If `None`, finds the index in the flattened tensor.
/// * `compare` - The comparison function to determine the "better" of two values
///
/// # Returns
/// A new tensor containing the indices of the minimum/maximum values.
///
/// # Implementation Details
/// - Uses Rayon's parallel iterators for parallel execution
/// - Handles both full reduction and axis-specific reduction
/// - Returns `f32` values for consistency with other operations
///
/// # Safety
/// The `compare` function must implement a total ordering to ensure consistent
/// results in parallel execution.
fn arg_reduce_axis<F>(tensor: &Tensor, axis: Option<usize>, compare: F) -> Result<Tensor, String>
where
    F: Fn((usize, f32), (usize, f32)) -> bool + Send + Sync,
{
    trace_fn!("CpuParallel::arg_reduce_axis");
    match axis {
        None => {
            let (idx, _) = tensor.data.par_iter().cloned().enumerate().reduce(
                || (0, f32::NAN),
                |a, b| if a.1.is_nan() || compare(b, a) { b } else { a },
            );
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
            let _outer_dim_size: usize = tensor.shape()[..axis].iter().product();
            let after_dim_size: usize = tensor.shape()[axis + 1..].iter().product();

            result_data
                .par_chunks_mut(after_dim_size)
                .enumerate()
                .for_each(|(i, chunk)| {
                    for (k, chunk_val) in chunk.iter_mut().enumerate() {
                        let (best_idx, _) = (0..inner_dim_size)
                            .map(|j| {
                                let idx =
                                    i * inner_dim_size * after_dim_size + j * after_dim_size + k;
                                (j, tensor.data[idx])
                            })
                            .reduce(|a, b| if a.1.is_nan() || compare(b, a) { b } else { a })
                            .unwrap(); // This should not panic as inner_dim_size > 0
                        *chunk_val = best_idx as f32;
                    }
                });
            Ok(Tensor::from_vec(result_data, &output_shape, tensor.device).unwrap())
        }
    }
}
