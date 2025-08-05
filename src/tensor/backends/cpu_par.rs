//! # Parallel CPU Backend
//!
//! A multi-threaded CPU implementation of tensor operations using Rayon for parallel execution.
//! Automatically scales across available CPU cores for improved performance on large tensors.
//!
//! ## Features
//! - Automatic work-stealing thread pool for load balancing
//! - Data parallelism for element-wise operations
//! - Chunked processing for cache efficiency
//! - Fallback to sequential execution for small tensors

use super::traits::*;
use crate::tensor::Tensor;
use crate::trace_fn;
use rayon::prelude::*;
use std::sync::Arc;

/// Marker type for the parallel CPU backend.
///
/// Implements all tensor operation traits using Rayon's parallel iterators.
/// Automatically selected when the `parallel` feature is enabled and the tensor
/// size exceeds the parallelization threshold.
pub struct CpuParallel;

use std::simd::{cmp::SimdPartialOrd, f32x8};

impl UnaryOps for CpuParallel {
    fn relu(tensor: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::relu");
        let mut data = tensor.data.to_vec();

        #[cfg(feature = "simd_and_parallel")]
        {
            data.par_chunks_mut(8).for_each(|chunk| {
                let simd_chunk = f32x8::from_slice(chunk);
                let mask = simd_chunk.simd_gt(f32x8::splat(0.0));
                let result = mask.select(simd_chunk, f32x8::splat(0.0));
                result.copy_to_slice(chunk);
            });
        }
        #[cfg(not(feature = "simd_and_parallel"))]
        {
            data = tensor
                .data
                .par_iter()
                .map(|&x| if x > 0.0 { x } else { 0.0 })
                .collect();
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }
}

impl BinaryElementwiseOps for CpuParallel {
    fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::add");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise addition".to_string());
        }

        let mut data = vec![0.0; a.data.len()];
        #[cfg(feature = "simd_and_parallel")]
        {
            data.par_chunks_mut(8)
                .zip(a.data.par_chunks(8))
                .zip(b.data.par_chunks(8))
                .for_each(|((out_chunk, a_chunk), b_chunk)| {
                    let simd_a = f32x8::from_slice(a_chunk);
                    let simd_b = f32x8::from_slice(b_chunk);
                    let result = simd_a + simd_b;
                    result.copy_to_slice(out_chunk);
                });
        }
        #[cfg(not(feature = "simd_and_parallel"))]
        {
            data = a
                .data
                .par_iter()
                .zip(b.data.par_iter())
                .map(|(&a, &b)| a + b)
                .collect();
        }

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

        let mut data = vec![0.0; a.data.len()];
        #[cfg(feature = "simd_and_parallel")]
        {
            data.par_chunks_mut(8)
                .zip(a.data.par_chunks(8))
                .zip(b.data.par_chunks(8))
                .for_each(|((out_chunk, a_chunk), b_chunk)| {
                    let simd_a = f32x8::from_slice(a_chunk);
                    let simd_b = f32x8::from_slice(b_chunk);
                    let result = simd_a - simd_b;
                    result.copy_to_slice(out_chunk);
                });
        }
        #[cfg(not(feature = "simd_and_parallel"))]
        {
            data = a
                .data
                .par_iter()
                .zip(b.data.par_iter())
                .map(|(&a, &b)| a - b)
                .collect();
        }

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

        let mut data = vec![0.0; a.data.len()];
        #[cfg(feature = "simd_and_parallel")]
        {
            data.par_chunks_mut(8)
                .zip(a.data.par_chunks(8))
                .zip(b.data.par_chunks(8))
                .for_each(|((out_chunk, a_chunk), b_chunk)| {
                    let simd_a = f32x8::from_slice(a_chunk);
                    let simd_b = f32x8::from_slice(b_chunk);
                    let result = simd_a * simd_b;
                    result.copy_to_slice(out_chunk);
                });
        }
        #[cfg(not(feature = "simd_and_parallel"))]
        {
            data = a
                .data
                .par_iter()
                .zip(b.data.par_iter())
                .map(|(&a, &b)| a * b)
                .collect();
        }

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

        let mut data = vec![0.0; a.data.len()];
        #[cfg(feature = "simd_and_parallel")]
        {
            data.par_chunks_mut(8)
                .zip(a.data.par_chunks(8))
                .zip(b.data.par_chunks(8))
                .for_each(|((out_chunk, a_chunk), b_chunk)| {
                    let simd_a = f32x8::from_slice(a_chunk);
                    let simd_b = f32x8::from_slice(b_chunk);
                    let result = simd_a / simd_b;
                    result.copy_to_slice(out_chunk);
                });
        }
        #[cfg(not(feature = "simd_and_parallel"))]
        {
            data = a
                .data
                .par_iter()
                .zip(b.data.par_iter())
                .map(|(&a, &b)| if b == 0.0 { f32::NAN } else { a / b })
                .collect();
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
            device: a.device,
            dtype: a.dtype,
        })
    }
}

impl MatOps for CpuParallel {
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
        result_data
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, row)| {
                for (j, row_val) in row.iter_mut().enumerate() {
                    *row_val = (0..k).map(|l| a.data[i * k + l] * b.data[l * n + j]).sum();
                }
            });

        Ok(Tensor::from_vec(result_data, &[m, n], a.device).unwrap())
    }
}

impl ReductionOps for CpuParallel {
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

// Helper function for parallel arg reduction operations
fn arg_reduce_axis<F>(tensor: &Tensor, axis: Option<usize>, compare: F) -> Result<Tensor, String>
where
    F: Fn((usize, f32), (usize, f32)) -> bool + Send + Sync,
{
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
