//! # SIMD and Parallel CPU Backend
//!
//! A multi-threaded, SIMD-accelerated CPU implementation of tensor operations.
//! This backend leverages both Rayon for parallelism and explicit SIMD intrinsics
//! for performance-critical element-wise operations.
//!
//! ## Features
//! - Combines data parallelism (SIMD) and task parallelism (Rayon)
//! - Optimized for CPUs with AVX support
//! - Ideal for large-scale numerical computations on modern CPUs

use super::traits::*;
use crate::tensor::{Shape, Tensor};
use crate::trace_fn;
use rayon::prelude::*;
use std::simd::num::SimdFloat;
use std::simd::StdFloat;
use std::simd::{cmp::SimdPartialOrd, f32x8};
use std::sync::Arc;

/// Marker type for the SIMD and parallel CPU backend.
///
/// This struct implements tensor operation traits using a combination of
/// Rayon's parallel iterators and explicit SIMD instructions. It is selected
/// when both the `simd` and `parallel` features are enabled.
pub struct CpuSimdPar;

impl UnaryOps for CpuSimdPar {
    fn relu(tensor: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::relu");
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);

        chunks.par_chunks_mut(8).for_each(|chunk| {
            let simd_chunk = f32x8::from_slice(chunk);
            let mask = simd_chunk.simd_gt(f32x8::splat(0.0));
            let result = mask.select(simd_chunk, f32x8::splat(0.0));
            result.copy_to_slice(chunk);
        });

        for val in remainder.iter_mut() {
            if *val < 0.0 {
                *val = 0.0;
            }
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }
}

impl BinaryElementwiseOps for CpuSimdPar {
    fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::add");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise addition".to_string());
        }

        let mut data = vec![0.0; a.data.len()];
        let len = a.data.len();

        let (a_chunks, a_remainder) = a.data.as_slice().split_at(len - len % 8);
        let (b_chunks, b_remainder) = b.data.as_slice().split_at(len - len % 8);
        let (out_chunks, out_remainder) = data.as_mut_slice().split_at_mut(len - len % 8);

        out_chunks
            .par_chunks_mut(8)
            .zip(a_chunks.par_chunks(8))
            .zip(b_chunks.par_chunks(8))
            .for_each(|((out_chunk, a_chunk), b_chunk)| {
                let simd_a = f32x8::from_slice(a_chunk);
                let simd_b = f32x8::from_slice(b_chunk);
                let result = simd_a + simd_b;
                result.copy_to_slice(out_chunk);
            });

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
        trace_fn!("CpuSimdPar::sub");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise subtraction".to_string());
        }

        let mut data = vec![0.0; a.data.len()];
        let len = a.data.len();

        let (a_chunks, a_remainder) = a.data.as_slice().split_at(len - len % 8);
        let (b_chunks, b_remainder) = b.data.as_slice().split_at(len - len % 8);
        let (out_chunks, out_remainder) = data.as_mut_slice().split_at_mut(len - len % 8);

        out_chunks
            .par_chunks_mut(8)
            .zip(a_chunks.par_chunks(8))
            .zip(b_chunks.par_chunks(8))
            .for_each(|((out_chunk, a_chunk), b_chunk)| {
                let simd_a = f32x8::from_slice(a_chunk);
                let simd_b = f32x8::from_slice(b_chunk);
                let result = simd_a - simd_b;
                result.copy_to_slice(out_chunk);
            });

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
        trace_fn!("CpuSimdPar::mul");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise multiplication".to_string());
        }

        let mut data = vec![0.0; a.data.len()];
        let len = a.data.len();

        let (a_chunks, a_remainder) = a.data.as_slice().split_at(len - len % 8);
        let (b_chunks, b_remainder) = b.data.as_slice().split_at(len - len % 8);
        let (out_chunks, out_remainder) = data.as_mut_slice().split_at_mut(len - len % 8);

        out_chunks
            .par_chunks_mut(8)
            .zip(a_chunks.par_chunks(8))
            .zip(b_chunks.par_chunks(8))
            .for_each(|((out_chunk, a_chunk), b_chunk)| {
                let simd_a = f32x8::from_slice(a_chunk);
                let simd_b = f32x8::from_slice(b_chunk);
                let result = simd_a * simd_b;
                result.copy_to_slice(out_chunk);
            });

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
        trace_fn!("CpuSimdPar::div");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise division".to_string());
        }

        let mut data = vec![0.0; a.data.len()];
        let len = a.data.len();

        let (a_chunks, a_remainder) = a.data.as_slice().split_at(len - len % 8);
        let (b_chunks, b_remainder) = b.data.as_slice().split_at(len - len % 8);
        let (out_chunks, out_remainder) = data.as_mut_slice().split_at_mut(len - len % 8);

        out_chunks
            .par_chunks_mut(8)
            .zip(a_chunks.par_chunks(8))
            .zip(b_chunks.par_chunks(8))
            .for_each(|((out_chunk, a_chunk), b_chunk)| {
                let simd_a = f32x8::from_slice(a_chunk);
                let simd_b = f32x8::from_slice(b_chunk);
                let result = simd_a / simd_b;
                result.copy_to_slice(out_chunk);
            });

        for ((a_val, b_val), out_val) in a_remainder
            .iter()
            .zip(b_remainder.iter())
            .zip(out_remainder.iter_mut())
        {
            *out_val = if *b_val == 0.0 {
                f32::NAN
            } else {
                a_val / b_val
            };
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
            device: a.device,
            dtype: a.dtype,
        })
    }
}

impl MatOps for CpuSimdPar {
    fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::matmul");
        if a.rank() != 2 || b.rank() != 2 {
            return Err("Matrix multiplication requires 2D tensors".to_string());
        }

        let m = a.shape()[0];
        let n = b.shape()[1];
        let k = a.shape()[1]; // Inner dimension

        if k != b.shape()[0] {
            return Err(format!(
                "Inner dimensions must match for matrix multiplication: A is [{}, {}], B is [{}, {}]",
                m, k, b.shape()[0], n
            ));
        }

        // --- Optimization Step 1: Transpose matrix B ---
        // This makes column access from B become contiguous row access in B_t.
        // This is crucial for cache performance and effective SIMD.
        let mut b_t_data = vec![0.0; k * n];
        for l in 0..k {
            for j in 0..n {
                b_t_data[j * k + l] = b.data[l * n + j];
            }
        }

        let mut result_data = vec![0.0; m * n];

        // --- Optimization Step 2: Parallelize over the rows of the output matrix ---
        // This is the same excellent parallel structure you already had.
        result_data
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, result_row)| {
                // Get the i-th row of A. This will be reused for all columns in the result.
                let a_row = &a.data[i * k..(i + 1) * k];

                for j in 0..n {
                    // Get the j-th row of the transposed B matrix.
                    let b_t_row = &b_t_data[j * k..(j + 1) * k];

                    // --- Optimization Step 3: SIMD dot product ---
                    const SIMD_WIDTH: usize = 8*2*2; // For f32x8
                    let mut sum_vec = f32x8::splat(0.0);

                    // Process the bulk of the data in SIMD chunks
                    let chunks = k / SIMD_WIDTH;
                    for l in 0..chunks {
                        let offset = l * SIMD_WIDTH;
                        let a_chunk = f32x8::from_slice(&a_row[offset..]);
                        let b_chunk = f32x8::from_slice(&b_t_row[offset..]);

                        // Fused Multiply-Add is faster than separate multiply and add
                        sum_vec = a_chunk.mul_add(b_chunk, sum_vec);
                    }

                    // Horizontally sum the partial sums in the SIMD vector
                    let mut dot_product = sum_vec.reduce_sum();

                    // Process any remaining elements that didn't fit in a SIMD chunk
                    let remainder_start = chunks * SIMD_WIDTH;
                    for l in remainder_start..k {
                        dot_product += a_row[l] * b_t_row[l];
                    }

                    result_row[j] = dot_product;
                }
            });

        Ok(Tensor::from_vec(result_data, &[m, n], a.device).unwrap())
    }
}

impl ShapeOps for CpuSimdPar {
    fn transpose(tensor: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::transpose");
        let axes: Vec<usize> = (0..tensor.rank()).rev().collect();
        CpuSimdPar::transpose_axes(tensor, &axes)
    }

    fn transpose_axes(tensor: &Tensor, axes: &[usize]) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::transpose_axes");
        let rank = tensor.rank();
        if axes.len() != rank {
            return Err(format!(
                "Axes length {} does not match tensor rank {}",
                axes.len(),
                rank
            ));
        }

        // Calculate new shape
        let new_dims: Vec<usize> = axes.iter().map(|&i| tensor.shape()[i]).collect();
        let new_shape = Shape::new(&new_dims);
        let new_len = new_shape.len();
        let mut new_data = vec![0.0f32; new_len]; // Use f32 for SIMD

        let old_strides = tensor.shape.strides();
        let new_strides = new_shape.strides();

        // Process in blocks for better cache locality
        let block_size = 64; // Cache line friendly block size

        // Use par_chunks_mut for safe, parallel, in-place mutation.
        new_data
            .par_chunks_mut(block_size)
            .enumerate() // Get the index of the block to calculate the global offset
            .for_each(|(block_idx, chunk)| {
                let start_idx = block_idx * block_size;

                // Process each chunk with SIMD
                // `base_i` is the index *within the chunk*
                for base_i in (0..chunk.len()).step_by(8) {
                    let remaining = (chunk.len() - base_i).min(8);

                    // The global index into the destination tensor
                    let global_base_idx = start_idx + base_i;

                    if remaining < 8 {
                        // Handle remaining elements that don't fit in SIMD chunk
                        for i in 0..remaining {
                            let idx = global_base_idx + i;
                            let mut old_indices = vec![0; rank];
                            let mut temp_index = idx;

                            // Calculate multi-dimensional indices
                            for (j, &stride) in new_strides.iter().enumerate() {
                                old_indices[axes[j]] = temp_index / stride;
                                temp_index %= stride;
                            }

                            // Calculate linear index in the original tensor
                            let old_i = old_indices
                                .iter()
                                .enumerate()
                                .fold(0, |acc, (j, &idx_val)| acc + idx_val * old_strides[j]);

                            chunk[base_i + i] = tensor.data[old_i];
                        }
                    } else {
                        // Process full SIMD chunk
                        let indices: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| {
                            let idx = global_base_idx + i;
                            let mut old_indices = vec![0; rank];
                            let mut temp_index = idx;

                            // Calculate multi-dimensional indices
                            for j in 0..rank - 1 {
                                old_indices[axes[j]] = temp_index / new_strides[j];
                                temp_index %= new_strides[j];
                            }
                            old_indices[axes[rank - 1]] = temp_index;

                            // Calculate linear index in the original tensor
                            old_indices
                                .iter()
                                .enumerate()
                                .fold(0, |acc, (j, &idx_val)| acc + idx_val * old_strides[j])
                        });

                        // Gather values using SIMD
                        let values = {
                            // Using gather is often faster but requires unsafe in stable Rust for now
                            // std::simd::f32x8::gather_or_default(&tensor.data, std::simd::Simd::from_array(indices))
                            // The following is a safe alternative if you can't use nightly/unsafe
                            std::simd::f32x8::from_array([
                                tensor.data[indices[0]],
                                tensor.data[indices[1]],
                                tensor.data[indices[2]],
                                tensor.data[indices[3]],
                                tensor.data[indices[4]],
                                tensor.data[indices[5]],
                                tensor.data[indices[6]],
                                tensor.data[indices[7]],
                            ])
                        };

                        // Store the gathered values directly into the mutable chunk
                        values.copy_to_slice(&mut chunk[base_i..base_i + 8]);
                    }
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

impl ReductionOps for CpuSimdPar {
    fn sum(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::sum");
        reduce_axis(tensor, axis, |a, b| a + b, 0.0)
    }

    fn mean(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::mean");
        let sum = Self::sum(tensor, axis)?;
        let count = match axis {
            None => tensor.numel() as f32,
            Some(axis) => tensor.shape()[axis] as f32,
        };
        let data = Arc::new(sum.data.par_iter().map(|&x| x / count).collect());
        Ok(Tensor {
            data,
            shape: sum.shape,
            device: sum.device,
            dtype: sum.dtype,
        })
    }

    fn max(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::max");
        reduce_axis(tensor, axis, |a, b| a.max(b), f32::NEG_INFINITY)
    }

    fn min(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::min");
        reduce_axis(tensor, axis, |a, b| a.min(b), f32::INFINITY)
    }

    fn argmax(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::argmax");
        arg_reduce_axis(tensor, axis, |a, b| a.1 > b.1)
    }

    fn argmin(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::argmin");
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
    trace_fn!("CpuSimdPar::reduce_axis");
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
            let after_dim_size: usize = tensor.shape()[axis + 1..].iter().product();

            result_data
                .par_chunks_mut(after_dim_size)
                .enumerate()
                .for_each(|(i, chunk)| {
                    for (k, chunk_val) in chunk.iter_mut().enumerate() {
                        *chunk_val = (0..inner_dim_size)
                            .map(|j| {
                                let idx = (i * inner_dim_size * after_dim_size)
                                    + (j * after_dim_size)
                                    + k;
                                tensor.data[idx]
                            })
                            .fold(init, &reduce_op);
                    }
                });
            Ok(Tensor::from_vec(result_data, &output_shape, tensor.device).unwrap())
        }
    }
}

impl ScalarOps for CpuSimdPar {
    fn add_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::add_scalar");
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let scalar_simd = f32x8::splat(scalar);

        chunks.par_chunks_mut(8).for_each(|chunk| {
            let simd_chunk = f32x8::from_slice(chunk);
            let result = simd_chunk + scalar_simd;
            result.copy_to_slice(chunk);
        });

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

    fn sub_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::sub_scalar");
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let scalar_simd = f32x8::splat(scalar);

        chunks.par_chunks_mut(8).for_each(|chunk| {
            let simd_chunk = f32x8::from_slice(chunk);
            let result = simd_chunk - scalar_simd;
            result.copy_to_slice(chunk);
        });

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

    fn mul_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::mul_scalar");
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let scalar_simd = f32x8::splat(scalar);

        chunks.par_chunks_mut(8).for_each(|chunk| {
            let simd_chunk = f32x8::from_slice(chunk);
            let result = simd_chunk * scalar_simd;
            result.copy_to_slice(chunk);
        });

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

    fn div_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::div_scalar");
        if scalar == 0.0 {
            return Err("Division by zero".to_string());
        }
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let scalar_simd = f32x8::splat(scalar);

        chunks.par_chunks_mut(8).for_each(|chunk| {
            let simd_chunk = f32x8::from_slice(chunk);
            let result = simd_chunk / scalar_simd;
            result.copy_to_slice(chunk);
        });

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

    fn r_add_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::r_add_scalar");
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let scalar_simd = f32x8::splat(scalar);

        chunks.par_chunks_mut(8).for_each(|chunk| {
            let simd_chunk = f32x8::from_slice(chunk);
            let result = scalar_simd + simd_chunk;
            result.copy_to_slice(chunk);
        });

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

    fn r_sub_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::r_sub_scalar");
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let scalar_simd = f32x8::splat(scalar);

        chunks.par_chunks_mut(8).for_each(|chunk| {
            let simd_chunk = f32x8::from_slice(chunk);
            let result = scalar_simd - simd_chunk;
            result.copy_to_slice(chunk);
        });

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

    fn r_mul_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::r_mul_scalar");
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let scalar_simd = f32x8::splat(scalar);

        chunks.par_chunks_mut(8).for_each(|chunk| {
            let simd_chunk = f32x8::from_slice(chunk);
            let result = scalar_simd * simd_chunk;
            result.copy_to_slice(chunk);
        });

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

    fn r_div_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::r_div_scalar");
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let scalar_simd = f32x8::splat(scalar);

        chunks.par_chunks_mut(8).for_each(|chunk| {
            let simd_chunk = f32x8::from_slice(chunk);
            let result = scalar_simd / simd_chunk;
            result.copy_to_slice(chunk);
        });

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

impl CreationOps for CpuSimdPar {
    fn random(shape: &[usize], device: crate::tensor::Device) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::random");
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

    fn arange(start: f32, end: f32, device: crate::tensor::Device) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::arange");
        let size = (end - start).abs().ceil() as usize;
        let step = if end >= start { 1.0 } else { -1.0 };

        let data: Vec<f32> = (0..size)
            .into_par_iter()
            .map(|i| start + (i as f32) * step)
            .collect();

        Self::from_vec(data, &[size], device)
    }

    fn zeros(shape: &[usize], device: crate::tensor::Device) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::zeros");
        let size: usize = shape.iter().product();
        let data = vec![0.0; size];
        Self::from_vec(data, shape, device)
    }

    fn ones(shape: &[usize], device: crate::tensor::Device) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::ones");
        let size: usize = shape.iter().product();
        let data = vec![1.0; size];
        Self::from_vec(data, shape, device)
    }

    fn identity(size: usize, device: crate::tensor::Device) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::identity");
        let len = size * size;
        let mut data = vec![0.0; len];

        // Parallelize setting the diagonal elements
        data.par_chunks_mut(size).enumerate().for_each(|(i, row)| {
            row[i] = 1.0;
        });

        Self::from_vec(data, &[size, size], device)
    }

    fn from_vec(
        data: Vec<f32>,
        shape: &[usize],
        device: crate::tensor::Device,
    ) -> Result<Tensor, String> {
        trace_fn!("CpuSimdPar::from_vec");
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
        trace_fn!("CpuSimdPar::from_slice");
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

fn arg_reduce_axis<F>(tensor: &Tensor, axis: Option<usize>, compare: F) -> Result<Tensor, String>
where
    F: Fn((usize, f32), (usize, f32)) -> bool + Send + Sync,
{
    trace_fn!("CpuSimdPar::arg_reduce_axis");
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
            let after_dim_size: usize = tensor.shape()[axis + 1..].iter().product();

            result_data
                .par_chunks_mut(after_dim_size)
                .enumerate()
                .for_each(|(i, chunk)| {
                    for (k, chunk_val) in chunk.iter_mut().enumerate() {
                        let (best_idx, _) = (0..inner_dim_size)
                            .map(|j| {
                                let idx = (i * inner_dim_size * after_dim_size)
                                    + (j * after_dim_size)
                                    + k;
                                (j, tensor.data[idx])
                            })
                            .reduce(|a, b| if a.1.is_nan() || compare(b, a) { b } else { a })
                            .unwrap();
                        *chunk_val = best_idx as f32;
                    }
                });
            Ok(Tensor::from_vec(result_data, &output_shape, tensor.device).unwrap())
        }
    }
}
