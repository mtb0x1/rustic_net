//! Parallel CPU backend implementation
//!
//! This module provides a parallel implementation of all tensor operations
//! that runs on the CPU using Rayon for parallelization.

#![cfg(feature = "parallel")]

use super::traits::*;
use super::{Device, DType, Shape, Tensor};
use crate::parallel;
use crate::trace_fn;
use rayon::prelude::*;
use std::sync::Arc;
use tracing::debug;

/// Parallel CPU backend
pub struct CpuParallel;

impl CreateTensor for CpuParallel {
    fn create_tensor(data: Vec<f32>, shape: Shape, device: Device, dtype: DType) -> Tensor {
        trace_fn!("CpuParallel::create_tensor");
        debug!(
            "Creating tensor with shape {:?}, device {:?}, dtype {:?}",
            shape.dims(),
            device,
            dtype
        );

        // For now, we only support f32 tensors
        assert_eq!(dtype, DType::F32);

        Tensor {
            data: Arc::new(data),
            shape,
            device,
            dtype,
        }
    }
}

impl UnaryOps for CpuParallel {
    fn relu(tensor: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::relu");
        let data = tensor
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
            .map(|(&a, &b)| {
                if b == 0.0 {
                    f32::INFINITY
                } else {
                    a / b
                }
            })
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
    fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::matmul");
        // Check if both tensors are 2D
        if a.shape().len() != 2 || b.shape().len() != 2 {
            return Err("Matrix multiplication requires 2D tensors".to_string());
        }

        // Check if the inner dimensions match
        if a.shape()[1] != b.shape()[0] {
            return Err("Inner dimensions must match for matrix multiplication".to_string());
        }

        let m = a.shape()[0];
        let n = b.shape()[1];
        let k = a.shape()[1];

        // Create a mutable vector for the result
        let result_data = vec![0.0; m * n];
        let result_data = Arc::new(std::sync::Mutex::new(result_data));

        // Parallelize the outer loop over rows
        (0..m).into_par_iter().for_each(|i| {
            let mut row_result = vec![0.0; n];
            
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a.data[i * k + l] * b.data[l * n + j];
                }
                row_result[j] = sum;
            }
            
            // Lock the result data and update the row
            let mut data = result_data.lock().unwrap();
            for j in 0..n {
                data[i * n + j] = row_result[j];
            }
        });

        // Extract the result from the mutex
        let result_data = Arc::try_unwrap(result_data)
            .unwrap()
            .into_inner()
            .unwrap();

        Ok(Tensor {
            data: Arc::new(result_data),
            shape: Shape::new(&[m, n]),
            device: a.device,
            dtype: a.dtype,
        })
    }
}

impl ReductionOps for CpuParallel {
    fn sum(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::sum");
        match axis {
            None => {
                // Parallel sum of all elements
                let sum = tensor.data.par_iter().sum();
                Ok(Tensor::from_vec(vec![sum], &[1], tensor.device).unwrap())
            }
            Some(axis) => {
                // Sum along the specified axis
                if axis >= tensor.shape().len() {
                    return Err(format!(
                        "Axis {} out of bounds for tensor of rank {}",
                        axis,
                        tensor.shape().len()
                    ));
                }

                // Calculate output shape (remove the dimension we're summing over)
                let mut output_shape = tensor.shape().dims().to_vec();
                output_shape.remove(axis);
                if output_shape.is_empty() {
                    output_shape.push(1); // Ensure at least 1D output
                }

                // Calculate the total number of elements in the result
                let output_size: usize = output_shape.iter().product();
                let result_data = vec![0.0; output_size];
                let result_data = Arc::new(std::sync::Mutex::new(result_data));

                // Calculate the number of elements to sum for each output element
                let elements_per_sum = tensor.shape().dims()[axis];
                let elements_before = tensor.shape().dims()[..axis].iter().product::<usize>();
                let elements_after = tensor.shape().dims()[axis + 1..].iter().product::<usize>();

                // Parallelize the reduction
                (0..elements_before).into_par_iter().for_each(|i| {
                    let mut local_sums = vec![0.0; elements_after];
                    
                    for j in 0..elements_per_sum {
                        for k in 0..elements_after {
                            let idx = (i * elements_per_sum + j) * elements_after + k;
                            local_sums[k] += tensor.data[idx];
                        }
                    }
                    
                    // Update the result in a thread-safe manner
                    let mut data = result_data.lock().unwrap();
                    for k in 0..elements_after {
                        let out_idx = i * elements_after + k;
                        data[out_idx] = local_sums[k];
                    }
                });

                // Extract the result from the mutex
                let result_data = Arc::try_unwrap(result_data)
                    .unwrap()
                    .into_inner()
                    .unwrap();

                Ok(Tensor::from_vec(result_data, &output_shape, tensor.device).unwrap())
            }
        }
    }

    fn mean(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::mean");
        let sum = Self::sum(tensor, axis)?;
        let count = match axis {
            None => tensor.data.len() as f32,
            Some(axis) => tensor.shape().dims()[axis] as f32,
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
        self::reduce_axis(tensor, axis, |a, b| a.max(*b), f32::NEG_INFINITY)
    }

    fn min(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::min");
        self::reduce_axis(tensor, axis, |a, b| a.min(*b), f32::INFINITY)
    }

    fn argmax(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::argmax");
        self::arg_reduce_axis(tensor, axis, |a, b| a.1 < b.1)
    }

    fn argmin(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuParallel::argmin");
        self::arg_reduce_axis(tensor, axis, |a, b| a.1 > b.1)
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
    F: Fn(f32, &f32) -> f32 + Send + Sync,
{
    match axis {
        None => {
            // Global reduction
            let result = tensor
                .data
                .par_iter()
                .fold(|| init, |acc, x| reduce_op(acc, x))
                .reduce(|| init, |a, b| reduce_op(a, &b));
            
            Ok(Tensor::from_vec(vec![result], &[1], tensor.device).unwrap())
        }
        Some(axis) => {
            if axis >= tensor.shape().len() {
                return Err(format!(
                    "Axis {} out of bounds for tensor of rank {}",
                    axis,
                    tensor.shape().len()
                ));
            }

            // Calculate output shape (remove the dimension we're reducing over)
            let mut output_shape = tensor.shape().dims().to_vec();
            output_shape.remove(axis);
            if output_shape.is_empty() {
                output_shape.push(1); // Ensure at least 1D output
            }

            // Calculate the total number of elements in the result
            let output_size: usize = output_shape.iter().product();
            let result_data = vec![init; output_size];
            let result_data = Arc::new(std::sync::Mutex::new(result_data));

            // Calculate the number of elements to reduce for each output element
            let elements_per_reduce = tensor.shape().dims()[axis];
            let elements_before = tensor.shape().dims()[..axis].iter().product::<usize>();
            let elements_after = tensor.shape().dims()[axis + 1..].iter().product::<usize>();

            // Parallelize the reduction
            (0..elements_before).into_par_iter().for_each(|i| {
                let mut local_results = vec![init; elements_after];
                
                for j in 0..elements_per_reduce {
                    for k in 0..elements_after {
                        let idx = (i * elements_per_reduce + j) * elements_after + k;
                        local_results[k] = reduce_op(local_results[k], &tensor.data[idx]);
                    }
                }
                
                // Update the result in a thread-safe manner
                let mut data = result_data.lock().unwrap();
                for k in 0..elements_after {
                    let out_idx = i * elements_after + k;
                    data[out_idx] = local_results[k];
                }
            });

            // Extract the result from the mutex
            let result_data = Arc::try_unwrap(result_data)
                .unwrap()
                .into_inner()
                .unwrap();

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
            // Global reduction
            let (idx, _) = tensor
                .data
                .par_iter()
                .enumerate()
                .fold(
                    || (0, f32::NAN),
                    |acc, (i, &x)| {
                        if i == 0 || compare(acc, (i, x)) {
                            (i, x)
                        } else {
                            acc
                        }
                    },
                )
                .reduce(
                    || (0, f32::NAN),
                    |a, b| if compare(a, b) { b } else { a },
                );
            
            Ok(Tensor::from_vec(vec![idx as f32], &[1], tensor.device).unwrap())
        }
        Some(axis) => {
            if axis >= tensor.shape().len() {
                return Err(format!(
                    "Axis {} out of bounds for tensor of rank {}",
                    axis,
                    tensor.shape().len()
                ));
            }

            // Calculate output shape (remove the dimension we're reducing over)
            let mut output_shape = tensor.shape().dims().to_vec();
            output_shape.remove(axis);
            if output_shape.is_empty() {
                output_shape.push(1); // Ensure at least 1D output
            }

            // Calculate the total number of elements in the result
            let output_size: usize = output_shape.iter().product();
            let result_data = vec![0.0; output_size];
            let result_data = Arc::new(std::sync::Mutex::new(result_data));

            // Calculate the number of elements to reduce for each output element
            let elements_per_reduce = tensor.shape().dims()[axis];
            let elements_before = tensor.shape().dims()[..axis].iter().product::<usize>();
            let elements_after = tensor.shape().dims()[axis + 1..].iter().product::<usize>();

            // Parallelize the reduction
            (0..elements_before).into_par_iter().for_each(|i| {
                let mut local_best = vec![(0, f32::NAN); elements_after];
                
                for j in 0..elements_per_reduce {
                    for k in 0..elements_after {
                        let idx = (i * elements_per_reduce + j) * elements_after + k;
                        let val = tensor.data[idx];
                        
                        if j == 0 || compare(local_best[k], (j, val)) {
                            local_best[k] = (j, val);
                        }
                    }
                }
                
                // Update the result in a thread-safe manner
                let mut data = result_data.lock().unwrap();
                for k in 0..elements_after {
                    let out_idx = i * elements_after + k;
                    data[out_idx] = local_best[k].0 as f32;
                }
            });

            // Extract the result from the mutex
            let result_data = Arc::try_unwrap(result_data)
                .unwrap()
                .into_inner()
                .unwrap();

            Ok(Tensor::from_vec(result_data, &output_shape, tensor.device).unwrap())
        }
    }
}
