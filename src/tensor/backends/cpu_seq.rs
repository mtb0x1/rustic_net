//! Sequential CPU backend implementation
//!
//! This module provides a sequential implementation of all tensor operations
//! that runs on the CPU without any parallelization.

use super::traits::*;
use super::{Device, DType, Shape, Tensor};
use crate::trace_fn;
use rand::Rng;
use std::sync::Arc;
use tracing::debug;

/// Sequential CPU backend
pub struct CpuSequential;

impl CreateTensor for CpuSequential {
    fn create_tensor(data: Vec<f32>, shape: Shape, device: Device, dtype: DType) -> Tensor {
        trace_fn!("CpuSequential::create_tensor");
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

impl UnaryOps for CpuSequential {
    fn relu(tensor: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::relu");
        let data = tensor
            .data
            .iter()
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

impl BinaryElementwiseOps for CpuSequential {
    fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::add");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise addition".to_string());
        }

        let data = a
            .data
            .iter()
            .zip(b.data.iter())
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
        trace_fn!("CpuSequential::sub");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise subtraction".to_string());
        }

        let data = a
            .data
            .iter()
            .zip(b.data.iter())
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
        trace_fn!("CpuSequential::mul");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise multiplication".to_string());
        }

        let data = a
            .data
            .iter()
            .zip(b.data.iter())
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
        trace_fn!("CpuSequential::div");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise division".to_string());
        }

        let data = a
            .data
            .iter()
            .zip(b.data.iter())
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

impl MatOps for CpuSequential {
    fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::matmul");
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

        Ok(Tensor {
            data: Arc::new(result_data),
            shape: Shape::new(&[m, n]),
            device: a.device,
            dtype: a.dtype,
        })
    }
}

impl ReductionOps for CpuSequential {
    fn sum(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::sum");
        match axis {
            None => {
                // Sum all elements
                let sum = tensor.data.iter().sum();
                Ok(Tensor::from_vec(vec![sum], &[1], tensor.device).unwrap())
            }
            Some(axis) => {
                // Sum along the specified axis
                if axis >= tensor.shape().len() {
                    return Err(format!("Axis {} out of bounds for tensor of rank {}", 
                                     axis, tensor.shape().len()));
                }

                // Calculate output shape (remove the dimension we're summing over)
                let mut output_shape = tensor.shape().dims().to_vec();
                output_shape.remove(axis);
                if output_shape.is_empty() {
                    output_shape.push(1); // Ensure at least 1D output
                }

                // Calculate the total number of elements in the result
                let output_size: usize = output_shape.iter().product();
                let mut result_data = vec![0.0; output_size];

                // Calculate the number of elements to sum for each output element
                let elements_per_sum = tensor.shape().dims()[axis];
                let elements_before = tensor.shape().dims()[..axis].iter().product::<usize>();
                let elements_after = tensor.shape().dims()[axis + 1..].iter().product::<usize>();

                // Perform the sum
                for i in 0..elements_before {
                    for k in 0..elements_after {
                        let mut sum = 0.0;
                        for j in 0..elements_per_sum {
                            let idx = (i * elements_per_sum + j) * elements_after + k;
                            sum += tensor.data[idx];
                        }
                        let out_idx = i * elements_after + k;
                        result_data[out_idx] = sum;
                    }
                }

                Ok(Tensor::from_vec(result_data, &output_shape, tensor.device).unwrap())
            }
        }
    }

    fn mean(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::mean");
        let sum = Self::sum(tensor, axis)?;
        let count = match axis {
            None => tensor.data.len() as f32,
            Some(axis) => tensor.shape().dims()[axis] as f32,
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
        trace_fn!("CpuSequential::max");
        self::reduce_axis(tensor, axis, |a, b| a.max(*b), f32::NEG_INFINITY)
    }

    fn min(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::min");
        self::reduce_axis(tensor, axis, |a, b| a.min(*b), f32::INFINITY)
    }

    fn argmax(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::argmax");
        self::arg_reduce_axis(tensor, axis, |a, b| a.1 < b.1)
    }

    fn argmin(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::argmin");
        self::arg_reduce_axis(tensor, axis, |a, b| a.1 > b.1)
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
    F: Fn(f32, &f32) -> f32,
{
    match axis {
        None => {
            // Global reduction
            let result = tensor.data.iter().fold(init, |acc, x| reduce_op(acc, x));
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
            let mut result_data = vec![init; output_size];

            // Calculate the number of elements to reduce for each output element
            let elements_per_reduce = tensor.shape().dims()[axis];
            let elements_before = tensor.shape().dims()[..axis].iter().product::<usize>();
            let elements_after = tensor.shape().dims()[axis + 1..].iter().product::<usize>();

            // Perform the reduction
            for i in 0..elements_before {
                for k in 0..elements_after {
                    let mut acc = init;
                    for j in 0..elements_per_reduce {
                        let idx = (i * elements_per_reduce + j) * elements_after + k;
                        acc = reduce_op(acc, &tensor.data[idx]);
                    }
                    let out_idx = i * elements_after + k;
                    result_data[out_idx] = acc;
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
    match axis {
        None => {
            // Global reduction
            let (idx, _) = tensor
                .data
                .iter()
                .enumerate()
                .fold((0, f32::NAN), |acc, (i, &x)| {
                    if i == 0 || compare(acc, (i, x)) {
                        (i, x)
                    } else {
                        acc
                    }
                });
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
            let mut result_data = vec![0.0; output_size];

            // Calculate the number of elements to reduce for each output element
            let elements_per_reduce = tensor.shape().dims()[axis];
            let elements_before = tensor.shape().dims()[..axis].iter().product::<usize>();
            let elements_after = tensor.shape().dims()[axis + 1..].iter().product::<usize>();

            // Perform the reduction
            for i in 0..elements_before {
                for k in 0..elements_after {
                    let mut best_idx = 0;
                    let mut best_val = f32::NAN;

                    for j in 0..elements_per_reduce {
                        let idx = (i * elements_per_reduce + j) * elements_after + k;
                        let val = tensor.data[idx];

                        if j == 0 || compare((best_idx, best_val), (j, val)) {
                            best_idx = j;
                            best_val = val;
                        }
                    }

                    let out_idx = i * elements_after + k;
                    result_data[out_idx] = best_idx as f32;
                }
            }

            Ok(Tensor::from_vec(result_data, &output_shape, tensor.device).unwrap())
        }
    }
}
