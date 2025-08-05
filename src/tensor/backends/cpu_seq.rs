//! # Sequential CPU Backend
//!
//! A single-threaded CPU implementation of all tensor operations.
//! This backend is used when parallel execution is not available or desired.
//!
//! ## Features
//! - Pure Rust implementation with no external dependencies
//! - Deterministic execution
//! - Lower memory overhead than parallel version
//! - Better for small tensors due to lack of threading overhead

use super::traits::*;
use crate::tensor::{Shape, Tensor};
use crate::trace_fn;
use std::sync::Arc;

/// Marker type for the sequential CPU backend.
///
/// Implements all tensor operation traits using a single-threaded approach.
/// This is the fallback backend when parallel execution is not available.
pub struct CpuSequential;

impl UnaryOps for CpuSequential {
    fn relu(tensor: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::relu");
        let data: Vec<f32> = tensor
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
                    f32::NAN // Using NAN for division by zero is more standard than INFINITY
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

impl ReductionOps for CpuSequential {
    fn sum(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::sum");
        reduce_axis(tensor, axis, |a, b| a + b, 0.0)
    }

    fn mean(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::mean");
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
        trace_fn!("CpuSequential::max");
        reduce_axis(tensor, axis, |a, b| a.max(b), f32::NEG_INFINITY)
    }

    fn min(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::min");
        reduce_axis(tensor, axis, |a, b| a.min(b), f32::INFINITY)
    }

    fn argmax(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::argmax");
        arg_reduce_axis(tensor, axis, |a, b| a.1 > b.1)
    }

    fn argmin(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::argmin");
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
    trace_fn!("CpuSequential::reduce_axis");
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

impl ScalarOps for CpuSequential {
    fn add_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::add_scalar");
        let data: Vec<f32> = tensor.data.iter().map(|&x| x + scalar).collect();
        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    fn sub_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::sub_scalar");
        let data: Vec<f32> = tensor.data.iter().map(|&x| x - scalar).collect();
        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    fn mul_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::mul_scalar");
        let data: Vec<f32> = tensor.data.iter().map(|&x| x * scalar).collect();
        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    fn div_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::div_scalar");
        if scalar == 0.0 {
            return Err("Division by zero".to_string());
        }
        let data: Vec<f32> = tensor.data.iter().map(|&x| x / scalar).collect();
        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    fn r_add_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::r_add_scalar");
        let data: Vec<f32> = tensor.data.iter().map(|&x| scalar + x).collect();
        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    fn r_sub_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::r_sub_scalar");
        let data: Vec<f32> = tensor.data.iter().map(|&x| scalar - x).collect();
        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    fn r_mul_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::r_mul_scalar");
        let data: Vec<f32> = tensor.data.iter().map(|&x| scalar * x).collect();
        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }

    fn r_div_scalar(tensor: &Tensor, scalar: f32) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::r_div_scalar");
        let data: Vec<f32> = tensor
            .data
            .iter()
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

impl ShapeOps for CpuSequential {
    fn transpose(tensor: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::transpose");
        let axes: Vec<usize> = (0..tensor.rank()).rev().collect();
        CpuSequential::transpose_axes(tensor, &axes)
    }

    fn transpose_axes(tensor: &Tensor, axes: &[usize]) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::transpose_axes");
        let rank = tensor.rank();
        if axes.len() != rank {
            return Err(format!(
                "Axes length {} does not match tensor rank {}",
                axes.len(),
                rank
            ));
        }

        let new_dims: Vec<usize> = axes.iter().map(|&i| tensor.shape()[i]).collect();
        let new_shape = Shape::new(&new_dims);
        let mut new_data = vec![0.0; tensor.numel()];

        let old_strides = tensor.shape.strides();
        let new_strides = new_shape.strides();

        for (i, &val) in tensor.data.iter().enumerate() {
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
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }
}

impl CreationOps for CpuSequential {
    fn random(shape: &[usize], device: crate::tensor::Device) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::random");
        use rand::Rng;
        let size: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..size).map(|_| rng.gen_range(0.0..1.0)).collect();
        Self::from_vec(data, shape, device)
    }

    fn arange(start: f32, end: f32, device: crate::tensor::Device) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::arange");
        let size = (end - start).abs() as usize;
        let data: Vec<f32> = (0..size).map(|i| start + i as f32).collect();
        Self::from_vec(data, &[size], device)
    }

    fn zeros(shape: &[usize], device: crate::tensor::Device) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::zeros");
        let size: usize = shape.iter().product();
        let data = vec![0.0; size];
        Self::from_vec(data, shape, device)
    }

    fn ones(shape: &[usize], device: crate::tensor::Device) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::ones");
        let size: usize = shape.iter().product();
        let data = vec![1.0; size];
        Self::from_vec(data, shape, device)
    }

    fn identity(size: usize, device: crate::tensor::Device) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::identity");
        let mut data = vec![0.0; size * size];
        for i in 0..size {
            data[i * size + i] = 1.0;
        }
        Self::from_vec(data, &[size, size], device)
    }

    fn from_vec(
        data: Vec<f32>,
        shape: &[usize],
        device: crate::tensor::Device,
    ) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::from_vec");
        let size: usize = shape.iter().product();
        if data.len() != size {
            return Err(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                size
            ));
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: Shape::new(shape),
            device,
            dtype: crate::tensor::DType::F32,
        })
    }

    fn from_slice(
        slice: &[f32],
        shape: &[usize],
        device: crate::tensor::Device,
    ) -> Result<Tensor, String> {
        trace_fn!("CpuSequential::from_slice");
        let size: usize = shape.iter().product();
        if slice.len() != size {
            return Err(format!(
                "Slice length {} does not match shape {:?} (expected {})",
                slice.len(),
                shape,
                size
            ));
        }

        Ok(Tensor {
            data: Arc::new(slice.to_vec()),
            shape: Shape::new(shape),
            device,
            dtype: crate::tensor::DType::F32,
        })
    }
}

// Helper function for arg reduction operations
fn arg_reduce_axis<F>(tensor: &Tensor, axis: Option<usize>, compare: F) -> Result<Tensor, String>
where
    F: Fn((usize, f32), (usize, f32)) -> bool,
{
    trace_fn!("CpuSequential::arg_reduce_axis");
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
