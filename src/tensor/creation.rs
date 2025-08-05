//! Tensor creation functions
//!
//! This module contains functions for creating tensors with various initializations.

use super::{DType, Device, Shape, Tensor};
use crate::trace_fn;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Creates a new tensor from a vector with the given shape and device
pub fn from_vec<T: Into<Vec<f32>>>(
    data: T,
    shape: &[usize],
    device: Device,
) -> Result<Tensor, String> {
    trace_fn!("tensor::creation::from_vec");
    let data = data.into();
    let shape_obj = Shape::new(shape);

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
        data: Arc::new(data),
        shape: shape_obj,
        device,
        dtype: DType::F32, // For now, we only support f32 tensors
    })
}

/// Creates a new tensor from a slice with the given shape
pub fn from_slice(slice: &[f32], shape: &[usize], device: Device) -> Result<Tensor, String> {
    trace_fn!("tensor::creation::from_slice");
    from_vec(slice.to_vec(), shape, device)
}

/// Creates a new tensor filled with zeros
pub fn zeros(shape: &[usize], device: Device) -> Tensor {
    trace_fn!("tensor::creation::zeros");
    let size: usize = shape.iter().product();
    let data = vec![0.0; size];
    from_vec(data, shape, device).unwrap()
}

/// Creates a new tensor filled with ones
pub fn ones(shape: &[usize], device: Device) -> Tensor {
    trace_fn!("tensor::creation::ones");
    let size: usize = shape.iter().product();
    let data = vec![1.0; size];
    from_vec(data, shape, device).unwrap()
}

/// Creates an identity matrix of the given size
pub fn identity(size: usize, device: Device) -> Tensor {
    trace_fn!("tensor::creation::identity");
    let mut data = vec![0.0; size * size];
    for i in 0..size {
        data[i * size + i] = 1.0;
    }
    from_vec(data, &[size, size], device).unwrap()
}

/// Creates a new tensor with random values between 0.0 and 1.0
pub fn random(shape: &[usize], device: Device) -> Tensor {
    trace_fn!("tensor::creation::random");
    let size: usize = shape.iter().product();

    #[cfg(feature = "parallel")]
    let data: Vec<f32> = {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        (0..size)
            .into_par_iter()
            .map_init(
                || ChaCha8Rng::from_seed(rand::thread_rng().gen()),
                |rng, _| rng.gen_range(0.0..1.0),
            )
            .collect()
    };

    #[cfg(not(feature = "parallel"))]
    let data: Vec<f32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..size).map(|_| rng.gen_range(0.0..1.0)).collect()
    };

    from_vec(data, shape, device).unwrap()
}

/// Creates a new 1D tensor with values from start to end (exclusive)
pub fn arange(start: f32, end: f32, device: Device) -> Tensor {
    trace_fn!("tensor::creation::arange");
    let size = (end - start).abs() as usize;

    #[cfg(feature = "parallel")]
    let data: Vec<f32> = (0..size)
        .into_par_iter()
        .map(|i| start + i as f32)
        .collect();

    #[cfg(not(feature = "parallel"))]
    let data: Vec<f32> = (0..size).map(|i| start + i as f32).collect();

    from_vec(data, &[size], device).unwrap()
}
