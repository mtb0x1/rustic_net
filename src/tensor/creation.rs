//! # Tensor Creation
//!
//! Factory functions for creating tensors with various initializations and data sources.
//! Provides both simple constructors and more advanced factory methods for tensor creation.

use super::{DType, Device, Shape, Tensor};
use crate::trace_fn;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Creates a tensor from a vector with the specified shape and device.
///
/// # Arguments
/// * `data` - Vector of elements to populate the tensor
/// * `shape` - Desired shape of the tensor
/// * `device` - Target compute device
///
/// # Errors
/// Returns an error if the data length doesn't match the specified shape.
///
/// # Example
/// ```
/// # use rustic_net::tensor::creation::from_vec;
/// # use rustic_net::tensor::Device;
/// let data = vec![1.0, 2.0, 3.0, 4.0];
/// let tensor = from_vec(data, &[2, 2], Device::default()).unwrap();
/// assert_eq!(tensor.shape(), &[2, 2]);
/// ```
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

/// Creates a tensor from a slice with the specified shape.
///
/// This is a convenience wrapper around `from_vec` that avoids explicit vector allocation.
///
/// # Arguments
/// * `slice` - Slice of elements to populate the tensor
/// * `shape` - Desired shape of the tensor
/// * `device` - Target compute device
///
/// # Example
/// ```
/// # use rustic_net::tensor::creation::from_slice;
/// # use rustic_net::tensor::Device;
/// let data = [1.0, 2.0, 3.0, 4.0];
/// let tensor = from_slice(&data, &[2, 2], Device::default()).unwrap();
/// ```
pub fn from_slice(slice: &[f32], shape: &[usize], device: Device) -> Result<Tensor, String> {
    trace_fn!("tensor::creation::from_slice");
    from_vec(slice.to_vec(), shape, device)
}

/// Creates a tensor filled with zeros.
///
/// # Arguments
/// * `shape` - Desired shape of the tensor
/// * `device` - Target compute device
///
/// # Example
/// ```
/// # use rustic_net::tensor::creation::zeros;
/// # use rustic_net::tensor::Device;
/// let tensor = zeros(&[2, 3], Device::default());
/// assert_eq!(tensor.to_vec(), vec![0.0; 6]);
/// ```
pub fn zeros(shape: &[usize], device: Device) -> Tensor {
    trace_fn!("tensor::creation::zeros");
    let size: usize = shape.iter().product();
    let data = vec![0.0; size];
    from_vec(data, shape, device).unwrap()
}

/// Creates a tensor filled with ones.
///
/// # Arguments
/// * `shape` - Desired shape of the tensor
/// * `device` - Target compute device
///
/// # Example
/// ```
/// # use rustic_net::tensor::creation::ones;
/// # use rustic_net::tensor::Device;
/// let tensor = ones(&[2, 2], Device::default());
/// assert_eq!(tensor.to_vec(), vec![1.0; 4]);
/// ```
pub fn ones(shape: &[usize], device: Device) -> Tensor {
    trace_fn!("tensor::creation::ones");
    let size: usize = shape.iter().product();
    let data = vec![1.0; size];
    from_vec(data, shape, device).unwrap()
}

/// Creates an identity matrix (2D tensor with ones on the diagonal).
///
/// # Arguments
/// * `size` - Size of the square matrix
/// * `device` - Target compute device
///
/// # Example
/// ```
/// # use rustic_net::tensor::creation::identity;
/// # use rustic_net::tensor::Device;
/// let eye = identity(2, Device::default());
/// assert_eq!(eye.to_vec(), vec![1.0, 0.0, 0.0, 1.0]);
/// ```
pub fn identity(size: usize, device: Device) -> Tensor {
    trace_fn!("tensor::creation::identity");
    let mut data = vec![0.0; size * size];
    for i in 0..size {
        data[i * size + i] = 1.0;
    }
    from_vec(data, &[size, size], device).unwrap()
}

/// Creates a tensor with uniformly distributed random values in [0, 1).
///
/// Uses ChaCha8Rng for parallel generation when the `parallel` feature is enabled.
///
/// # Arguments
/// * `shape` - Desired shape of the tensor
/// * `device` - Target compute device
///
/// # Example
/// ```
/// # use rustic_net::tensor::creation::random;
/// # use rustic_net::tensor::Device;
/// let tensor = random(&[2, 2], Device::default());
/// // Values will be between 0.0 (inclusive) and 1.0 (exclusive)
/// ```
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

/// Creates a 1D tensor with values in the range [start, end).
///
/// The step size is always 1.0. For non-integer steps, use `linspace`.
///
/// # Arguments
/// * `start` - First value (inclusive)
/// * `end` - End value (exclusive)
/// * `device` - Target compute device
///
/// # Example
/// ```
/// # use rustic_net::tensor::creation::arange;
/// # use rustic_net::tensor::Device;
/// let tensor = arange(2.0, 5.0, Device::default());
/// assert_eq!(tensor.to_vec(), vec![2.0, 3.0, 4.0]);
/// ```
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
