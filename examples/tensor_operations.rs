//! # Tensor Operations Example with Parallel Processing
//!
//! This example demonstrates various tensor operations available in the Rustic Net library,
//! including parallel processing capabilities.

#[cfg(feature = "parallel")]
use rustic_net::init_thread_pool;
use rustic_net::tensor::{Device, Tensor};
use rustic_net::RusticNetInitTracingWith;
use tracing::{debug, warn};

#[inline]
pub fn infer_shape_squareish(size: usize) -> Vec<usize> {
    let mut x = (size as f64).sqrt().floor() as usize;
    while x > 0 {
        if size % x == 0 {
            return vec![x, size / x];
        }
        x -= 1;
    }
    vec![size, 1] // Fallback for prime numbers
}
#[inline]
pub fn infer_shape_aspect(size: usize, target_ratio: f64) -> Vec<usize> {
    let mut best_diff = f64::MAX;
    let mut best_opt = (1, size);

    let sqrt = (size as f64).sqrt().ceil() as usize;
    for x in 1..=sqrt {
        if size % x == 0 {
            let y = size / x;
            let ratio = x as f64 / y as f64;
            let diff = (ratio - target_ratio).abs();
            if diff < best_diff {
                best_opt = (x, y);
                best_diff = diff;
            }
        }
    }

    vec![best_opt.0, best_opt.1]
}
fn main() {
    // Setup tracing yourself
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter("rustic_net=trace,tensor_operations=trace")
        .with_target(true)
        .finish();

    RusticNetInitTracingWith(subscriber);

    //wrap all example code under a span
    let _span = tracing::span!(tracing::Level::TRACE, "tensor_operations").entered();

    #[cfg(feature = "parallel")]
    init_thread_pool();

    // Set up the device (CPU with parallel processing)
    let device = Device::default();

    // Create a large tensor to demonstrate parallel operations
    let size = std::env::var("RUSTIC_NET_EXAMPLE_TENSOR_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000); // Default to 1 million elements if env var not set

    warn!(
        "====> Tensor size: {}, requested size(RUSTIC_NET_EXAMPLE_TENSOR_SIZE): {:?} <====",
        size,
        std::env::var("RUSTIC_NET_EXAMPLE_TENSOR_SIZE").unwrap_or("".to_string())
    );

    let large_data: Vec<f32> = (0..size).map(|x| x as f32).collect();
    //set dimensions from size, shape is &[x,y]    where x*y = size
    let dims = infer_shape_squareish(size);
    let large_tensor = Tensor::from_vec(large_data, dims.as_ref(), device.clone()).unwrap();

    debug!("\n=== Tensor Creation ===");

    // Create a tensor from a vector
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], device.clone()).unwrap();
    debug!(
        "\nTensor from vector: shape {:?}, data: {:?}",
        a.shape(),
        a.to_vec()
    );

    // Create a tensor of ones
    let ones = Tensor::ones(&[2, 2], device.clone());
    debug!("\nTensor of ones: {:?}", ones.to_vec());

    // Create a tensor of zeros
    let zeros = Tensor::zeros(&[3, 3], device.clone());
    debug!("\nTensor of zeros: {:?}", zeros.to_vec());

    // Create an identity matrix
    let identity = Tensor::identity(3, device.clone());
    debug!("\nIdentity matrix: {:?}", identity.to_vec());

    // Create a tensor with random values
    let random = Tensor::random(&[2, 2], device.clone());
    debug!(
        "\nRandom tensor (values between 0 and 1): {:?}",
        random.to_vec()
    );

    // Create a range tensor
    let range = Tensor::arange(0.0, 5.0, device.clone());
    debug!("\nRange tensor: {:?}", range.to_vec());

    debug!("\n=== Shape Operations ===");

    // Reshape a tensor
    let b = range.reshape(&[1, 5]).unwrap();
    debug!(
        "\nReshaped range tensor: shape {:?}, data: {:?}",
        b.shape(),
        b.to_vec()
    );

    // Transpose a tensor
    let c = a.transpose().unwrap();
    debug!(
        "\nTransposed tensor: shape {:?}, data: {:?}",
        c.shape(),
        c.to_vec()
    );

    // Expand dimensions
    let d = a.expand_dims(0).unwrap();
    debug!("\nExpanded dimensions: shape {:?}", d.shape());

    // Squeeze dimensions
    let e = d.squeeze(Some(0)).unwrap();
    debug!("\nSqueezed dimensions: shape {:?}", e.shape());

    debug!("\n=== Reduction Operations ===");

    // Sum of all elements
    let sum = a.sum(None).unwrap();
    debug!("\nSum of all elements: {:?}", sum.to_vec());

    // Sum along axis 0
    let sum_axis0 = a.sum(Some(0)).unwrap();
    debug!("\nSum along axis 0: {:?}", sum_axis0.to_vec());

    // Mean of all elements
    let mean = a.mean(None).unwrap();
    debug!("\nMean of all elements: {:?}", mean.to_vec());

    // Max value and its index
    let max_val = a.max(None).unwrap();
    let argmax = a.argmax(None).unwrap();
    debug!(
        "\nMax value: {:?}, at index: {:?}",
        max_val.to_vec(),
        argmax.to_vec()
    );

    // Min value and its index
    let min_val = a.min(None).unwrap();
    let argmin = a.argmin(None).unwrap();
    debug!(
        "\nMin value: {:?}, at index: {:?}",
        min_val.to_vec(),
        argmin.to_vec()
    );

    debug!("\n=== Element-wise Operations ===");

    // Element-wise operations with scalars
    let f = a.clone() + 1.0;
    debug!("\nAdd scalar: {:?}", f.to_vec());

    let g = a.clone() * 2.0;
    debug!("\nMultiply by scalar: {:?}", g.to_vec());

    // Element-wise operations between tensors
    let h = a.clone().add(&ones).unwrap();
    debug!("\nAdd tensors: {:?}", h.to_vec());

    let i = a.mul(&a).unwrap();
    debug!("\nMultiply tensors: {:?}", i.to_vec());

    debug!("\n=== Matrix Multiplication ===");

    // Matrix multiplication
    let j = a.matmul(&a.transpose().unwrap()).unwrap();
    debug!(
        "\nMatrix multiplication result: shape {:?}, data: {:?}",
        j.shape(),
        j.to_vec()
    );

    // Dot product of two vectors
    let k = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], device.clone()).unwrap();
    let l = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3], device).unwrap();
    let dot = k
        .reshape(&[1, 3])
        .unwrap()
        .matmul(&l.reshape(&[3, 1]).unwrap())
        .unwrap();
    debug!("\nDot product of [1,2,3] and [4,5,6]: {}", dot.to_vec()[0]);

    debug!("\n=== ReLU Activation ===");

    // ReLU activation
    let m = Tensor::from_vec(vec![-1.0, 0.0, 2.0, -3.0], &[2, 2], Device::Cpu(None)).unwrap();
    let relu = m.relu().unwrap();
    debug!("\nReLU of {:?} = {:?}", m.to_vec(), relu.to_vec());

    // Benchmark operations
    debug!("\n=== Benchmarking Operations ===");

    // Benchmark sum
    {
        let sum = large_tensor.sum(None).unwrap();
        debug!("\nSum of {} elements: {:.2}", size, sum.to_vec()[0]);
    }

    // Benchmark element-wise operations
    debug!("\nElement-wise addition");
    {
        let _result = large_tensor.clone() + 1.0;
        debug!("\n addition completed {}", _result.to_vec()[0]);
    }

    debug!(
        "\nBig Matrix multiplication {:?}x{:?}:",
        dims,
        dims.iter().rev().collect::<Vec<&usize>>()
    );

    {
        let _result = large_tensor
            .matmul(&large_tensor.transpose().unwrap())
            .unwrap();
        debug!("\n Matrix multiplication completed {}", _result.to_vec()[0]);
    }

    debug!("\n=== Example completed!");
    debug!("Check the console output above for detailed tracing information.");
    debug!("Set RUST_LOG=rustic_net=info for high-level info or RUST_LOG=rustic_net=trace for maximum detail.");
    debug!("Control parallelism with RUSTIC_NET_NUM_THREADS environment variable (e.g., RAYON_NUM_THREADS=4).");
}
