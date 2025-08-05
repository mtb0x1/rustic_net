//! # Tensor Operations Example with Parallel Processing
//!
//! This example demonstrates various tensor operations available in the Rustic Net library,
//! including parallel processing capabilities.

#[cfg(feature = "parallel")]
use rustic_net::parallel::init_thread_pool;
use rustic_net::tensor::{Device, Tensor};
use rustic_net::RusticNetInitTracingInit;
use std::time::Instant;

fn main() {
    RusticNetInitTracingInit();

    #[cfg(feature = "parallel")]
    init_thread_pool();

    // Set up the device (CPU with parallel processing)
    let device = Device::default();

    // Display available parallelism

    // Create a large tensor to demonstrate parallel operations
    let size = 1_000_000; // 1 million elements
    let large_data: Vec<f32> = (0..size).map(|x| x as f32).collect();
    let large_tensor = Tensor::from_vec(large_data.clone(), &[size], device.clone()).unwrap();

    println!("=== Tensor Creation ===");

    // Create a tensor from a vector
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], device.clone()).unwrap();
    println!(
        "Tensor from vector: shape {:?}, data: {:?}",
        a.shape(),
        a.to_vec()
    );

    // Create a tensor of ones
    let ones = Tensor::ones(&[2, 2], device.clone());
    println!("\nTensor of ones: {:?}", ones.to_vec());

    // Create a tensor of zeros
    let zeros = Tensor::zeros(&[3, 3], device.clone());
    println!("\nTensor of zeros: {:?}", zeros.to_vec());

    // Create an identity matrix
    let identity = Tensor::identity(3, device.clone());
    println!("\nIdentity matrix: {:?}", identity.to_vec());

    // Create a tensor with random values
    let random = Tensor::random(&[2, 2], device.clone());
    println!(
        "\nRandom tensor (values between 0 and 1): {:?}",
        random.to_vec()
    );

    // Create a range tensor
    let range = Tensor::arange(0.0, 5.0, device.clone());
    println!("\nRange tensor: {:?}", range.to_vec());

    println!("\n=== Shape Operations ===");

    // Reshape a tensor
    let b = range.reshape(&[1, 5]).unwrap();
    println!(
        "\nReshaped range tensor: shape {:?}, data: {:?}",
        b.shape(),
        b.to_vec()
    );

    // Transpose a tensor
    let c = a.transpose().unwrap();
    println!(
        "\nTransposed tensor: shape {:?}, data: {:?}",
        c.shape(),
        c.to_vec()
    );

    // Expand dimensions
    let d = a.expand_dims(0).unwrap();
    println!("\nExpanded dimensions: shape {:?}", d.shape());

    // Squeeze dimensions
    let e = d.squeeze(Some(0)).unwrap();
    println!("Squeezed dimensions: shape {:?}", e.shape());

    println!("\n=== Reduction Operations ===");

    // Sum of all elements
    let sum = a.sum(None).unwrap();
    println!("\nSum of all elements: {:?}", sum.to_vec());

    // Sum along axis 0
    let sum_axis0 = a.sum(Some(0)).unwrap();
    println!("Sum along axis 0: {:?}", sum_axis0.to_vec());

    // Mean of all elements
    let mean = a.mean(None).unwrap();
    println!("\nMean of all elements: {:?}", mean.to_vec());

    // Max value and its index
    let max_val = a.max(None).unwrap();
    let argmax = a.argmax(None).unwrap();
    println!(
        "\nMax value: {:?}, at index: {:?}",
        max_val.to_vec(),
        argmax.to_vec()
    );

    // Min value and its index
    let min_val = a.min(None).unwrap();
    let argmin = a.argmin(None).unwrap();
    println!(
        "Min value: {:?}, at index: {:?}",
        min_val.to_vec(),
        argmin.to_vec()
    );

    println!("\n=== Element-wise Operations ===");

    // Element-wise operations with scalars
    let f = a.clone() + 1.0;
    println!("\nAdd scalar: {:?}", f.to_vec());

    let g = a.clone() * 2.0;
    println!("Multiply by scalar: {:?}", g.to_vec());

    // Element-wise operations between tensors
    let h = a.add(&ones).unwrap();
    println!("\nAdd tensors: {:?}", h.to_vec());

    let i = a.mul(&a).unwrap();
    println!("Multiply tensors: {:?}", i.to_vec());

    println!("\n=== Matrix Multiplication ===");

    // Matrix multiplication
    let j = a.matmul(&a.transpose().unwrap()).unwrap();
    println!(
        "\nMatrix multiplication result: shape {:?}, data: {:?}",
        j.shape(),
        j.to_vec()
    );

    // Dot product of two vectors
    let k = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], device.clone()).unwrap();
    let l = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3], device).unwrap();
    let dot = k.matmul(&l.reshape(&[3, 1]).unwrap()).unwrap();
    println!("\nDot product of [1,2,3] and [4,5,6]: {}", dot.to_vec()[0]);

    println!("\n=== ReLU Activation ===");

    // ReLU activation
    let m = Tensor::from_vec(vec![-1.0, 0.0, 2.0, -3.0], &[2, 2], Device::Cpu(None)).unwrap();
    let relu = m.relu().unwrap();
    println!("\nReLU of {:?} = {:?}", m.to_vec(), relu.to_vec());

    // Benchmark parallel operations
    println!("\n=== Benchmarking Parallel Operations ===");

    // Benchmark parallel sum
    let start = Instant::now();
    let sum = large_tensor.sum(None).unwrap();
    let duration = start.elapsed();
    println!(
        "Parallel sum of {} elements: {:.2} (took {:?})",
        size,
        sum.to_vec()[0],
        duration
    );

    // Benchmark element-wise operations
    let start = Instant::now();
    let _result = large_tensor.clone() + 1.0;
    let duration = start.elapsed();
    println!("Parallel element-wise addition: {:?}", duration);

    // Benchmark matrix multiplication (if size is appropriate)
    if size <= 10_000 {
        // Keep it reasonable for demonstration
        let m = (size as f32).sqrt() as usize;
        let square_tensor = Tensor::from_vec(
            (0..m * m).map(|x| x as f32).collect::<Vec<f32>>(),
            &[m, m],
            device.clone(),
        )
        .unwrap();

        let start = Instant::now();
        let _result = square_tensor.matmul(&square_tensor.transpose().unwrap());
        let duration = start.elapsed();
        println!(
            "Parallel matrix multiplication ({}x{}): {:?}",
            m, m, duration
        );
    }

    println!("\n=== Example completed!");
    println!("Check the console output above for detailed tracing information.");
    println!("Set RUST_LOG=rustic_net=info for high-level info or RUST_LOG=rustic_net=trace for maximum detail.");
    println!("Control parallelism with RAYON_NUM_THREADS environment variable (e.g., RAYON_NUM_THREADS=4).");
}
