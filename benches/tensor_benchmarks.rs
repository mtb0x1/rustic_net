use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rustic_net::tensor::{Device, Tensor};
use std::thread::available_parallelism;
use std::time::Duration;

fn tensor_creation_benchmark(c: &mut Criterion) {
    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    let device = Device::default();

    let mut group = c.benchmark_group("Tensor Creation");
    group.measurement_time(Duration::from_secs(5));

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("from_vec", size), &size, |b, &size| {
            let data = vec![1.0; size];
            b.iter(|| {
                std::hint::black_box(
                    Tensor::from_vec(data.clone(), &[size], device.clone()).unwrap(),
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("ones", size), &size, |b, &size| {
            b.iter(|| std::hint::black_box(Tensor::ones(&[size], device.clone())))
        });
    }
    group.finish();
}

fn tensor_ops_benchmark(c: &mut Criterion) {
    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    let device = Device::default();

    let mut group = c.benchmark_group("Tensor Operations");
    group.measurement_time(Duration::from_secs(5));

    for &size in &sizes {
        let t1 = Tensor::ones(&[size], device.clone());
        let t2 = Tensor::ones(&[size], device.clone());

        group.throughput(Throughput::Elements(size as u64));

        // Element-wise operations
        group.bench_with_input(BenchmarkId::new("add_scalar", size), &size, |b, _| {
            let t = t1.clone();
            b.iter(|| std::hint::black_box(t.clone() + 1.0))
        });

        group.bench_with_input(BenchmarkId::new("add_tensors", size), &size, |b, _| {
            let t1 = t1.clone();
            let t2 = t2.clone();
            b.iter(|| std::hint::black_box(t1.add(&t2).unwrap()))
        });

        group.bench_with_input(BenchmarkId::new("multiply_tensors", size), &size, |b, _| {
            let t1 = t1.clone();
            let t2 = t2.clone();
            b.iter(|| std::hint::black_box(t1.mul(&t2).unwrap()))
        });

        // Reduction operations
        group.bench_with_input(BenchmarkId::new("sum", size), &size, |b, _| {
            let t = t1.clone();
            b.iter(|| std::hint::black_box(t.sum(None).unwrap()))
        });

        group.bench_with_input(BenchmarkId::new("mean", size), &size, |b, _| {
            let t = t1.clone();
            b.iter(|| std::hint::black_box(t.mean(None).unwrap()))
        });

        group.bench_with_input(BenchmarkId::new("relu", size), &size, |b, _| {
            let t = t1.clone() - 0.5; // Some negative values for ReLU
            b.iter(|| std::hint::black_box(t.relu().unwrap()))
        });
    }
    group.finish();
}

fn matrix_ops_benchmark(c: &mut Criterion) {
    let sizes = [10, 32, 64, 128, 256]; // Matrix dimensions (n x n)
    let device = Device::default();

    let mut group = c.benchmark_group("Matrix Operations");
    group.measurement_time(Duration::from_secs(10));

    for &size in &sizes {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));

        let m1 = Tensor::ones(&[size, size], device.clone());
        let m2 = Tensor::ones(&[size, size], device.clone());

        group.bench_with_input(BenchmarkId::new("matmul", size), &size, |b, _| {
            b.iter(|| std::hint::black_box(m1.matmul(&m2).unwrap()))
        });
    }
    group.finish();
}

fn parallel_scaling_benchmark(c: &mut Criterion) {
    let size = 1_000_000;
    let device = Device::default();
    let t = Tensor::ones(&[size], device.clone());

    let mut group = c.benchmark_group("Parallel Scaling");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(size as u64));

    // Test with different thread counts
    let max_threads = available_parallelism().map(|n| n.get()).unwrap_or(1);
    let thread_counts = [1, max_threads / 2, max_threads, max_threads * 2];

    for &threads in &thread_counts {
        std::env::set_var("RAYON_NUM_THREADS", threads.to_string());

        group.bench_with_input(
            BenchmarkId::new("sum_threads", threads),
            &threads,
            |b, _| b.iter(|| std::hint::black_box(t.sum(None).unwrap())),
        );
    }
    group.finish();
}

#[cfg(feature = "simd")]
fn simd_comparison_benchmark(c: &mut Criterion) {
    use rustic_net::tensor::backends::cpu_simd::CpuSimd;
    let size = 1_000_000;
    let device = Device::default();
    let t = Tensor::ones(&[size], device.clone()) - 0.5;

    let mut group = c.benchmark_group("SIMD Comparison");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(size as u64));

    group.bench_function("relu_seq", |b| {
        b.iter(|| {
            let _ = rustic_net::tensor::backends::cpu_seq::CpuSequential::relu(&t);
        })
    });

    group.bench_function("relu_par", |b| {
        b.iter(|| {
            let _ = rustic_net::tensor::backends::cpu_par::CpuParallel::relu(&t);
        })
    });

    group.bench_function("relu_simd", |b| {
        b.iter(|| {
            let _ = CpuSimd::relu(&t);
        })
    });

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1));
    targets = tensor_creation_benchmark, tensor_ops_benchmark, matrix_ops_benchmark, parallel_scaling_benchmark
);

#[cfg(feature = "simd")]
criterion_group!(
    name = simd_benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1));
    targets = simd_comparison_benchmark
);

#[cfg(not(feature = "simd"))]
criterion_main!(benches);
#[cfg(feature = "simd")]
criterion_main!(benches, simd_benches);
criterion_main!(benches);
