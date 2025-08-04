use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rustic_net::tensor::{Device, Tensor};

fn tensor_creation_benchmark(c: &mut Criterion) {
    let sizes = [10, 100, 1000];

    let mut group = c.benchmark_group("Tensor Creation");

    for &size in &sizes {
        group.bench_with_input(BenchmarkId::new("from_vec", size), &size, |b, &size| {
            let data = vec![1.0; size];
            b.iter(|| {
                std::hint::black_box(
                    Tensor::from_vec(data.clone(), &[size], Device::Cpu(None)).unwrap(),
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("ones", size), &size, |b, &size| {
            b.iter(|| std::hint::black_box(Tensor::ones(&[size], Device::Cpu(None))))
        });
    }
    group.finish();
}

fn tensor_ops_benchmark(c: &mut Criterion) {
    let sizes = [10, 100, 1000];

    let mut group = c.benchmark_group("Tensor Operations");

    for &size in &sizes {
        let t1 = Tensor::ones(&[size], Device::Cpu(None));
        let t2 = Tensor::ones(&[size], Device::Cpu(None));

        group.bench_with_input(BenchmarkId::new("relu", size), &size, |b, _| {
            b.iter(|| std::hint::black_box(t1.relu().unwrap()))
        });

        group.bench_with_input(BenchmarkId::new("add_scalar", size), &size, |b, _| {
            let t = t1.clone();
            b.iter(|| std::hint::black_box(t.clone() + 1.0))
        });
    }
    group.finish();
}

criterion_group!(benches, tensor_creation_benchmark, tensor_ops_benchmark);
criterion_main!(benches);
