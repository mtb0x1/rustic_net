#![cfg(feature = "simd")]

use rustic_net::tensor::backends::traits::{BinaryElementwiseOps, UnaryOps};
use rustic_net::tensor::{Device, Tensor};
use rustic_net::RusticNetInitTracing;
use std::sync::Once;

static TRACING_INIT: Once = Once::new();

fn setup() {
    TRACING_INIT.call_once(|| {
        RusticNetInitTracing();
    });
}

#[test]
fn test_simd_relu() {
    setup();
    let a = Tensor::from_vec(
        vec![-1.0, 0.0, 1.0, -2.0, 2.0, -3.0, 3.0, 0.5],
        &[8],
        Device::default(),
    )
    .unwrap();
    let relu_seq = rustic_net::tensor::backends::cpu_seq::CpuSequential::relu(&a).unwrap();
    let relu_simd = rustic_net::tensor::backends::cpu_simd::CpuSimd::relu(&a).unwrap();
    assert_eq!(relu_seq.to_vec(), relu_simd.to_vec());
}

#[test]
fn test_simd_add() {
    setup();
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[8],
        Device::default(),
    )
    .unwrap();
    let b = Tensor::from_vec(
        vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        &[8],
        Device::default(),
    )
    .unwrap();
    let add_seq = rustic_net::tensor::backends::cpu_seq::CpuSequential::add(&a, &b).unwrap();
    let add_simd = rustic_net::tensor::backends::cpu_simd::CpuSimd::add(&a, &b).unwrap();
    assert_eq!(add_seq.to_vec(), add_simd.to_vec());
}

#[test]
fn test_simd_sub() {
    setup();
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[8],
        Device::default(),
    )
    .unwrap();
    let b = Tensor::from_vec(
        vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        &[8],
        Device::default(),
    )
    .unwrap();
    let sub_seq = rustic_net::tensor::backends::cpu_seq::CpuSequential::sub(&a, &b).unwrap();
    let sub_simd = rustic_net::tensor::backends::cpu_simd::CpuSimd::sub(&a, &b).unwrap();
    assert_eq!(sub_seq.to_vec(), sub_simd.to_vec());
}

#[test]
fn test_simd_mul() {
    setup();
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[8],
        Device::default(),
    )
    .unwrap();
    let b = Tensor::from_vec(
        vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        &[8],
        Device::default(),
    )
    .unwrap();
    let mul_seq = rustic_net::tensor::backends::cpu_seq::CpuSequential::mul(&a, &b).unwrap();
    let mul_simd = rustic_net::tensor::backends::cpu_simd::CpuSimd::mul(&a, &b).unwrap();
    assert_eq!(mul_seq.to_vec(), mul_simd.to_vec());
}

#[test]
fn test_simd_div() {
    setup();
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[8],
        Device::default(),
    )
    .unwrap();
    let b = Tensor::from_vec(
        vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        &[8],
        Device::default(),
    )
    .unwrap();
    let div_seq = rustic_net::tensor::backends::cpu_seq::CpuSequential::div(&a, &b).unwrap();
    let div_simd = rustic_net::tensor::backends::cpu_simd::CpuSimd::div(&a, &b).unwrap();
    assert_eq!(div_seq.to_vec(), div_simd.to_vec());
}
