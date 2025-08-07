use rustic_net::tensor::{DType, Device, Tensor};
use rustic_net::RusticNetInitTracing;
use std::sync::Once;

static TRACING_INIT: Once = Once::new();

fn setup() {
    TRACING_INIT.call_once(|| {
        RusticNetInitTracing();
    });
}

#[test]
fn test_device_default() {
    setup();
    let device = Device::default();
    assert_eq!(device, Device::Cpu(Some(0)));
}

#[test]
fn test_dtype_conversion() {
    setup();
    let dtype: DType = "f32".try_into().unwrap();
    assert_eq!(dtype, DType::F32);

    let dtype_str: &str = DType::F32.try_into().unwrap();
    assert_eq!(dtype_str, "f32");

    assert_eq!(DType::F32.size_of(), 4);
}

#[test]
fn test_tensor_creation() {
    setup();
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], Device::default()).unwrap();
    assert_eq!(t.shape(), &[3]);
    assert_eq!(t.to_vec(), &vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_tensor_creation_zeros_ones_identity() {
    setup();
    let t_zeros = Tensor::zeros(&[2, 3], Device::default());
    assert_eq!(t_zeros.to_vec(), &vec![0.0; 6]);

    let t_ones = Tensor::ones(&[2, 3], Device::default());
    assert_eq!(t_ones.to_vec(), &vec![1.0; 6]);

    let t_identity = Tensor::identity(3, Device::default());
    assert_eq!(
        t_identity.to_vec(),
        &vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn test_tensor_creation_random() {
    setup();
    let t = Tensor::random(&[2, 3], Device::default());
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.to_vec().len(), 6);
    for &val in t.to_vec().iter() {
        assert!(val >= 0.0 && val < 1.0);
    }
}

#[test]
fn test_tensor_operations() {
    setup();
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], Device::default()).unwrap();
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3], Device::default()).unwrap();

    // Test element-wise addition
    let c = a.add(&b).unwrap();
    assert_eq!(c.to_vec(), &vec![5.0, 7.0, 9.0]);

    // Test reduction
    let e = c.sum(None).unwrap();
    assert_eq!(e.to_vec(), &vec![21.0]);
}

#[test]
fn test_tensor_binary_ops() {
    setup();
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], Device::default()).unwrap();
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3], Device::default()).unwrap();

    let sub_res = a.sub(&b).unwrap();
    assert_eq!(sub_res.to_vec(), &vec![-3.0, -3.0, -3.0]);

    let mul_res = a.mul(&b).unwrap();
    assert_eq!(mul_res.to_vec(), &vec![4.0, 10.0, 18.0]);

    let div_res = a.div(&b).unwrap();
    assert_eq!(div_res.to_vec(), &vec![0.25, 0.4, 0.5]);
}

#[test]
fn test_tensor_unary_ops() {
    setup();
    let a = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3], Device::default()).unwrap();
    let relu_res = a.relu().unwrap();
    assert_eq!(relu_res.to_vec(), &vec![0.0, 0.0, 1.0]);
}

#[test]
fn test_tensor_reductions() {
    setup();
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        Device::default(),
    )
    .unwrap();

    let mean_res = a.mean(None).unwrap();
    assert_eq!(mean_res.to_vec(), &vec![3.5]);

    let max_res = a.max(None).unwrap();
    assert_eq!(max_res.to_vec(), &vec![6.0]);

    let min_res = a.min(None).unwrap();
    assert_eq!(min_res.to_vec(), &vec![1.0]);
}

#[test]
fn test_matmul() {
    setup();
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::default()).unwrap();
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], Device::default()).unwrap();
    let matmul_res = a.matmul(&b).unwrap();
    assert_eq!(matmul_res.to_vec(), &vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_reshape() {
    setup();
    let t = Tensor::arange(0.0, 6.0, Device::default());
    let t = t.reshape(&[2, 3]).unwrap();
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.to_vec(), &vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_transpose() {
    setup();
    let t = Tensor::from_vec(
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        &[2, 3],
        Device::default(),
    )
    .unwrap();
    let t_t = t.transpose().unwrap();
    assert_eq!(t_t.shape(), &[3, 2]);
    assert_eq!(t_t.to_vec(), &vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}

#[test]
fn test_transpose_axes() {
    setup();
    let t = Tensor::from_vec(
        (0..24).map(|x| x as f32).collect::<Vec<f32>>(),
        &[2, 3, 4],
        Device::default(),
    )
    .unwrap();
    let t_t = t.transpose_axes(&[1, 2, 0]).unwrap();
    assert_eq!(t_t.shape(), &[3, 4, 2]);
    // Manually check a few values
    // Original (0,0,0) -> 0. Transposed (0,0,0) -> 0
    // Original (0,1,2) -> 6. Transposed (1,2,0) -> 6
    // Original (1,2,3) -> 23. Transposed (2,3,1) -> 23
    assert_eq!(t_t.to_vec()[0], 0.0);
    assert_eq!(t_t.to_vec()[12], 6.0);
    assert_eq!(t_t.to_vec()[23], 23.0);
}

#[test]
fn test_squeeze_expand() {
    setup();
    let t = Tensor::arange(0.0, 6.0, Device::default());
    let t = t.reshape(&[1, 2, 1, 3, 1]).unwrap();
    assert_eq!(t.shape(), &[1, 2, 1, 3, 1]);

    let t_squeezed = t.squeeze(None).unwrap();
    assert_eq!(t_squeezed.shape(), &[2, 3]);

    let t_expanded = t_squeezed.expand_dims(1).unwrap();
    assert_eq!(t_expanded.shape(), &[2, 1, 3]);
}
