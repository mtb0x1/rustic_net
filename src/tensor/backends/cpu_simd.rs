#![cfg(feature = "simd")]
#![feature(portable_simd)]

use super::traits::*;
use crate::tensor::Tensor;
use crate::trace_fn;
use std::simd::{cmp::SimdPartialOrd, f32x8};
use std::sync::Arc;

/// Marker type for the SIMD CPU backend.
///
/// Implements all tensor operation traits using SIMD instructions.
pub struct CpuSimd;

impl UnaryOps for CpuSimd {
    fn relu(tensor: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::relu");
        let mut data = tensor.data.to_vec();
        let len = data.len();
        let (chunks, remainder) = data.as_mut_slice().split_at_mut(len - len % 8);
        let chunks = chunks.chunks_mut(8);

        for chunk in chunks {
            let simd_chunk = f32x8::from_slice(chunk);
            let mask = simd_chunk.simd_gt(f32x8::splat(0.0));
            let result = mask.select(simd_chunk, f32x8::splat(0.0));
            result.copy_to_slice(chunk);
        }

        for val in remainder.iter_mut() {
            if *val < 0.0 {
                *val = 0.0;
            }
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: tensor.shape.clone(),
            device: tensor.device,
            dtype: tensor.dtype,
        })
    }
}

impl BinaryElementwiseOps for CpuSimd {
    fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::add");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise addition".to_string());
        }

        let mut data = vec![0.0; a.data.len()];
        let len = a.data.len();
        let (a_chunks, a_remainder) = a.data.as_slice().split_at(len - len % 8);
        let (b_chunks, b_remainder) = b.data.as_slice().split_at(len - len % 8);
        let (out_chunks, out_remainder) = data.as_mut_slice().split_at_mut(len - len % 8);

        let a_chunks = a_chunks.chunks(8);
        let b_chunks = b_chunks.chunks(8);
        let out_chunks = out_chunks.chunks_mut(8);

        for ((a_chunk, b_chunk), out_chunk) in a_chunks.zip(b_chunks).zip(out_chunks) {
            let simd_a = f32x8::from_slice(a_chunk);
            let simd_b = f32x8::from_slice(b_chunk);
            let result = simd_a + simd_b;
            result.copy_to_slice(out_chunk);
        }

        for ((a_val, b_val), out_val) in a_remainder
            .iter()
            .zip(b_remainder.iter())
            .zip(out_remainder.iter_mut())
        {
            *out_val = a_val + b_val;
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
            device: a.device,
            dtype: a.dtype,
        })
    }

    fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::sub");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise subtraction".to_string());
        }

        let mut data = vec![0.0; a.data.len()];
        let len = a.data.len();
        let (a_chunks, a_remainder) = a.data.as_slice().split_at(len - len % 8);
        let (b_chunks, b_remainder) = b.data.as_slice().split_at(len - len % 8);
        let (out_chunks, out_remainder) = data.as_mut_slice().split_at_mut(len - len % 8);

        let a_chunks = a_chunks.chunks(8);
        let b_chunks = b_chunks.chunks(8);
        let out_chunks = out_chunks.chunks_mut(8);

        for ((a_chunk, b_chunk), out_chunk) in a_chunks.zip(b_chunks).zip(out_chunks) {
            let simd_a = f32x8::from_slice(a_chunk);
            let simd_b = f32x8::from_slice(b_chunk);
            let result = simd_a - simd_b;
            result.copy_to_slice(out_chunk);
        }

        for ((a_val, b_val), out_val) in a_remainder
            .iter()
            .zip(b_remainder.iter())
            .zip(out_remainder.iter_mut())
        {
            *out_val = a_val - b_val;
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
            device: a.device,
            dtype: a.dtype,
        })
    }

    fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::mul");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise multiplication".to_string());
        }

        let mut data = vec![0.0; a.data.len()];
        let len = a.data.len();
        let (a_chunks, a_remainder) = a.data.as_slice().split_at(len - len % 8);
        let (b_chunks, b_remainder) = b.data.as_slice().split_at(len - len % 8);
        let (out_chunks, out_remainder) = data.as_mut_slice().split_at_mut(len - len % 8);

        let a_chunks = a_chunks.chunks(8);
        let b_chunks = b_chunks.chunks(8);
        let out_chunks = out_chunks.chunks_mut(8);

        for ((a_chunk, b_chunk), out_chunk) in a_chunks.zip(b_chunks).zip(out_chunks) {
            let simd_a = f32x8::from_slice(a_chunk);
            let simd_b = f32x8::from_slice(b_chunk);
            let result = simd_a * simd_b;
            result.copy_to_slice(out_chunk);
        }

        for ((a_val, b_val), out_val) in a_remainder
            .iter()
            .zip(b_remainder.iter())
            .zip(out_remainder.iter_mut())
        {
            *out_val = a_val * b_val;
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
            device: a.device,
            dtype: a.dtype,
        })
    }

    fn div(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        trace_fn!("CpuSimd::div");
        if a.shape() != b.shape() {
            return Err("Shapes must match for element-wise division".to_string());
        }

        let mut data = vec![0.0; a.data.len()];
        let len = a.data.len();
        let (a_chunks, a_remainder) = a.data.as_slice().split_at(len - len % 8);
        let (b_chunks, b_remainder) = b.data.as_slice().split_at(len - len % 8);
        let (out_chunks, out_remainder) = data.as_mut_slice().split_at_mut(len - len % 8);

        let a_chunks = a_chunks.chunks(8);
        let b_chunks = b_chunks.chunks(8);
        let out_chunks = out_chunks.chunks_mut(8);

        for ((a_chunk, b_chunk), out_chunk) in a_chunks.zip(b_chunks).zip(out_chunks) {
            let simd_a = f32x8::from_slice(a_chunk);
            let simd_b = f32x8::from_slice(b_chunk);
            let result = simd_a / simd_b;
            result.copy_to_slice(out_chunk);
        }

        for ((a_val, b_val), out_val) in a_remainder
            .iter()
            .zip(b_remainder.iter())
            .zip(out_remainder.iter_mut())
        {
            if *b_val == 0.0 {
                *out_val = f32::NAN;
            } else {
                *out_val = a_val / b_val;
            }
        }

        Ok(Tensor {
            data: Arc::new(data),
            shape: a.shape.clone(),
            device: a.device,
            dtype: a.dtype,
        })
    }
}
