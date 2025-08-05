//! Backend implementations for tensor operations
//!
//! This module contains the backend implementations for tensor operations.
//! Each backend provides implementations of the operation traits defined in `traits.rs`.

// Public re-exports
pub mod traits;

// Backend implementations
#[cfg(feature = "parallel")]
pub mod cpu_par;
pub mod cpu_seq;

// Conditional imports
#[cfg(feature = "parallel")]
use crate::parallel;

use super::{Device, DType, Shape, Tensor};
use std::sync::Arc;

/// The default backend type to use
#[cfg(not(feature = "parallel"))]
type DefaultBackend = cpu_seq::CpuSequential;

/// The default backend type to use (parallel version)
#[cfg(feature = "parallel")]
type DefaultBackend = cpu_par::CpuParallel;

/// Creates a tensor with the default backend
pub(crate) fn create_tensor(data: Vec<f32>, shape: Shape, device: Device, dtype: DType) -> Tensor {
    DefaultBackend::create_tensor(data, shape, device, dtype)
}

/// Dispatches a unary operation to the appropriate backend
pub(crate) fn unary_op<F>(tensor: &Tensor, op: F) -> Result<Tensor, String>
where
    F: FnOnce(&Tensor) -> Result<Tensor, String>,
{
    match tensor.device {
        Device::Cpu(_) => op(tensor),
        Device::Cuda(_) | Device::WebGpu(_) => {
            // For now, fall back to CPU for non-CPU devices
            // This will be implemented when we add GPU support
            Err(format!("Device not yet supported: {:?}", tensor.device))
        }
    }
}

/// Dispatches a binary operation to the appropriate backend
pub(crate) fn binary_op<F>(a: &Tensor, b: &Tensor, op: F) -> Result<Tensor, String>
where
    F: FnOnce(&Tensor, &Tensor) -> Result<Tensor, String>,
{
    match (a.device, b.device) {
        (Device::Cpu(_), Device::Cpu(_)) => op(a, b),
        _ => {
            // For now, require both tensors to be on the same device
            // This will be enhanced with device transfer logic later
            Err(format!(
                "Operation between tensors on different devices not supported: {:?} vs {:?}",
                a.device, b.device
            ))
        }
    }
}
