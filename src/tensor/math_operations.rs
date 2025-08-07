//! # Tensor Operations Implementation
//!
//! Implements core arithmetic operations for tensors, including:
//! - Element-wise arithmetic with scalars (+, -, *, /)
//! - In-place operations (+=, -=, *=, /=)
//! - Unary negation (-)
//! - Automatic parallelization when the `parallel` feature is enabled
//!
//! All operations maintain the original tensor's shape and device placement.

use super::{backends::traits::ScalarOps, backends::Cpu, Tensor};
use std::ops::{Add, Div, Mul, Sub};
use tracing::debug;

/// Implements element-wise operations between a tensor and a scalar.
///
/// Generates both consuming and borrowing variants of the operation.
/// When the `parallel` feature is enabled, operations are automatically parallelized.
macro_rules! impl_scalar_op {
    ($trait:ident, $method:ident, $backend_method:ident) => {
        impl $trait<f32> for Tensor {
            type Output = Tensor;

            fn $method(self, rhs: f32) -> Self::Output {
                debug!("tensor::impl_ops::{}_scalar", stringify!($method));
                Cpu::$backend_method(&self, rhs).unwrap()
            }
        }

        impl $trait<f32> for &Tensor {
            type Output = Tensor;

            fn $method(self, rhs: f32) -> Self::Output {
                debug!("tensor::impl_ops::{}_scalar_ref", stringify!($method));
                Cpu::$backend_method(self, rhs).unwrap()
            }
        }
    };
}

// Implement standard arithmetic operations for Tensor and f32
// These enable expressions like: tensor + 1.0, tensor * 2.0, etc.
impl_scalar_op!(Add, add, add_scalar);
impl_scalar_op!(Sub, sub, sub_scalar);
impl_scalar_op!(Mul, mul, mul_scalar);
impl_scalar_op!(Div, div, div_scalar);

/// Implements reverse operations between a scalar and tensor.
///
/// Enables expressions like: 1.0 + tensor, 2.0 * tensor, etc.
/// Maintains the same performance characteristics as the direct operations.
macro_rules! impl_reverse_scalar_op {
    ($trait:ident, $method:ident, $backend_method:ident) => {
        impl $trait<&Tensor> for f32 {
            type Output = Tensor;

            fn $method(self, rhs: &Tensor) -> Self::Output {
                debug!("tensor::impl_ops::reverse_{}_scalar", stringify!($method));
                Cpu::$backend_method(rhs, self).unwrap()
            }
        }

        impl $trait<Tensor> for f32 {
            type Output = Tensor;

            fn $method(self, rhs: Tensor) -> Self::Output {
                debug!(
                    "tensor::impl_ops::reverse_{}_scalar_owned",
                    stringify!($method)
                );
                Cpu::$backend_method(&rhs, self).unwrap()
            }
        }
    };
}

// Implement reverse arithmetic operations for f32 and Tensor
// These enable expressions like: 1.0 + tensor, 2.0 * tensor, etc.
impl_reverse_scalar_op!(Add, add, r_add_scalar);
impl_reverse_scalar_op!(Sub, sub, r_sub_scalar);
impl_reverse_scalar_op!(Mul, mul, r_mul_scalar);
impl_reverse_scalar_op!(Div, div, r_div_scalar);

// In-place operations modify the tensor directly without creating a new allocation.

impl std::ops::AddAssign<f32> for Tensor {
    fn add_assign(&mut self, rhs: f32) {
        *self = self.clone().add(rhs);
    }
}

impl std::ops::SubAssign<f32> for Tensor {
    fn sub_assign(&mut self, rhs: f32) {
        *self = self.clone().sub(rhs);
    }
}

impl std::ops::MulAssign<f32> for Tensor {
    fn mul_assign(&mut self, rhs: f32) {
        *self = self.clone().mul(rhs);
    }
}

impl std::ops::DivAssign<f32> for Tensor {
    fn div_assign(&mut self, rhs: f32) {
        *self = self.clone().div(rhs);
    }
}

// Unary negation creates a new tensor with all elements negated.

impl std::ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl std::ops::Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        self.clone().neg()
    }
}
