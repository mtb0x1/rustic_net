//! Implementations of scalar operations for Tensor
//!
//! This module contains implementations of basic arithmetic operations between tensors and scalars.

use super::Tensor;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;
use tracing::debug;

// Helper macro to implement scalar operations
macro_rules! impl_scalar_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait<f32> for Tensor {
            type Output = Tensor;

            fn $method(self, rhs: f32) -> Self::Output {
                debug!("tensor::impl_ops::{}_scalar", stringify!($method));

                // For scalar operations, we can use the parallel version if available
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;

                    let data = self.data.par_iter().map(|&x| x $op rhs).collect();
                    Tensor {
                        data: Arc::new(data),
                        shape: self.shape,
                        device: self.device,
                        dtype: self.dtype,
                    }
                }

                #[cfg(not(feature = "parallel"))]
                {
                    let data = self.data.iter().map(|&x| x $op rhs).collect();
                    Tensor {
                        data: Arc::new(data),
                        shape: self.shape,
                        device: self.device,
                        dtype: self.dtype,
                    }
                }
            }
        }

        impl $trait<f32> for &Tensor {
            type Output = Tensor;

            fn $method(self, rhs: f32) -> Self::Output {
                self.clone().$method(rhs)
            }
        }
    };
}

// Implement Add, Sub, Mul, Div for Tensor + f32
impl_scalar_op!(Add, add, +);
impl_scalar_op!(Sub, sub, -);
impl_scalar_op!(Mul, mul, *);
impl_scalar_op!(Div, div, /);

// Implement reverse operations (f32 + Tensor, etc.)

macro_rules! impl_reverse_scalar_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait<&Tensor> for f32 {
            type Output = Tensor;

            fn $method(self, rhs: &Tensor) -> Self::Output {
                debug!("tensor::impl_ops::reverse_{}_scalar", stringify!($method));

                // For scalar operations, we can use the parallel version if available
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;

                    let data = rhs.data.par_iter().map(|&x| self $op x).collect();
                    Tensor {
                        data: Arc::new(data),
                        shape: rhs.shape.clone(),
                        device: rhs.device,
                        dtype: rhs.dtype,
                    }
                }

                #[cfg(not(feature = "parallel"))]
                {
                    let data = rhs.data.iter().map(|&x| self $op x).collect();
                    Tensor {
                        data: Arc::new(data),
                        shape: rhs.shape.clone(),
                        device: rhs.device,
                        dtype: rhs.dtype,
                    }
                }
            }
        }

        impl $trait<Tensor> for f32 {
            type Output = Tensor;

            fn $method(self, rhs: Tensor) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

// Implement reverse operations for f32 + Tensor, etc.
impl_reverse_scalar_op!(Add, add, +);
impl_reverse_scalar_op!(Sub, sub, -);
impl_reverse_scalar_op!(Mul, mul, *);
impl_reverse_scalar_op!(Div, div, /);

// Implement in-place operations

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

// Implement negation

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
