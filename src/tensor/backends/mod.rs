//! # Tensor Backends
//!
//! Provides backend-specific implementations of tensor operations with different
//! performance characteristics and hardware support.
//!
//! ## Available Backends
//! - `cpu_seq`: Single-threaded CPU implementation
//! - `cpu_par`: Multi-threaded CPU implementation (enabled with `parallel` feature)
//!
//! The backend is selected automatically based on feature flags and available hardware.
//! All backends implement the traits defined in the `traits` module.

#[cfg(feature = "parallel")]
pub mod cpu_par;
pub mod cpu_seq;
#[cfg(feature = "simd")]
pub mod cpu_simd;
pub mod traits;
