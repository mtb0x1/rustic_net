// //! # Tensor Backends
// //!
// //! Provides backend-specific implementations of tensor operations with different
// //! performance characteristics and hardware support.
// //!
// //! ## Available Backends
// //! - `cpu_seq`: Single-threaded CPU implementation
// //! - `cpu_par`: Multi-threaded CPU implementation (enabled with `parallel` feature)
// //!
// //! The backend is selected automatically based on feature flags and available hardware.
// //! All backends implement the traits defined in the `traits` module.

// #[cfg(feature = "parallel")]
// pub mod cpu_par;
// pub mod cpu_seq;
// #[cfg(feature = "simd")]
// pub mod cpu_simd;
// pub mod traits;
//! # Tensor Backends
//!
//! This module provides different backends for tensor operations.
//! The backend is chosen at compile time based on the features enabled.
//!
//! The available backends are:
//! - `cpu_seq`: A single-threaded, sequential backend.
//! - `cpu_simd`: A single-threaded backend that uses SIMD instructions.
//! - `cpu_par`: A multi-threaded backend that uses rayon.
//! - `cpu_simd_par`: A multi-threaded backend that uses both rayon and SIMD instructions.

pub mod traits;

#[cfg(all(
    not(feature = "parallel"),
    not(feature = "simd"),
    not(feature = "simd_and_parallel"),
    not(feature = "cuda"),
    not(feature = "webgpu")
))]
pub mod cpu_seq;

#[cfg(all(
    not(feature = "parallel"),
    not(feature = "simd"),
    not(feature = "simd_and_parallel"),
    not(feature = "cuda"),
    not(feature = "webgpu")
))]
pub use cpu_seq::CpuSequential as Cpu;


#[cfg(feature = "simd")]
pub mod cpu_simd;

#[cfg(feature = "simd")]
pub use cpu_simd::CpuSimd as Cpu;

#[cfg(feature = "parallel")]
pub mod cpu_par;

#[cfg(feature = "parallel")]
pub use cpu_par::CpuParallel as Cpu;

#[cfg(feature = "simd_and_parallel")]
pub mod cpu_simd_par;

#[cfg(feature = "simd_and_parallel")]
pub use cpu_simd_par::CpuSimdPar as Cpu;
