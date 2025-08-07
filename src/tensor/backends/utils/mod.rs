//! CPU-specific implementations and utilities for tensor operations.
//!
//! This module contains CPU-specific implementations of tensor operations,
//! including parallel computation utilities and SIMD-accelerated operations.
#[cfg(any(feature = "parallel", feature = "simd_and_parallel"))]
pub mod parallel_utils;
#[cfg(any(feature = "simd", feature = "simd_and_parallel"))]
pub mod simd_utils;

#[cfg(any(feature = "parallel", feature = "simd_and_parallel"))]
pub use parallel_utils::*;
#[cfg(any(feature = "simd", feature = "simd_and_parallel"))]
pub use simd_utils::*;
