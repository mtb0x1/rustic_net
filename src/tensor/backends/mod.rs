//! Backend implementations for tensor operations
//!
//! This module contains the backend implementations for tensor operations.
//! Each backend provides implementations of the operation traits defined in `traits.rs`.

#[cfg(feature = "parallel")]
pub mod cpu_par;
pub mod cpu_seq;
pub mod traits;
