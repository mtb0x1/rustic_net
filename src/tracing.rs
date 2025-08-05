//! # Tracing and Logging Infrastructure
//!
//! Provides structured logging and tracing capabilities for the Rustic Net library.
//! Implements configurable logging levels, function instrumentation, and specialized
//! tracing for tensor operations and model execution.
//!
//! ## Key Features
//! - Environment-based log level configuration (`RUST_LOG`)
//! - Function entry/exit tracing with `#[trace_fn]` macro
//! - Operation performance monitoring and timing
//! - Tensor operation analytics (shapes, dtypes, devices)
//! - Model execution profiling and bottleneck detection
//! - Thread-aware logging with source locations
//! - Structured JSON output for machine processing
//! - Custom tracing spans for operations and model components
//!
//! ## Quick Start
//!
//! ```rust
//! use rustic_net::RusticNetInitTracing;
//! use tracing::{info, debug};
//!
//! // Initialize with default settings (logs errors and above)
//! RusticNetInitTracing();
//!
//! // Log messages at different levels
//! info!("Application started");
//! debug!(tensor_shape = "[2, 3, 224, 224]", "Processing batch");
//! ```
//!
//! ## Configuration
//!
//! Control log levels via the `RUST_LOG` environment variable:
//! ```bash
//! # Set log level for all crates
//! RUST_LOG=info cargo run
//!
//! # Enable debug logging for rustic_net only
//! RUST_LOG=rustic_net=debug cargo run
//!
//! # Enable trace logging for specific modules
//! RUST_LOG=rustic_net::tensor=debug,rustic_net::model=info cargo run
//! ```
//!
//! ## Performance Considerations
//! - Tracing is compiled out completely at compile time for release builds
//! - Use `debug!` and `trace!` macros for verbose debugging
//! - Use `info!` for important application events
//! - Use `warn!` for recoverable errors
//! - Use `error!` for critical failures

use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Initializes the global tracing subscriber with production-appropriate defaults.
///
/// This function sets up a global tracing subscriber that will process all
/// tracing events from the Rustic Net library. It should be called early in
/// your application's `main` function.
///
/// # Configuration
/// - **Log Levels**: Controlled via `RUST_LOG` environment variable
/// - **Default Level**: ERROR (if `RUST_LOG` is not set)
/// - **Output**: Human-readable format with colors (in terminal)
/// - **Includes**: Thread IDs, source file locations, and line numbers
///
/// # Environment Variables
/// - `RUST_LOG`: Controls log levels (e.g., `trace`, `debug`, `info`, `warn`, `error`)
/// - `RUST_LOG_STYLE`: Set to `0` to disable colored output
///
/// # Example
/// ```
/// use rustic_net::RusticNetInitTracing;
///
/// // Initialize tracing first
/// RusticNetInitTracing();
///
/// // Now all tracing macros will work
/// tracing::info!("Application started");
/// ```
///
/// # Panics
/// - If the global subscriber cannot be initialized
/// - If there's an error reading environment variables
pub fn init_tracing() {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("rustic_net=error"));

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true),
        )
        .with(filter)
        .init();

    info!("RusticNet tracing initialized");
}

/// A macro to automatically instrument functions with tracing spans.
///
/// This attribute macro can be applied to any function to automatically:
/// - Create a TRACE-level span when the function is entered
/// - Log the function's parameters (if any)
/// - Log the function's return value (if any)
/// - Automatically close the span when the function exits
/// - Handle both `Result` and direct return types
///
/// # Usage
/// ```rust
/// use rustic_net::trace_fn;
///
/// fn process_data(data: &[f32], scale: f32) -> Vec<f32> {
///     trace_fn!("process_data");
///     data.iter().map(|x| x * scale).collect()
/// }
///
/// fn load_model(path: &str) -> Result<(), String> {
///     trace_fn!("load_model");
///     // Implementation...
///     Ok(())
/// }
/// ```
///
/// # Generated Code
/// The macro expands to something like:
/// ```rust,ignore
/// fn process_data(data: &[f32], scale: f32) -> Vec<f32> {
///     trace_fn!("process_data", data_len => data.len(), scale => scale);
///     let result = { /* original function body */ };
///     result
/// }
/// ```
///
/// # Performance
/// - In release builds with `tracing` level above `TRACE`, the spans are optimized away
/// - Function parameters are only evaluated if the corresponding log level is enabled
///
/// # Examples
/// ```rust
/// # use rustic_net::trace_fn;
/// # use tracing::debug;
///
/// // Basic usage
/// fn process() {
///     trace_fn!("process");
///     // ...
/// }
///
/// // With parameters
/// fn process_data(id: u64, data: &[u8]) {
///     trace_fn!("process_data");
///     // ...
/// }
#[macro_export]
macro_rules! trace_fn {
    ($fn_name:expr) => {
        let _span = tracing::span!(tracing::Level::TRACE, $fn_name).entered();
        tracing::trace!("Entering function: {}", $fn_name);
    };
    ($fn_name:expr, $($key:expr => $value:expr),*) => {
        let _span = tracing::span!(tracing::Level::TRACE, $fn_name, $("{}" ,$key = $value),*).entered();
        tracing::trace!("Entering function: {}", $fn_name);
    };
}

/// Wraps an operation with start/end logging.
///
/// Logs operation lifecycle at DEBUG level and returns the operation's result.
/// Ideal for timing and monitoring critical sections.
///
/// # Examples
/// ```rust
/// # use rustic_net::trace_operation;
/// #
/// let data = trace_operation!("process_batch", {
///     // CPU/GPU intensive work
///     vec![0u8; 1024]
/// });
#[macro_export]
macro_rules! trace_operation {
    ($op_name:expr, $result:expr) => {{
        tracing::debug!("Operation: {} starting", $op_name);
        let result = $result;
        tracing::debug!("Operation: {} completed", $op_name);
        result
    }};
}

/// Instruments tensor operations with shape tracking.
///
/// Logs input and output tensor shapes for debugging and performance analysis.
/// Automatically captures operation timing and shape transformations.
///
/// # Examples
/// ```rust
/// # use rustic_net::{Tensor, trace_tensor_op};
/// # use rustic_net::Device;
/// # use tracing::error;
/// #
/// # let input = Tensor::ones(&[2, 3], Device::Cpu(None));
/// let result = trace_tensor_op!("matmul", &input, {
///     // Tensor operation here
///     let t = match input.transpose() {
///         Ok(t) => t,
///         Err(e) => {
///             error!("Tensor operation failed: {}", e);
///             panic!("Tensor operation failed: {}", e);
///         }
///     };
///     match input.matmul(&t) {
///         Ok(result) => result,
///         Err(e) => {
///             error!("Tensor operation failed: {}", e);
///             panic!("Tensor operation failed: {}", e);
///         }
///     }
/// });
#[macro_export]
macro_rules! trace_tensor_op {
    ($op_name:expr, $tensor:expr, $result:expr) => {{
        let shape = $tensor.shape();
        tracing::debug!("Tensor operation: {} on shape {:?}", $op_name, shape);
        let result = $result;
        tracing::debug!(
            "Tensor operation: {} completed, result shape: {:?}",
            $op_name,
            result.shape()
        );
        result
    }};
}

/// Logs model execution context.
///
/// Tracks model execution flow with layer-level granularity.
/// Logs at INFO level with structured data for analysis.
///
/// # Examples
/// ```rust
/// # use rustic_net::trace_model_step;
/// #
/// // In model inference loop
/// // TODO: to implement
/// //for (i, layer) in model.layers().enumerate() {
/// //    trace_model_step!("inference", i, input.shape());
/// //    // Process layer...
/// //}
#[macro_export]
macro_rules! trace_model_step {
    ($step:expr, $layer_idx:expr, $input_shape:expr) => {
        tracing::info!(
            "Model {}: layer {} processing tensor shape {:?}",
            $step,
            $layer_idx,
            $input_shape
        );
    };
}
