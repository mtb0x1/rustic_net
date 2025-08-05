//! # Tracing and Logging Infrastructure
//!
//! Provides structured logging and tracing capabilities for the Rustic Net library.
//! Implements configurable logging levels, function instrumentation, and specialized
//! tracing for tensor operations and model execution.
//!
//! ## Key Features
//! - Environment-based log level configuration (`RUST_LOG`)
//! - Function entry/exit tracing
//! - Operation performance monitoring
//! - Tensor operation analytics
//! - Model execution profiling

use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Initializes the global tracing subscriber with production-appropriate defaults.
///
/// Configures and installs a tracing subscriber with:
/// - Environment-based log level filtering (`RUST_LOG`)
/// - Thread-aware logging with source locations
/// - Structured JSON output for machine processing
/// - Default log level: ERROR
///
/// # Panics
/// If the global subscriber cannot be initialized.
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

/// Instruments a function with automatic entry/exit tracing.
///
/// Creates a TRACE-level span that:
/// - Automatically logs function entry
/// - Captures function parameters (optional)
/// - Closes span on scope exit
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
