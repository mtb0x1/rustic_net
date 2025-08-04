//! # Tracing Utilities for Rustic Net
//!
//! This module provides tracing and logging functionality for the Rustic Net library.
//! It includes initialization of the tracing subscriber and various macros for instrumenting
//! the code with detailed logging information.
//!
//! ## Features
//! - Configurable logging levels via environment variables
//! - Function call tracing with `trace_fn!` macro
//! - Operation tracing with `trace_operation!` macro
//! - Tensor operation tracing with `trace_tensor_op!` macro
//! - Model step tracing with `trace_model_step!` macro

use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Initializes the global tracing subscriber with default configuration.
///
/// This function sets up a tracing subscriber with the following features:
/// - Log level controlled by `RUST_LOG` environment variable (defaults to `rustic_net=trace`)
/// - Includes thread IDs in log output
/// - Includes source file and line numbers in log output
/// - Outputs to stderr
///
/// # Panics
/// Panics if the global default subscriber cannot be set.
pub fn init_tracing() {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("rustic_net=trace"));

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

/// Creates a tracing span for function entry and exit.
///
/// This macro creates a TRACE-level span that automatically logs when a function is entered.
/// The span will be automatically closed when it goes out of scope.
///
/// # Examples
/// ```
/// # use rustic_net::trace_fn;
/// fn my_function() {
///     trace_fn!("my_function");
///     // Function body
/// }
///
/// fn function_with_parameters(x: i32, y: &str) {
///     trace_fn!("function_with_parameters", x => x, y => y);
///     // Function body
/// }
/// ```
#[macro_export]
macro_rules! trace_fn {
    ($fn_name:expr) => {
        let _span = tracing::span!(tracing::Level::TRACE, $fn_name).entered();
        tracing::trace!("Entering function: {}", $fn_name);
    };
    ($fn_name:expr, $($key:expr => $value:expr),*) => {
        let _span = tracing::span!(tracing::Level::TRACE, $fn_name, $($key = $value),*).entered();
        tracing::trace!("Entering function: {}", $fn_name);
    };
}

/// Traces the start and end of an operation with DEBUG level logs.
///
/// This macro takes an operation name and a block of code, and logs when the operation
/// starts and completes. It returns the result of the operation.
///
/// # Examples
/// ```
/// # use rustic_net::trace_operation;
/// let result = trace_operation!("expensive_calculation", {
///     // Expensive calculation here
///     42
/// });
/// ```
#[macro_export]
macro_rules! trace_operation {
    ($op_name:expr, $result:expr) => {{
        tracing::debug!("Operation: {} starting", $op_name);
        let result = $result;
        tracing::debug!("Operation: {} completed", $op_name);
        result
    }};
}

/// Traces tensor operations with shape information.
///
/// This macro is specifically designed for tensor operations. It logs the operation name
/// and input tensor shape before the operation, and the result shape after the operation.
///
/// # Examples
/// ```
/// # use rustic_net::{Tensor, trace_tensor_op};
/// # let input = Tensor::ones(&[2, 3], Device::Cpu(None));
/// let result = trace_tensor_op!("matrix_multiply", input, {
///     // Tensor operation here
///     input.matmul(&input.transpose(None).unwrap()).unwrap()
/// });
/// ```
#[macro_export]
macro_rules! trace_tensor_op {
    ($op_name:expr, $tensor:expr, $result:expr) => {{
        let shape = $tensor.shape;
        tracing::debug!("Tensor operation: {} on shape {:?}", $op_name, shape);
        let result = $result;
        tracing::debug!(
            "Tensor operation: {} completed, result shape: {:?}",
            $op_name,
            result.shape
        );
        result
    }};
}

/// Logs information about model processing steps.
///
/// This macro is used to log information about model processing steps, such as when
/// a layer is processing input. It logs at INFO level and includes the step name,
/// layer index, and input shape.
///
/// # Examples
/// ```
/// # use rustic_net::trace_model_step;
/// # let layer_idx = 0;
/// # let input_shape = [1, 28, 28];
/// trace_model_step!("inference", layer_idx, input_shape);
/// ```
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
