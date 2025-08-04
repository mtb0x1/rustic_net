use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

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

#[macro_export]
macro_rules! trace_operation {
    ($op_name:expr, $result:expr) => {{
        tracing::debug!("Operation: {} starting", $op_name);
        let result = $result;
        tracing::debug!("Operation: {} completed", $op_name);
        result
    }};
}

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
