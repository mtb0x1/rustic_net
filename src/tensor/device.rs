use crate::trace_fn;
use tracing::debug;

/// Compute device for tensor storage and operations.
///
/// Tensors can be allocated on different devices, with operations
/// automatically dispatched to the appropriate backend. The device
/// determines where tensor computations are performed.
///
/// # Examples
/// ```rust
/// use rustic_net::tensor::Device;
///
/// // Default CPU device
/// let device1 = Device::default();
///
/// // Specific CPU device (for multi-socket systems)
/// let device2 = Device::Cpu(Some(1));
///
/// // CUDA device (requires 'cuda' feature)
/// #[cfg(feature = "cuda")]
/// let device3 = Device::Cuda(0);
/// ```
///
/// # Thread Safety
/// All device variants are `Send` and `Sync`, allowing them to be shared across threads.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Device {
    /// CPU device with optional device ID
    ///
    /// - `None`: Default CPU device
    /// - `Some(n)`: Specific CPU device (useful for NUMA systems)
    Cpu(Option<usize>),

    /// CUDA device with device ID
    ///
    /// Requires the `cuda` feature. The ID corresponds to the CUDA device index.
    #[cfg(feature = "cuda")]
    Cuda(usize),

    /// WebGPU device with device ID
    ///
    /// Requires the `wasm` feature. The ID corresponds to the WebGPU adapter index.
    #[cfg(feature = "webgpu")]
    WebGpu(usize),
}

impl Default for Device {
    fn default() -> Self {
        trace_fn!("Device::default");
        debug!("Creating default CPU device with ID 0");
        // Default to CPU device with ID 0
        Device::Cpu(Some(0))
    }
}
