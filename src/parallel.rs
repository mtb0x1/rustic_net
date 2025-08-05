//! # Parallel Computation Utilities
//!
//! Provides thread pool management and parallel execution primitives for Rustic Net.
//! Leverages Rayon for efficient work-stealing parallelism with automatic load balancing.
//!
//! ## Features
//! - Automatic thread pool configuration with optimal defaults
//! - Environment variable overrides for fine-tuning
//! - Optimal default thread count (80% of available cores by default)
//! - Thread-safe, one-time initialization
//! - Work-stealing scheduler for load balancing
//! - Automatic chunk sizing for parallel iterators
//!
//! ## Usage
//! The thread pool is automatically initialized when needed, but you can also initialize it explicitly:
//! ```rust
//! use rustic_net::init_thread_pool;
//!
//! // Initialize with default settings (80% of available cores)
//! init_thread_pool();
//!
//! // Or set a custom thread count via environment variable
//! // RUSTIC_NET_NUM_THREADS=4 cargo run
//! ```
//!
//! ## Performance Considerations
//! - The default thread count (80% of cores) provides a good balance between
//!   parallelism and system responsiveness
//! - For I/O-bound workloads, consider increasing the thread count
//! - For CPU-bound workloads, the default is usually optimal
//! - Set `RUSTIC_NET_NUM_THREADS=1` to disable parallelism for debugging

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;
use std::thread::available_parallelism;
use tracing::{debug, warn};

/// Environment variable for overriding the default thread count
const THREAD_ENV_VAR: &str = "RUSTIC_NET_NUM_THREADS";

/// Global flag to track thread pool initialization state
static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Synchronization primitive for thread-safe one-time initialization
static INIT: Once = Once::new();

use {rayon, std::env};

/// Initializes the global thread pool with optimal settings.
///
/// This function configures the global Rayon thread pool with settings optimized
/// for typical machine learning workloads. It's automatically called when needed,
/// so explicit initialization is usually not required.
///
/// # Configuration
/// - Uses 80% of available CPU cores by default (minimum 1)
/// - Respects `RUSTIC_NET_NUM_THREADS` environment variable
/// - Thread-safe and idempotent (safe to call multiple times)
/// - Uses work-stealing scheduler for optimal load balancing
///
/// # Environment Variables
/// - `RUSTIC_NET_NUM_THREADS`: Override the number of threads to use
///   - If set to 0, uses the default (80% of cores)
///   - If set to a positive number, uses exactly that many threads
///   - If set to a value greater than the number of cores, logs a warning
///
/// # Panics
/// - If the global thread pool cannot be initialized
/// - If the system cannot determine the number of available CPU cores
///
/// # Example
/// ```no_run
/// # use rustic_net::init_thread_pool;
/// // Initialize with default settings
/// init_thread_pool();
///
/// // Or set a custom thread count via environment variable
/// std::env::set_var("RUSTIC_NET_NUM_THREADS", "4");
/// init_thread_pool();
/// ```
pub fn init_thread_pool() {
    INIT.call_once(|| {
        // Get the number of available CPU cores
        let num_cpus = available_parallelism().map(|n| n.get()).unwrap_or(1);

        // Calculate default thread count (80% of cores, minimum 1)
        let default_threads = (num_cpus as f32 * 0.8).floor().max(1.0) as usize;

        // Check for environment variable override
        let num_threads = match env::var(THREAD_ENV_VAR) {
            Ok(val) => match val.parse::<usize>() {
                Ok(n) if n > 0 => {
                    if n > num_cpus {
                        warn!(
                            "Using {} threads (set via {}), which is more than available CPU cores ({}).",
                            n, THREAD_ENV_VAR, num_cpus
                        );
                    }
                    n
                }
                _ => {
                    warn!(
                        "Invalid value for {}. Using default of {} threads.",
                        THREAD_ENV_VAR, default_threads
                    );
                    default_threads
                }
            },
            Err(_) => default_threads,
        };

        // Initialize the global thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .expect("Failed to initialize global thread pool");

        debug!(
            "Initialized Rayon thread pool with {} threads.",
            num_threads
        );

        INITIALIZED.store(true, Ordering::SeqCst);
    });
}

/// Returns the current thread pool size.
///
/// This function returns the number of threads currently configured in the global
/// thread pool. If the thread pool hasn't been initialized yet, it will be
/// initialized with default settings first.
///
/// # Returns
/// The number of threads in the global thread pool.
///
/// # Example
/// ```
/// # use rustic_net::current_num_threads;
/// let num_threads = current_num_threads();
/// assert!(num_threads > 0);
/// ```
pub fn current_num_threads() -> usize {
    init_thread_pool();
    rayon::current_num_threads()
}

/// Calculates optimal chunk size for parallel operations.
///
/// This helper function determines an efficient chunk size for parallel iteration
/// based on the length of the data and the number of available threads. It aims to:
/// - Distribute work evenly across threads
/// - Minimize scheduling overhead
/// - Maximize cache efficiency
/// - Avoid excessive task spawning
///
/// # Arguments
/// * `len` - The total number of items to process
///
/// # Returns
/// The recommended number of items to process in each parallel task.
///
/// # Example
/// ```
/// # use rustic_net::recommended_chunk_size;
/// let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let chunk_size = recommended_chunk_size(data.len());
/// assert!(chunk_size > 0);
///
/// // Use with Rayon's parallel iterators
/// use rayon::prelude::*;
/// let sum: i32 = data.par_chunks(chunk_size)
///     .map(|chunk| chunk.iter().sum::<i32>())
///     .sum();
/// assert_eq!(sum, 55);
/// ```
pub fn recommended_chunk_size(len: usize) -> usize {
    let num_threads = current_num_threads();
    if num_threads == 0 {
        return len; // Avoid division by zero if something goes wrong
    }
    len.div_ceil(num_threads) // Equivalent to ceiling division
}
