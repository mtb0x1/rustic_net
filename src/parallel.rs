//! # Parallel Computation Utilities
//!
//! Provides thread pool management and parallel execution primitives for Rustic Net.
//! Leverages Rayon for efficient work-stealing parallelism with automatic load balancing.
//!
//! ## Features
//! - Automatic thread pool configuration
//! - Environment variable overrides
//! - Optimal default thread count (80% of available cores)
//! - Thread-safe initialization

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
/// Configuration:
/// - Uses 80% of available CPU cores by default (minimum 1)
/// - Respects `RUSTIC_NET_NUM_THREADS` environment variable
/// - Thread-safe and idempotent (safe to call multiple times)
///
/// # Panics
/// If the global thread pool cannot be initialized.
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
/// Initializes the thread pool with default settings if not already initialized.
pub fn current_num_threads() -> usize {
    init_thread_pool();
    rayon::current_num_threads()
}

/// Calculates optimal chunk size for parallel operations.
///
/// Distributes work evenly across available threads to minimize
/// scheduling overhead and maximize cache efficiency.
pub fn recommended_chunk_size(len: usize) -> usize {
    let num_threads = current_num_threads();
    if num_threads == 0 {
        return len; // Avoid division by zero if something goes wrong
    }
    len.div_ceil(num_threads) // Equivalent to ceiling division
}
