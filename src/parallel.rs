//! Parallel computation utilities for Rustic Net
//!
//! This module provides configuration and utilities for parallel execution
//! using the Rayon thread pool.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;
use std::thread::available_parallelism;
use tracing::{debug, warn};

// Environment variable for overriding the thread count
const THREAD_ENV_VAR: &str = "RUSTIC_NET_NUM_THREADS";

// Global flag to track if we've already initialized the thread pool
static INITIALIZED: AtomicBool = AtomicBool::new(false);

static INIT: Once = Once::new();

use {rayon, std::env};

/// Initializes the global thread pool with the optimal number of threads.
///
/// By default, uses 80% of the available CPU cores (minimum 1).
/// Can be overridden by setting the `RUSTIC_NET_NUM_THREADS` environment variable.
///
/// # Panics
/// Panics if the thread pool has already been initialized.
///
/// # Note
/// This is a no-op when the `parallel` feature is not enabled.
pub fn init_thread_pool() {
    // Only initialize once
    if INITIALIZED.load(Ordering::SeqCst) {
        return;
    }

    INIT.call_once(|| {
        // Get the number of available CPU cores kinda
        // TODO: Check docs
        let num_cpus = available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        // Calculate default thread count (80% of cores, minimum 1)
        let default_threads = (num_cpus as f32 * 0.8).floor().max(1.0) as usize;
        // Check for environment variable override
        let num_threads = match env::var(THREAD_ENV_VAR) {
            Ok(val) => {
                match val.parse::<usize>() {
                    Ok(n) if n > 0 => {
                        // Warn if user is setting a high thread count that might starve the system
                        if n > num_cpus {
                            warn!(
                                "Using {} threads (set via {}), which is more than available CPU cores ({}).
                                This might lead to system starvation and degraded performance.",
                                n, THREAD_ENV_VAR, num_cpus
                            );
                        } else if n > default_threads {
                            warn!(
                                "Using {} threads (set via {}), which is more than the recommended {} threads.
                                This might lead to system starvation.",
                                n, THREAD_ENV_VAR, default_threads
                            );
                        }
                        n
                    },
                    Ok(_) => {
                        warn!(
                            "Invalid value for {}: must be greater than 0. Using default of {} threads.",
                            THREAD_ENV_VAR, default_threads
                        );
                        default_threads
                    },
                    Err(_) => {
                        warn!(
                            "Could not parse {} value. Using default of {} threads.",
                            THREAD_ENV_VAR, default_threads
                        );
                        default_threads
                    },
                }
            }
            Err(_) => default_threads,
        };

        // Initialize the global thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .expect("Failed to initialize global thread pool");

        debug!(
            "Initialized Rayon thread pool with {} threads ({} CPUs available, {} used, {:.0}% utilization)",
            num_threads,
            num_cpus,
            num_threads,
            (num_threads as f32 / num_cpus as f32) * 100.0
        );

        // Mark as initialized
        INITIALIZED.store(true, Ordering::SeqCst);
    });

    #[cfg(not(feature = "parallel"))]
    INITIALIZED.store(true, Ordering::SeqCst);
}

/// Returns the current number of threads in the global thread pool.
/// If the thread pool hasn't been initialized yet, it will be initialized with default settings.
///
/// # Panics
/// Panics if the thread pool cannot be initialized.
///
/// # Note
/// Returns 1 when the `parallel` feature is not enabled.
pub fn current_num_threads() -> usize {
    if !INITIALIZED.load(Ordering::SeqCst) {
        init_thread_pool();
    }

    rayon::current_num_threads()
}

/// Returns the recommended chunk size for parallel operations
/// based on the number of threads available.
pub fn recommended_chunk_size(len: usize) -> usize {
    let num_threads = current_num_threads();
    len.div_ceil(num_threads)
}

#[allow(unused_imports)]
#[cfg(test)]
mod tests {
    // TODO: Remove this once lazy_static is no longer needed
    use super::*;
    use lazy_static::lazy_static;
    use std::sync::Mutex;

    // Ensure tests run sequentially to avoid interference
    lazy_static! {
        static ref TEST_MUTEX: Mutex<()> = Mutex::new(());
    }

    #[test]
    fn test_thread_pool_initialization() {
        let _lock = TEST_MUTEX.lock().unwrap();

        // Reset the initialization state for testing
        unsafe {
            let init = &INITIALIZED as *const _ as *mut AtomicBool;
            (*init).store(false, Ordering::SeqCst);
        }

        // Test default initialization
        init_thread_pool();
        assert_eq!(INITIALIZED.load(Ordering::SeqCst), false);

        // Should not panic on second call
        init_thread_pool();
    }

    #[test]
    fn test_recommended_chunk_size() {
        let _lock = TEST_MUTEX.lock().unwrap();

        // Reset the initialization state for testing
        unsafe {
            let init = &INITIALIZED as *const _ as *mut AtomicBool;
            (*init).store(false, Ordering::SeqCst);
        }

        // Test with known number of threads
        std::env::set_var(THREAD_ENV_VAR, "4");
        init_thread_pool();

        assert_eq!(recommended_chunk_size(100), 25); // 100/4 = 25
        assert_eq!(recommended_chunk_size(101), 26); // 101/4 = 25.25 -> 26
        assert_eq!(recommended_chunk_size(1), 1); // Edge case: small input

        // Clean up
        std::env::remove_var(THREAD_ENV_VAR);
    }
}
