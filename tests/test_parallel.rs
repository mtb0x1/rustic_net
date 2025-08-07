#[cfg(feature = "parallel")]
use rustic_net::{current_num_threads, init_thread_pool, recommended_chunk_size};

#[cfg(feature = "parallel")]
#[test]
fn test_thread_pool_initialization_is_safe() {
    // Calling init multiple times should be safe and not panic.
    init_thread_pool();
    let first_call_threads = current_num_threads();
    assert!(first_call_threads >= 1);

    init_thread_pool();
    let second_call_threads = current_num_threads();
    assert_eq!(first_call_threads, second_call_threads);
}

#[cfg(feature = "parallel")]
#[test]
fn test_recommended_chunk_size() {
    init_thread_pool();
    let num_threads = current_num_threads();
    assert_eq!(
        recommended_chunk_size(100),
        (100 + num_threads - 1) / num_threads
    );
    assert_eq!(
        recommended_chunk_size(101),
        (101 + num_threads - 1) / num_threads
    );
    assert_eq!(recommended_chunk_size(1), 1);
}
