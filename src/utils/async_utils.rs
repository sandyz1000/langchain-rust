//! Async Operation Optimization Tools
//!
//! Provides async optimization features such as parallelization and batch processing.

use std::future::Future;

/// Execute multiple Futures in parallel and return all results
///
/// # Parameters
/// - `futures`: List of Futures
///
/// # Returns
/// Vector of results from all Futures
///
/// # Example
/// ```rust,no_run
/// use langchain_ai_rust::utils::join_all;
///
/// # async fn example() {
/// let futures = vec![
///     async { 1 },
///     async { 2 },
///     async { 3 },
/// ];
/// let results = join_all(futures).await;
/// assert_eq!(results, vec![1, 2, 3]);
/// # }
/// ```
pub async fn join_all<T, F>(futures: Vec<F>) -> Vec<T>
where
    F: Future<Output = T> + Send,
    T: Send,
{
    futures::future::join_all(futures).await
}

/// Execute multiple Futures in parallel and return the first successful result or all errors
///
/// Similar to `futures::future::try_join_all`, but provides clearer error handling.
pub async fn try_join_all<T, E, F>(futures: Vec<F>) -> Result<Vec<T>, E>
where
    F: Future<Output = Result<T, E>> + Send,
    T: Send,
    E: Send,
{
    futures::future::try_join_all(futures).await
}

/// Batch process data with parallel execution
///
/// # Parameters
/// - `items`: Data items to process
/// - `batch_size`: Number of items per batch
/// - `processor`: Processing function
///
/// # Returns
/// Vector of all processing results
///
/// # Example
/// ```rust,no_run
/// use langchain_ai_rust::utils::batch_process;
///
/// # async fn example() {
/// let items = vec![1, 2, 3, 4, 5];
/// let results = batch_process(items, 2, |item| async move {
///     item * 2
/// }).await;
/// assert_eq!(results, vec![2, 4, 6, 8, 10]);
/// # }
/// ```
pub async fn batch_process<T, R, F, Fut>(items: Vec<T>, batch_size: usize, processor: F) -> Vec<R>
where
    T: Send + Sync + Clone,
    R: Send,
    F: Fn(T) -> Fut + Send + Sync,
    Fut: Future<Output = R> + Send,
{
    let mut results = Vec::with_capacity(items.len());

    for chunk in items.chunks(batch_size) {
        let futures: Vec<_> = chunk.iter().map(|item| processor(item.clone())).collect();
        let chunk_results = join_all(futures).await;
        results.extend(chunk_results);
    }

    results
}

/// Batch process data with parallel execution (with error handling)
pub async fn batch_process_result<T, R, E, F, Fut>(
    items: Vec<T>,
    batch_size: usize,
    processor: F,
) -> Result<Vec<R>, E>
where
    T: Send + Sync + Clone,
    R: Send,
    E: Send,
    F: Fn(T) -> Fut + Send + Sync,
    Fut: Future<Output = Result<R, E>> + Send,
{
    let mut results = Vec::with_capacity(items.len());

    for chunk in items.chunks(batch_size) {
        let futures: Vec<_> = chunk.iter().map(|item| processor(item.clone())).collect();
        let chunk_results = try_join_all(futures).await?;
        results.extend(chunk_results);
    }

    Ok(results)
}

/// Execute multiple Futures in parallel using tokio::spawn
///
/// Suitable for CPU-intensive or long-running tasks.
pub async fn spawn_all<T, F>(futures: Vec<F>) -> Vec<Result<T, tokio::task::JoinError>>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let handles: Vec<_> = futures.into_iter().map(|f| tokio::spawn(f)).collect();
    join_all(handles).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;

    #[tokio::test]
    async fn test_join_all() {
        type PinBoxi32 = Pin<Box<dyn Future<Output = i32> + Send>>;

        let futures: Vec<PinBoxi32> = vec![
            Box::pin(async { 1 }),
            Box::pin(async { 2 }),
            Box::pin(async { 3 }),
        ];
        let results = join_all(futures).await;
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn test_try_join_all() {
        type PinBoxResultI32 = Pin<Box<dyn Future<Output = Result<i32, String>> + Send>>;

        let futures: Vec<PinBoxResultI32> = vec![
            Box::pin(async { Ok::<i32, String>(1) }),
            Box::pin(async { Ok(2) }),
            Box::pin(async { Ok(3) }),
        ];
        let results = try_join_all(futures).await.unwrap();
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn test_batch_process() {
        let items = vec![1, 2, 3, 4, 5];
        let results = batch_process(items, 2, |item| async move { item * 2 }).await;
        assert_eq!(results, vec![2, 4, 6, 8, 10]);
    }

    #[tokio::test]
    async fn test_batch_process_result() {
        let items = vec![1, 2, 3];
        let results =
            batch_process_result(items, 2, |item| async move { Ok::<i32, &str>(item * 2) })
                .await
                .unwrap();
        assert_eq!(results, vec![2, 4, 6]);
    }
}
