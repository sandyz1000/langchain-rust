//! Vector operation helper functions
//! Provides common operations such as averaging and summing vectors.

/// Calculates the average of multiple vectors
///
/// # Parameters
/// - `embeddings`: A list of vectors
///
/// # Returns
/// The average vector
///
/// # Example
/// ```rust
/// use langchain_ai_rust::utils::mean_embedding_f64;
///
/// let embeddings = vec![
///     vec![1.0, 2.0],
///     vec![3.0, 4.0],
/// ];
/// let mean = mean_embedding_f64(&embeddings);
/// assert_eq!(mean, vec![2.0, 3.0]);
/// ```
pub fn mean_embedding_f64(embeddings: &[Vec<f64>]) -> Vec<f64> {
    if embeddings.is_empty() {
        return Vec::new();
    }

    embeddings
        .iter()
        .fold(
            vec![0f64; embeddings[0].len()],
            |mut accumulator, embedding_vec| {
                for (i, &value) in embedding_vec.iter().enumerate() {
                    accumulator[i] += value;
                }
                accumulator
            },
        )
        .iter()
        .map(|&sum| sum / embeddings.len() as f64)
        .collect()
}

/// Calculates the sum of multiple vectors
///
/// # Parameters
/// - `vectors`: A list of vectors
///
/// # Returns
/// The sum of vectors
///
/// # Example
/// ```rust
/// use langchain_ai_rust::utils::sum_vectors_f64;
///
/// let vectors = vec![
///     vec![1.0, 2.0],
///     vec![3.0, 4.0],
/// ];
/// let sum = sum_vectors_f64(&vectors);
/// assert_eq!(sum, vec![4.0, 6.0]);
/// ```
pub fn sum_vectors_f64(vectors: &[Vec<f64>]) -> Vec<f64> {
    if vectors.is_empty() {
        return Vec::new();
    }

    let mut sum_vec = vec![0.0; vectors[0].len()];
    for vec in vectors {
        for (i, &value) in vec.iter().enumerate() {
            sum_vec[i] += value;
        }
    }
    sum_vec
}

/// Calculates the average of multiple vectors (f32 precision)
pub fn mean_embedding_f32(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }

    embeddings
        .iter()
        .fold(
            vec![0f32; embeddings[0].len()],
            |mut accumulator, embedding_vec| {
                for (i, &value) in embedding_vec.iter().enumerate() {
                    accumulator[i] += value;
                }
                accumulator
            },
        )
        .iter()
        .map(|&sum| sum / embeddings.len() as f32)
        .collect()
}

/// Calculates the sum of multiple vectors (f32 precision)
pub fn sum_vectors_f32(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }

    let mut sum_vec = vec![0.0f32; vectors[0].len()];
    for vec in vectors {
        for (i, &value) in vec.iter().enumerate() {
            sum_vec[i] += value;
        }
    }
    sum_vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_embedding_f64() {
        let embeddings = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let mean = mean_embedding_f64(&embeddings);
        assert_eq!(mean, vec![2.0, 3.0]);
    }

    #[test]
    fn test_mean_embedding_f64_empty() {
        let embeddings: Vec<Vec<f64>> = vec![];
        let mean = mean_embedding_f64(&embeddings);
        assert!(mean.is_empty());
    }

    #[test]
    fn test_sum_vectors_f64() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let sum = sum_vectors_f64(&vectors);
        assert_eq!(sum, vec![4.0, 6.0]);
    }

    #[test]
    fn test_sum_vectors_f64_empty() {
        let vectors: Vec<Vec<f64>> = vec![];
        let sum = sum_vectors_f64(&vectors);
        assert!(sum.is_empty());
    }
}
