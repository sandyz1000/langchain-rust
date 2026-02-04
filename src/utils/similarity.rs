//! Similarity calculation helper functions
//! Provides unified similarity calculation functions supporting multiple precisions and batch operations.

/// Calculates the cosine similarity between two vectors (f64 precision)
///
/// # Parameters
/// - `vec1`: The first vector
/// - `vec2`: The second vector
///
/// # Returns
/// Cosine similarity value, ranging from [-1, 1]
///
/// # Example
/// ```rust
/// use langchain_ai_rust::utils::cosine_similarity_f64;
///
/// let vec1 = vec![1.0, 2.0, 3.0];
/// let vec2 = vec![1.0, 2.0, 3.0];
/// let similarity = cosine_similarity_f64(&vec1, &vec2);
/// assert!((similarity - 1.0).abs() < 1e-10);
/// ```
pub fn cosine_similarity_f64(vec1: &[f64], vec2: &[f64]) -> f64 {
    if vec1.len() != vec2.len() {
        return 0.0;
    }

    let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
    let magnitude_vec1: f64 = vec1.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    let magnitude_vec2: f64 = vec2.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

    if magnitude_vec1 == 0.0 || magnitude_vec2 == 0.0 {
        return 0.0;
    }

    dot_product / (magnitude_vec1 * magnitude_vec2)
}

/// Calculates the cosine similarity between two vectors (f32 precision)
///
/// # Parameters
/// - `vec1`: The first vector
/// - `vec2`: The second vector
///
/// # Returns
/// Cosine similarity value, ranging from [-1, 1] (returns f64 to maintain precision)
///
/// # Example
/// ```rust
/// use langchain_ai_rust::utils::cosine_similarity_f32;
///
/// let vec1 = vec![1.0f32, 2.0, 3.0];
/// let vec2 = vec![1.0f32, 2.0, 3.0];
/// let similarity = cosine_similarity_f32(&vec1, &vec2);
/// assert!((similarity - 1.0).abs() < 1e-6);
/// ```
pub fn cosine_similarity_f32(vec1: &[f32], vec2: &[f32]) -> f64 {
    if vec1.len() != vec2.len() {
        return 0.0;
    }

    let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    (dot_product / (norm_a * norm_b)) as f64
}

/// Batch calculates cosine similarity
///
/// Calculates the similarity between a query vector and multiple target vectors.
///
/// # Parameters
/// - `query`: The query vector
/// - `targets`: A list of target vectors
///
/// # Returns
/// A list of similarity values for each target vector with the query vector
///
/// # Example
/// ```rust
/// use langchain_ai_rust::utils::batch_cosine_similarity_f64;
///
/// let query = vec![1.0, 0.0];
/// let targets = vec![
///     vec![1.0, 0.0],
///     vec![0.0, 1.0],
///     vec![1.0, 1.0],
/// ];
/// let similarities = batch_cosine_similarity_f64(&query, &targets);
/// assert_eq!(similarities.len(), 3);
/// ```
pub fn batch_cosine_similarity_f64(query: &[f64], targets: &[Vec<f64>]) -> Vec<f64> {
    targets
        .iter()
        .map(|target| cosine_similarity_f64(query, target))
        .collect()
}

/// Batch calculates cosine similarity (f32 precision)
pub fn batch_cosine_similarity_f32(query: &[f32], targets: &[Vec<f32>]) -> Vec<f64> {
    targets
        .iter()
        .map(|target| cosine_similarity_f32(query, target))
        .collect()
}

/// Calculates text similarity (based on simple word overlap method)
///
/// Uses Jaccard similarity to calculate the similarity between two texts.
///
/// # Parameters
/// - `text1`: The first text
/// - `text2`: The second text
///
/// # Returns
/// Similarity value, ranging from [0, 1]
///
/// # Example
/// ```rust
/// use langchain_ai_rust::utils::text_similarity;
///
/// let text1 = "hello world";
/// let text2 = "world hello";
/// let similarity = text_similarity(text1, text2);
/// assert!((similarity - 1.0).abs() < 1e-10);
/// ```
pub fn text_similarity(text1: &str, text2: &str) -> f64 {
    let words1: std::collections::HashSet<String> =
        text1.split_whitespace().map(|w| w.to_lowercase()).collect();
    let words2: std::collections::HashSet<String> =
        text2.split_whitespace().map(|w| w.to_lowercase()).collect();

    let intersection: usize = words1.intersection(&words2).count();
    let union: usize = words1.union(&words2).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_f64_identical() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        let similarity = cosine_similarity_f64(&vec1, &vec2);
        assert!((similarity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_f64_orthogonal() {
        let vec1 = vec![1.0, 0.0];
        let vec2 = vec![0.0, 1.0];
        let similarity = cosine_similarity_f64(&vec1, &vec2);
        assert!((similarity - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_f64_different_lengths() {
        let vec1 = vec![1.0, 2.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        let similarity = cosine_similarity_f64(&vec1, &vec2);
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_cosine_similarity_f32() {
        let vec1 = vec![1.0f32, 2.0, 3.0];
        let vec2 = vec![1.0f32, 2.0, 3.0];
        let similarity = cosine_similarity_f32(&vec1, &vec2);
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_cosine_similarity_f64() {
        let query = vec![1.0, 0.0];
        let targets = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let similarities = batch_cosine_similarity_f64(&query, &targets);
        assert_eq!(similarities.len(), 2);
        assert!((similarities[0] - 1.0).abs() < 1e-10);
        assert!((similarities[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_text_similarity() {
        let text1 = "hello world";
        let text2 = "world hello";
        let similarity = text_similarity(text1, text2);
        assert!((similarity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_text_similarity_different() {
        let text1 = "hello world";
        let text2 = "goodbye universe";
        let similarity = text_similarity(text1, text2);
        assert!(similarity < 0.5);
    }
}
