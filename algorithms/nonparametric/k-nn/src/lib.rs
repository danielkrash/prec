use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
// These are the core components from our shared library.
use prec_helpers::{DataPoint, Distance};

// ndarray and prec_helpers are used in the public function signatures.
use ndarray::ArrayView1;
use prec_helpers::Float;
/// Errors that can occur when using the k-NN classifier.
#[derive(Debug, Clone, PartialEq)]
pub enum KnnError {
    /// k cannot be zero for a k-NN classifier
    InvalidK,
    /// Cannot predict with an empty training set
    EmptyTrainingSet,
    /// Invalid distance comparison (likely due to NaN values in data)
    InvalidDistance,
    /// Could not determine a majority class among neighbors
    NoMajorityClass,
}

impl Display for KnnError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            KnnError::InvalidK => write!(f, "k cannot be zero for a k-NN classifier"),
            KnnError::EmptyTrainingSet => write!(f, "Cannot predict with an empty training set"),
            KnnError::InvalidDistance => write!(
                f,
                "Invalid distance comparison (likely due to NaN values in data)"
            ),
            KnnError::NoMajorityClass => {
                write!(f, "Could not determine a majority class among neighbors")
            }
        }
    }
}

impl Error for KnnError {}

/// A k-Nearest Neighbors (k-NN) classifier.
///
/// This classifier predicts the label of a new data point by finding the `k`
/// most similar points in its training set and taking a majority vote among their labels.
///
/// # Type Parameters
///
/// * `L`: The type of the label (e.g., `String`, `i32`, or a custom `enum`).
/// * `F`: The float type for the features (e.g., `f32`, `f64`).
/// * `D`: The distance metric, which must implement the `ml_core::Distance` trait.
#[derive(Debug, Clone)]
pub struct KnnClassifier<L, F, D>
where
    L: Clone + Eq + std::hash::Hash + Debug,
    F: Float,
    D: Distance<F>,
{
    k: usize,
    training_data: Vec<DataPoint<L, F>>,
    distance: D,
}

impl<L, F, D> KnnClassifier<L, F, D>
where
    L: Clone + Eq + std::hash::Hash + Debug,
    F: Float,
    D: Distance<F>,
{
    /// Creates a new k-NN classifier.
    ///
    /// # Arguments
    ///
    /// * `k`: The number of neighbors to consider for classification. Must be greater than 0.
    /// * `training_data`: A vector of `DataPoint`s that the classifier will learn from.
    /// * `distance`: An instance of a struct that implements the `Distance` trait (e.g., `L2Dist`).
    ///
    /// # Errors
    ///
    /// Returns `KnnError::InvalidK` if `k` is 0, as this is not a valid configuration.
    pub fn new(
        k: usize,
        training_data: Vec<DataPoint<L, F>>,
        distance: D,
    ) -> Result<Self, KnnError> {
        if k == 0 {
            return Err(KnnError::InvalidK);
        }
        Ok(Self {
            k,
            training_data,
            distance,
        })
    }

    /// Predicts the label for a new, unseen data point.
    ///
    /// # Arguments
    ///
    /// * `features`: An `ArrayView1` containing the features of the point to classify.
    ///
    /// # Returns
    ///
    /// Returns the predicted label `L` on success.
    ///
    /// # Errors
    ///
    /// Returns `KnnError::EmptyTrainingSet` if the training data is empty.
    /// Returns `KnnError::InvalidDistance` if distance comparison fails (e.g., due to NaN values).
    /// Returns `KnnError::NoMajorityClass` if no majority class can be determined.
    pub fn predict(&self, features: ArrayView1<F>) -> Result<L, KnnError> {
        if self.training_data.is_empty() {
            return Err(KnnError::EmptyTrainingSet);
        }

        // 1. Calculate the "relative distance" (e.g., squared Euclidean) from the new point
        //    to every point in the training set. This is faster than the true distance.
        let mut distances: Vec<(F, &L)> = self
            .training_data
            .iter()
            .map(|dp| {
                let dist = self.distance.rdistance(dp.features.view(), features);
                (dist, &dp.label)
            })
            .collect();

        // 2. Sort the training points by their distance to the new point (ascending).
        //    We use `sort_unstable_by` as it's typically faster and the order of
        //    elements with equal distance doesn't matter.
        //    `.partial_cmp` is used because floats don't have a total ordering (due to NaN).
        distances.sort_unstable_by(|a, b| {
            { a.0.partial_cmp(&b.0).ok_or(KnnError::InvalidDistance) }
                .unwrap_or_else(|_| std::cmp::Ordering::Equal)
        });

        // 3. Take the top `k` neighbors.
        //    We use `min` to handle cases where k is larger than the training set size.
        let num_neighbors = self.k.min(distances.len());
        let neighbors = &distances[..num_neighbors];

        // 4. Count the votes for each label among the neighbors.
        let mut votes: HashMap<&L, usize> = HashMap::new();
        for (_, label) in neighbors {
            *votes.entry(label).or_insert(0) += 1;
        }

        // 1. Find the highest vote count.
        // .values() gets an iterator over the counts (the `usize` values).
        // .max() finds the largest count.
        // We expect this to be Some, since we checked that neighbors is not empty.
        let max_votes = *votes.values().max().unwrap();
    
        // 2. Collect all labels that have the maximum vote count.
        let winners: Vec<_> = votes
            .iter()
            .filter(|&(_, &count)| count == max_votes)
            .map(|(&label, _)| label)
            .collect();
    
        // 3. Check if there is a tie.
        if winners.len() == 1 {
            // Exactly one winner, no tie.
            Ok(winners[0].clone())
        } else {
            // More than one winner means there was a tie.
            Err(KnnError::NoMajorityClass)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use prec_helpers::L2Dist; // Assuming L2Dist is exported from ml_core

    #[test]
    fn test_knn_classification_simple() {
        // Define training data
        let training_data = vec![
            DataPoint::new(array![1.0, 1.0], "A"),
            DataPoint::new(array![2.0, 2.0], "A"),
            DataPoint::new(array![1.0, 2.0], "A"),
            DataPoint::new(array![8.0, 8.0], "B"),
            DataPoint::new(array![9.0, 8.0], "B"),
            DataPoint::new(array![8.0, 9.0], "B"),
        ];

        // Create a classifier with k=3 and L2 (Euclidean) distance
        let classifier = KnnClassifier::new(3, training_data, L2Dist).unwrap();

        // Point close to class "A"
        let point_a = array![2.5, 2.5];
        assert_eq!(classifier.predict(point_a.view()).unwrap(), "A");

        // Point close to class "B"
        let point_b = array![7.5, 8.5];
        assert_eq!(classifier.predict(point_b.view()).unwrap(), "B");
    }

    #[test]
    fn test_knn_k_larger_than_dataset() {
        let training_data = vec![
            DataPoint::new(array![1.0], "A"),
            DataPoint::new(array![2.0], "A"),
            DataPoint::new(array![10.0], "B"),
        ];

        // k=5 is larger than the dataset size of 3, but this should work fine.
        // The two 'A's will outvote the one 'B'.
        let classifier = KnnClassifier::new(5, training_data, L2Dist).unwrap();
        let point = array![3.0];
        assert_eq!(classifier.predict(point.view()).unwrap(), "A");
    }

    #[test]
    fn test_error_on_k_zero() {
        let training_data: Vec<DataPoint<&str, f64>> = vec![];
        let result = KnnClassifier::new(0, training_data, L2Dist);
        assert!(matches!(result, Err(KnnError::InvalidK)));
    }

    #[test]
    fn test_error_on_empty_training_set() {
        let training_data: Vec<DataPoint<&str, f64>> = vec![];
        let classifier = KnnClassifier::new(3, training_data, L2Dist).unwrap();
        let result = classifier.predict(array![1.0, 1.0].view());
        assert!(matches!(result, Err(KnnError::EmptyTrainingSet)));
    }
}
