use ndarray::Array1;
use crate::Float;
use std::fmt::Debug;

/// Represents a single data point with features and a label.
///
/// L: The type of the label (e.g., String, i32, enum).
/// F: The float type for the features (e.g., f32, f64).
#[derive(Debug, Clone)]
pub struct DataPoint<L, F>
where
    L: Clone + Eq + std::hash::Hash + Debug,
    F: Float,
{
    pub features: Array1<F>,
    pub label: L,
}

impl<L, F> DataPoint<L, F>
where
    L: Clone + Eq + std::hash::Hash + Debug,
    F: Float,
{
    pub fn new(features: Array1<F>, label: L) -> Self {
        DataPoint { features, label }
    }
}
