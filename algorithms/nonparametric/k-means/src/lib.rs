use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
// Core components from shared library
use prec::{DataPoint, Distance, Float};

/// Errors that can occur during k-means clustering.
#[derive(Debug, Clone, PartialEq)]
pub enum KMeansError {
    InvalidK,
    EmptyDataSet,
    KTooLarge,
    MismatchedDimensions,
    InvalidDistance,
    NotFitted,
}

impl Display for KMeansError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for KMeansError {}

/// A k-means clustering model.
#[derive(Debug, Clone)]
pub struct KMeans<F, D>
where
    F: Float,
    D: Distance<F>,
{
    pub k: usize,
    max_iter: u32,
    tolerance: F,
    distance: D,
    centroids: Option<Array2<F>>,
}

impl<F, D> KMeans<F, D>
where
    F: Float,
    D: Distance<F>,
{
    pub fn new(k: usize, max_iter: u32, tolerance: F, distance: D) -> Self {
        Self { k, max_iter, tolerance, distance, centroids: None }
    }

    pub fn fit<L: Debug + Eq + Hash + Clone>(
        &mut self,
        data: &[DataPoint<L, F>],
    ) -> Result<(Vec<usize>, Array2<F>), KMeansError> {
        if self.k == 0 { return Err(KMeansError::InvalidK); }
        if data.is_empty() { return Err(KMeansError::EmptyDataSet); }
        if self.k > data.len() { return Err(KMeansError::KTooLarge); }

        let n_features = data[0].features.len();
        for dp in data { if dp.features.len() != n_features { return Err(KMeansError::MismatchedDimensions); }}

        // K-Means++ initialization: choose centroids probabilistically to be far apart
        let mut centroids = Array2::zeros((self.k, n_features));
        self.kmeans_plus_plus_init(data, &mut centroids);

        let mut assignments = vec![0; data.len()];
        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();

        for iter in 0..self.max_iter {
            println!("Iteration {}", iter + 1);
            // Assignment step
            let mut changes = 0;
            for (i, dp) in data.iter().enumerate() {
                let old = assignments[i];
                let mut best_idx = 0;
                let mut best_dist = F::infinity();
                for (c, centroid) in centroids.rows().into_iter().enumerate() {
                    let d = self.distance.rdistance(dp.features.view(), centroid);
                    if d.is_nan() { return Err(KMeansError::InvalidDistance); }
                    if d < best_dist { best_dist = d; best_idx = c; }
                }
                assignments[i] = best_idx;
                if old != best_idx { changes += 1; }
            }
            println!("  {} points changed", changes);
            if changes == 0 {
                println!("Converged (no assignment changes)");
                break;
            }

            // Update step
            clusters.clear();
            for (i, &a) in assignments.iter().enumerate() {
                clusters.entry(a).or_default().push(i);
            }

            let mut new_centroids = Array2::zeros((self.k, n_features));
            for i in 0..self.k {
                if let Some(points) = clusters.get(&i) {
                    if !points.is_empty() {
                        let mut sum = Array1::zeros(n_features);
                        for &pi in points { sum += &data[pi].features; }
                        sum /= F::from(points.len()).unwrap();
                        new_centroids.row_mut(i).assign(&sum);
                        continue;
                    }
                }
                // empty cluster: keep old
                new_centroids.row_mut(i).assign(&centroids.row(i));
            }

            // Check convergence by max shift
            let mut max_shift = F::zero();
            for (old, new) in centroids.rows().into_iter().zip(new_centroids.rows()) {
                let shift = self.distance.distance(old, new);
                if shift.is_nan() { return Err(KMeansError::InvalidDistance); }
                if shift > max_shift { max_shift = shift; }
            }
            println!("  max shift = {:?}, tol = {:?}", max_shift, self.tolerance);

            centroids = new_centroids;
            if max_shift < self.tolerance {
                println!("Converged (centroid shift < tol)");
                break;
            }
        }

        self.centroids = Some(centroids.clone());
        Ok((assignments, centroids))
    }

    pub fn predict(&self, point: ArrayView1<F>) -> Result<usize, KMeansError> {
        let centroids = self.centroids.as_ref().ok_or(KMeansError::NotFitted)?;
        let mut best = F::infinity();
        let mut idx = 0;
        for (i, c) in centroids.rows().into_iter().enumerate() {
            let d = self.distance.rdistance(point, c);
            if d.is_nan() { return Err(KMeansError::InvalidDistance); }
            if d < best { best = d; idx = i; }
        }
        Ok(idx)
    }

    pub fn centroids(&self) -> Result<ndarray::ArrayView2<F>, KMeansError> {
        self.centroids
            .as_ref()
            .map(|c| c.view())
            .ok_or(KMeansError::NotFitted)
    }

    /// K-Means++ initialization for better clustering results
    /// Chooses centroids probabilistically to be far apart from each other
    fn kmeans_plus_plus_init<L: Debug + Eq + Hash + Clone>(
        &self,
        data: &[DataPoint<L, F>],
        centroids: &mut Array2<F>,
    ) {
        let mut rng = rand::rng();
        
        // Step 1: Choose the first centroid uniformly at random
        let first_idx = rng.random_range(0..data.len());
        centroids.row_mut(0).assign(&data[first_idx].features);
        
        // Step 2: For each remaining centroid
        for k in 1..self.k {
            // Calculate squared distances from each point to its nearest centroid
            let mut distances: Vec<F> = Vec::with_capacity(data.len());
            let mut total_weight = F::zero();
            
            for dp in data {
                let mut min_dist_sq = F::infinity();
                
                // Find distance to nearest existing centroid
                for j in 0..k {
                    let centroid = centroids.row(j);
                    let dist = self.distance.distance(dp.features.view(), centroid);
                    let dist_sq = dist * dist;
                    if dist_sq < min_dist_sq {
                        min_dist_sq = dist_sq;
                    }
                }
                
                distances.push(min_dist_sq);
                total_weight = total_weight + min_dist_sq;
            }
            
            // Choose next centroid with probability proportional to squared distance
            if total_weight > F::zero() {
                let target = F::from(rng.random::<f64>()).unwrap() * total_weight;
                let mut cumulative = F::zero();
                
                for (i, &dist_sq) in distances.iter().enumerate() {
                    cumulative = cumulative + dist_sq;
                    if cumulative >= target {
                        centroids.row_mut(k).assign(&data[i].features);
                        break;
                    }
                }
            } else {
                // Fallback: if all distances are zero, choose randomly
                let idx = rng.random_range(0..data.len());
                centroids.row_mut(k).assign(&data[idx].features);
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use prec::L2Dist;

    fn get_test_data() -> Vec<DataPoint<&'static str, f64>> {
        vec![
            DataPoint::new(array![1.0, 1.0], "A"),
            DataPoint::new(array![1.5, 2.0], "A"),
            DataPoint::new(array![3.0, 4.0], "A"),
            DataPoint::new(array![5.0, 7.0], "B"),
            DataPoint::new(array![3.5, 5.0], "B"),
            DataPoint::new(array![4.5, 5.0], "B"),
            DataPoint::new(array![3.5, 4.5], "B"),
        ]
    }

    #[test]
    fn test_kmeans_fit() {
        let data = get_test_data();
        let mut kmeans = KMeans::new(2, 100, 1e-4, L2Dist);
        let result = kmeans.fit(&data);

        assert!(result.is_ok());
        let (assignments, centroids) = result.unwrap();

        assert_eq!(assignments.len(), data.len());
        assert_eq!(centroids.nrows(), 2);

        // Note: The exact cluster assignments and centroid values can vary slightly
        // depending on initialization. These tests assume a deterministic outcome
        // based on the simple initialization used.
        let cluster_0_count = assignments.iter().filter(|&&a| a == 0).count();
        let cluster_1_count = assignments.iter().filter(|&&a| a == 1).count();
        assert!(cluster_0_count > 0);
        assert!(cluster_1_count > 0);

        // Check that points are clustered reasonably.
        // Points (1.0, 1.0) and (1.5, 2.0) should be in one cluster.
        // The rest should be in another.
        let first_assignment = assignments[0];
        assert_eq!(assignments[1], first_assignment);
        assert_ne!(assignments[3], first_assignment);
    }

    #[test]
    fn test_kmeans_predict() {
        let data = get_test_data();
        let mut kmeans = KMeans::new(2, 100, 1e-4, L2Dist);
        let (assignments, _) = kmeans.fit(&data).unwrap();

        let point1 = array![1.0, 1.5]; // Should be in cluster of first two points
        let point2 = array![4.0, 5.0]; // Should be in the other cluster

        let prediction1 = kmeans.predict(point1.view());
        let prediction2 = kmeans.predict(point2.view());

        assert!(prediction1.is_ok());
        assert!(prediction2.is_ok());

        assert_eq!(prediction1.unwrap(), assignments[0]);
        assert_eq!(prediction2.unwrap(), assignments[3]);
    }

    #[test]
    fn test_predict_not_fitted() {
        let kmeans: KMeans<f64, L2Dist> = KMeans::new(2, 100, 1e-4, L2Dist);
        let point = array![1.0, 1.5];
        let result = kmeans.predict(point.view());
        assert!(matches!(result, Err(KMeansError::NotFitted)));
    }

    #[test]
    fn test_invalid_k() {
        let data = get_test_data();
        let mut kmeans = KMeans::new(0, 100, 1e-4, L2Dist);
        let result = kmeans.fit(&data);
        assert!(matches!(result, Err(KMeansError::InvalidK)));
    }

    #[test]
    fn test_empty_dataset() {
        let data: Vec<DataPoint<&'static str, f64>> = vec![];
        let mut kmeans = KMeans::new(2, 100, 1e-4, L2Dist);
        let result = kmeans.fit(&data);
        assert!(matches!(result, Err(KMeansError::EmptyDataSet)));
    }

    #[test]
    fn test_k_too_large() {
        let data = get_test_data();
        let mut kmeans = KMeans::new(10, 100, 1e-4, L2Dist);
        let result = kmeans.fit(&data);
        assert!(matches!(result, Err(KMeansError::KTooLarge)));
    }
        fn make_simple_data() -> Vec<DataPoint<(), f64>> {
        let mut v = Vec::new();
        for &(x, y) in &[(0.1, -0.2), (0.2, 0.0), (-0.1, 0.1)] {
            v.push(DataPoint::new(array![x, y], ()));
        }
        for &(x, y) in &[(9.8, 10.2), (10.1, 9.9), (10.0, 10.0)] {
            v.push(DataPoint::new(array![x, y], ()));
        }
        v
    }

    #[test]
    fn test_kmeans_basic() {
        let data = make_simple_data();
        let mut model = KMeans::new(2, 100, 1e-6, L2Dist);
        let (assignments, centroids) = model.fit(&data).unwrap();
        // Expect two clusters of size 3
        let mut counts = assignments.iter().fold(vec![0;2], |mut acc, &a| { acc[a] += 1; acc });
        counts.sort();
        assert_eq!(counts, vec![3,3]);
        // Centroids near expected
        let cent0 = centroids.row(0);
        let cent1 = centroids.row(1);
        let dist = |c: ndarray::ArrayView1<f64>, tx: f64, ty: f64| ((c[0]-tx).abs() + (c[1]-ty).abs());
        let d00 = dist(cent0, 0.0, 0.0);
        let d01 = dist(cent0, 10.0, 10.0);
        let d10 = dist(cent1, 0.0, 0.0);
        let d11 = dist(cent1, 10.0, 10.0);
        // Each centroid should be closer to one of the true centers
        assert!((d00 < d01 && d11 < d10) || (d01 < d00 && d10 < d11));
    }

    #[test]
    fn test_predict() {
        let data = make_simple_data();
        let mut model = KMeans::new(2, 100, 1e-6, L2Dist);
        let (assignments, _) = model.fit(&data).unwrap();
        let p1 = array![0.0, 0.0];
        let p2 = array![10.0, 10.0];
        let c1 = model.predict(p1.view()).unwrap();
        let c2 = model.predict(p2.view()).unwrap();
        assert_eq!(c1 != c2, true);
        assert_eq!(c1, assignments[0]);
        assert_eq!(c2, assignments[3]);
    }

    #[test]
    fn test_errors() {
        let data = make_simple_data();
        let mut m = KMeans::<f64, L2Dist>::new(0,10,1e-6,L2Dist);
        assert_eq!(m.fit(&data).unwrap_err(), KMeansError::InvalidK);
        let mut m = KMeans::new(2,10,1e-6,L2Dist);
        assert_eq!(m.fit::<&str>(&[]).unwrap_err(), KMeansError::EmptyDataSet);
        let mut m = KMeans::new(10,10,1e-6,L2Dist);
        assert_eq!(m.fit(&data).unwrap_err(), KMeansError::KTooLarge);
        let bad: Vec<DataPoint<(), f64>> = vec![DataPoint::new(array![1.0,2.0],()), DataPoint::new(array![3.0],())];
        let mut m = KMeans::new(2,10,1e-6,L2Dist);
        assert_eq!(m.fit(&bad).unwrap_err(), KMeansError::MismatchedDimensions);
        let m: KMeans<f64, L2Dist> = KMeans::new(2,10,1e-6,L2Dist);
        assert_eq!(m.predict(array![0.0,0.0].view()).unwrap_err(), KMeansError::NotFitted);
    }
}
