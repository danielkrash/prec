use eframe::egui;
use k_nn::KnnClassifier;
use k_means::{KMeans, KMeansError};
use ndarray::ArrayView1;
use prec::{DataPoint, Distance, L1Dist, L2Dist, LInfDist};
use std::error::Error;

/// A trait that defines the common interface for all classifiers in the visualizer.
pub trait Classifier: Send + Sync {
    /// Predict the class for a single point.
    fn predict(&self, features: ArrayView1<f64>) -> Result<String, Box<dyn Error + Send + Sync>>;

    /// Return the name of the classifier (e.g., "k-NN (L2)").
    fn name(&self) -> String;

    /// Get the UI controls specific to this classifier.
    fn ui(&mut self, ui: &mut egui::Ui) -> bool; // Returns true if state changed

    /// Re-train or update the classifier with new data.
    fn update_data(&mut self, data: &[DataPoint<String, f64>]);

    /// Get cluster centers if this is a clustering algorithm.
    /// Returns None for non-clustering algorithms like k-NN.
    fn cluster_centers(&self) -> Option<Vec<(f64, f64)>> {
        None
    }
    
    /// Get the cluster index for a prediction (for clustering algorithms).
    /// Returns None for non-clustering algorithms.
    fn predict_cluster_index(&self, _features: ArrayView1<f64>) -> Option<usize> {
        None
    }
    
    /// Get point assignments for clustering algorithms.
    /// Returns None for non-clustering algorithms.
    fn point_assignments(&self) -> Option<&[usize]> {
        None
    }
    
    /// Get stable color index for a cluster (for clustering algorithms).
    /// This ensures consistent colors across retraining.
    fn get_stable_cluster_color(&self, _cluster_idx: usize) -> Option<usize> {
        None
    }
}

// Now, we implement this trait for our k-NN classifiers.
// We'll create a single generic struct for this.

pub struct KnnAdapter {
    classifier: Option<Box<dyn KnnPredictor>>,
    k: usize,
    distance_metric: DistanceMetric,
}

// Helper trait to work with the generic k-NN classifier
trait KnnPredictor: Send + Sync {
    fn predict(&self, features: ArrayView1<f64>) -> Result<String, k_nn::KnnError>;
}

// Helper trait to work with the generic K-Means classifier
trait KMeansModel: Send + Sync {
    fn predict(&self, features: ArrayView1<f64>) -> Result<usize, KMeansError>;
    fn k(&self) -> usize;
    fn centroids(&self) -> Result<ndarray::ArrayView2<f64>, KMeansError>;
}

impl<D: 'static + Distance<f64> + Send + Sync> KMeansModel for KMeans<f64, D> {
    fn predict(&self, features: ArrayView1<f64>) -> Result<usize, KMeansError> {
        self.predict(features)
    }
    fn k(&self) -> usize {
        self.k
    }
    fn centroids(&self) -> Result<ndarray::ArrayView2<f64>, KMeansError> {
        self.centroids()
    }
}

impl<D: 'static + Distance<f64> + Send + Sync> KnnPredictor for KnnClassifier<String, f64, D> {
    fn predict(&self, features: ArrayView1<f64>) -> Result<String, k_nn::KnnError> {
        self.predict(features)
    }
}

// Now implement the main `Classifier` trait for our adapter
impl Classifier for KnnAdapter {
    fn predict(&self, features: ArrayView1<f64>) -> Result<String, Box<dyn Error + Send + Sync>> {
        self.classifier
            .as_ref()
            .ok_or_else(|| Box::<dyn Error + Send + Sync>::from("Classifier not initialized"))?
            .predict(features)
            .map_err(|e| e.into())
    }

    fn name(&self) -> String {
        format!("k-NN ({})", self.distance_metric.to_string())
    }

    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.heading("k-NN Parameters");
        changed |= ui
            .add(egui::Slider::new(&mut self.k, 1..=31).text("k Value"))
            .changed();
        ui.label("Distance Metric:");
        changed |= ui
            .radio_value(
                &mut self.distance_metric,
                DistanceMetric::L1,
                "L1 (Manhattan)",
            )
            .changed();
        changed |= ui
            .radio_value(
                &mut self.distance_metric,
                DistanceMetric::L2,
                "L2 (Euclidean)",
            )
            .changed();
        changed |= ui
            .radio_value(
                &mut self.distance_metric,
                DistanceMetric::LInf,
                "L-Infinity",
            )
            .changed();
        changed
    }

    fn update_data(&mut self, data: &[DataPoint<String, f64>]) {
        if data.is_empty() {
            self.classifier = None;
            return;
        }
        let k = self.k;
        let owned_data = data.to_vec();

        let new_classifier: Box<dyn KnnPredictor> = match self.distance_metric {
            DistanceMetric::L1 => Box::new(KnnClassifier::new(k, owned_data, L1Dist).unwrap()),
            DistanceMetric::L2 => Box::new(KnnClassifier::new(k, owned_data, L2Dist).unwrap()),
            DistanceMetric::LInf => Box::new(KnnClassifier::new(k, owned_data, LInfDist).unwrap()),
        };
        self.classifier = Some(new_classifier);
    }
}

// Add constructors and other helpers to KnnAdapter
impl KnnAdapter {
    pub fn new() -> Self {
        Self {
            classifier: None,
            k: 3,
            distance_metric: DistanceMetric::L2,
        }
    }
}

// The DistanceMetric enum can also live here
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistanceMetric {
    L1,
    L2,
    LInf,
}
impl ToString for DistanceMetric {
    fn to_string(&self) -> String {
        match self {
            DistanceMetric::L1 => "L1".to_string(),
            DistanceMetric::L2 => "L2".to_string(),
            DistanceMetric::LInf => "L-Infinity".to_string(),
        }
    }
}

// --- K-Means Adapter ---

pub struct KMeansAdapter {
    model: Option<Box<dyn KMeansModel>>,
    k: usize,
    max_iter: u32,
    tolerance: f64,
    distance_metric: DistanceMetric,
    // K-Means is unsupervised, so it doesn't predict labels directly in the same way.
    // We'll store the cluster assignments and map them to the original labels.
    cluster_labels: Option<Vec<String>>,
    assignments: Option<Vec<usize>>,
}

impl KMeansAdapter {
    pub fn new() -> Self {
        Self {
            model: None,
            k: 2, // Default to 2 clusters
            max_iter: 100,
            tolerance: 1e-4,
            distance_metric: DistanceMetric::L2,
            cluster_labels: None,
            assignments: None,
        }
    }

    /// Get the stable color index for a given cluster index
    pub fn get_cluster_color_index(&self, cluster_idx: usize) -> usize {
        // Simply return the cluster index - this ensures each cluster has a unique color
        // The color array in UI has 10 colors, so we cycle through them
        cluster_idx % 10
    }
}

impl Classifier for KMeansAdapter {
    fn predict(&self, features: ArrayView1<f64>) -> Result<String, Box<dyn Error + Send + Sync>> {
        if let (Some(model), Some(labels)) = (&self.model, &self.cluster_labels) {
            let cluster_idx = model.predict(features)?;
            Ok(labels.get(cluster_idx).cloned().unwrap_or_else(|| "Unknown".to_string()))
        } else {
            Err("Model not fitted".into())
        }
    }

    fn name(&self) -> String {
        format!("K-Means ({})", self.distance_metric.to_string())
    }

    fn point_assignments(&self) -> Option<&[usize]> {
        self.assignments.as_deref()
    }

    fn cluster_centers(&self) -> Option<Vec<(f64, f64)>> {
        if let Some(model) = &self.model {
            if let Ok(centroids) = model.centroids() {
                let centers: Vec<(f64, f64)> = centroids
                    .rows()
                    .into_iter()
                    .map(|row| (row[0], row[1]))
                    .collect();
                return Some(centers);
            }
        }
        None
    }

    fn predict_cluster_index(&self, features: ArrayView1<f64>) -> Option<usize> {
        if let Some(model) = &self.model {
            model.predict(features).ok()
        } else {
            None
        }
    }

    fn get_stable_cluster_color(&self, cluster_idx: usize) -> Option<usize> {
        Some(self.get_cluster_color_index(cluster_idx))
    }

    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.heading("K-Means Parameters");
        changed |= ui.add(egui::Slider::new(&mut self.k, 1..=10).text("Clusters (k)")).changed();
        changed |= ui.add(egui::Slider::new(&mut self.max_iter, 1..=1000).text("Max Iterations")).changed();
        
        // Better tolerance range - using a more intuitive scale
        let mut tolerance_exp = self.tolerance.log10();
        if ui.add(egui::Slider::new(&mut tolerance_exp, -6.0..=-1.0).text("Tolerance (log10)")).changed() {
            self.tolerance = 10_f64.powf(tolerance_exp);
            changed = true;
        }
        ui.label(format!("Current tolerance: {:.1e}", self.tolerance));

        ui.label("Distance Metric:");
        changed |= ui.radio_value(&mut self.distance_metric, DistanceMetric::L1, "L1 (Manhattan)").changed();
        changed |= ui.radio_value(&mut self.distance_metric, DistanceMetric::L2, "L2 (Euclidean)").changed();
        changed |= ui.radio_value(&mut self.distance_metric, DistanceMetric::LInf, "L-Infinity").changed();

        changed
    }

    fn update_data(&mut self, data: &[DataPoint<String, f64>]) {
        if data.is_empty() || self.k == 0 {
            self.model = None;
            self.cluster_labels = None;
            self.assignments = None;
            return;
        }
        println!("{:?}" , data);
        let (model, result) = match self.distance_metric {
            DistanceMetric::L1 => {
                let mut model = KMeans::new(self.k, self.max_iter, self.tolerance, L1Dist);
                let res = model.fit(data);
                (Box::new(model) as Box<dyn KMeansModel>, res)
            }
            DistanceMetric::L2 => {
                let mut model = KMeans::new(self.k, self.max_iter, self.tolerance, L2Dist);
                let res = model.fit(data);
                (Box::new(model) as Box<dyn KMeansModel>, res)
            }
            DistanceMetric::LInf => {
                let mut model = KMeans::new(self.k, self.max_iter, self.tolerance, LInfDist);
                let res = model.fit(data);
                (Box::new(model) as Box<dyn KMeansModel>, res)
            }
        };

        match result {
            Ok((assignments, _centroids)) => {
                // Create cluster labels - if there are more clusters than unique labels,
                // generate synthetic cluster names
                let mut cluster_label_map = vec![String::new(); self.k];
                let unique_labels: std::collections::HashSet<String> = data.iter().map(|dp| dp.label.clone()).collect();
                let unique_labels: Vec<String> = unique_labels.into_iter().collect();
                
                for i in 0..self.k {
                    // Find the most common original label in this cluster
                    let cluster_points: Vec<&str> = assignments
                        .iter()
                        .zip(data.iter())
                        .filter(|(cluster, _)| **cluster == i)
                        .map(|(_, dp)| dp.label.as_str())
                        .collect();
                    
                    println!("Cluster {}: {} points: {:?}", i, cluster_points.len(), cluster_points);
                    
                    if cluster_points.is_empty() {
                        // Empty cluster - assign a unique name
                        cluster_label_map[i] = format!("Cluster {}", i + 1);
                        println!("  -> Empty cluster, using 'Cluster {}'", i + 1);
                    } else {
                        // Find most common label in this cluster
                        let mut label_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
                        for label in &cluster_points {
                            *label_counts.entry(label).or_insert(0) += 1;
                        }
                        
                        println!("  -> Label counts: {:?}", label_counts);
                        
                        let most_common_label = label_counts
                            .into_iter()
                            .max_by_key(|(_, count)| *count)
                            .map(|(label, _)| label)
                            .unwrap_or("Unknown");
                        
                        // If we have more clusters than unique labels, append cluster number
                        if self.k > unique_labels.len() {
                            cluster_label_map[i] = format!("{}-C{}", most_common_label, i + 1);
                            println!("  -> Using '{}-C{}'", most_common_label, i + 1);
                        } else {
                            cluster_label_map[i] = most_common_label.to_string();
                            println!("  -> Using '{}'", most_common_label);
                        }
                    }
                }

                println!("Final cluster labels: {:?}", cluster_label_map);
                println!("Final assignments: {:?}", assignments);

                self.model = Some(model);
                self.cluster_labels = Some(cluster_label_map);
                self.assignments = Some(assignments);
            }
            Err(e) => {
                eprintln!("Failed to fit K-Means: {:?}", e);
                self.model = None;
                self.cluster_labels = None;
                self.assignments = None;
            }
        }
    }
}
