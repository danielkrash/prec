use eframe::egui;
use k_nn::KnnClassifier;
use k_means::{KMeans, KMeansError};
use lvq::{Lvq, PrototypeInit};
use ndarray::ArrayView1;
use prec::{DataPoint, Distance, L1Dist, L2Dist, LInfDist};
use std::collections::HashMap;
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
    
    /// Get prototypes with their labels for LVQ algorithms.
    /// Returns None for non-LVQ algorithms.
    fn get_prototypes(&self) -> Option<Vec<(f64, f64, String)>> {
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
        changed |= ui.add(egui::Slider::new(&mut self.k, 1..=100).text("Clusters (k)")).changed();
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

// --- LVQ Adapter ---

/// LVQ Algorithm variants
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LvqVariant {
    LVQ1,
    LVQ2,
    LVQ3,
}

impl ToString for LvqVariant {
    fn to_string(&self) -> String {
        match self {
            LvqVariant::LVQ1 => "LVQ1".to_string(),
            LvqVariant::LVQ2 => "LVQ2".to_string(),
            LvqVariant::LVQ3 => "LVQ3".to_string(),
        }
    }
}

pub struct LvqAdapter {
    model: Option<Box<dyn LvqModel>>,
    variant: LvqVariant,
    distance_metric: DistanceMetric,
    // Hyperparameters
    n_prototypes_per_class: usize,
    learning_rate: f64,
    epochs: u32,
    decay: f64,
    window: f64,       // For LVQ2/LVQ3
    epsilon: f64,      // For LVQ3
    init_strategy: LvqInitStrategy,
    seed: u64,
    // Prototype data for visualization
    prototypes: Vec<(f64, f64, String)>, // (x, y, label)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LvqInitStrategy {
    Random,
    ClassMean,
    KMeans,
}

impl ToString for LvqInitStrategy {
    fn to_string(&self) -> String {
        match self {
            LvqInitStrategy::Random => "Random".to_string(),
            LvqInitStrategy::ClassMean => "Class Mean".to_string(),
            LvqInitStrategy::KMeans => "K-Means".to_string(),
        }
    }
}

// Helper trait to work with the generic LVQ classifier
trait LvqModel: Send + Sync {
    fn predict(&self, features: ArrayView1<f64>) -> Result<String, Box<dyn Error + Send + Sync>>;
    fn get_prototypes(&self) -> Vec<(f64, f64, String)>;
}

impl<D: 'static + Distance<f64> + Send + Sync> LvqModel for Lvq<String, f64, D> {
    fn predict(&self, features: ArrayView1<f64>) -> Result<String, Box<dyn Error + Send + Sync>> {
        Ok(self.predict(features))
    }
    
    fn get_prototypes(&self) -> Vec<(f64, f64, String)> {
        self.codebook
            .iter()
            .filter_map(|proto| {
                if proto.features.len() >= 2 {
                    Some((proto.features[0], proto.features[1], proto.label.clone()))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Classifier for LvqAdapter {
    fn predict(&self, features: ArrayView1<f64>) -> Result<String, Box<dyn Error + Send + Sync>> {
        self.model
            .as_ref()
            .ok_or_else(|| Box::<dyn Error + Send + Sync>::from("LVQ model not initialized"))?
            .predict(features)
    }

    fn name(&self) -> String {
        format!("{} ({})", self.variant.to_string(), self.distance_metric.to_string())
    }

    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.heading("LVQ Parameters");

        // Algorithm variant
        ui.horizontal(|ui| {
            ui.label("Variant:");
            if ui.selectable_value(&mut self.variant, LvqVariant::LVQ1, "LVQ1").clicked() {
                changed = true;
            }
            if ui.selectable_value(&mut self.variant, LvqVariant::LVQ2, "LVQ2").clicked() {
                changed = true;
            }
            if ui.selectable_value(&mut self.variant, LvqVariant::LVQ3, "LVQ3").clicked() {
                changed = true;
            }
        });

        // Distance metric
        ui.horizontal(|ui| {
            ui.label("Distance:");
            if ui.selectable_value(&mut self.distance_metric, DistanceMetric::L2, "L2").clicked() {
                changed = true;
            }
            if ui.selectable_value(&mut self.distance_metric, DistanceMetric::L1, "L1").clicked() {
                changed = true;
            }
            if ui.selectable_value(&mut self.distance_metric, DistanceMetric::LInf, "L∞").clicked() {
                changed = true;
            }
        });

        // Initialization strategy
        ui.horizontal(|ui| {
            ui.label("Initialization:");
            if ui.selectable_value(&mut self.init_strategy, LvqInitStrategy::Random, "Random").clicked() {
                changed = true;
            }
            if ui.selectable_value(&mut self.init_strategy, LvqInitStrategy::ClassMean, "Class Mean").clicked() {
                changed = true;
            }
            if ui.selectable_value(&mut self.init_strategy, LvqInitStrategy::KMeans, "K-Means").clicked() {
                changed = true;
            }
        });

        // Basic parameters
        ui.horizontal(|ui| {
            ui.label("Prototypes per class:");
            changed |= ui.add(egui::Slider::new(&mut self.n_prototypes_per_class, 1..=20)).changed();
        });

        ui.horizontal(|ui| {
            ui.label("Learning rate:");
            changed |= ui.add(egui::Slider::new(&mut self.learning_rate, 0.01..=0.5).logarithmic(false)).changed();
        });

        ui.horizontal(|ui| {
            ui.label("Epochs:");
            changed |= ui.add(egui::Slider::new(&mut self.epochs, 10..=500)).changed();
        });

        ui.horizontal(|ui| {
            ui.label("Decay:");
            changed |= ui.add(egui::Slider::new(&mut self.decay, 0.9..=1.0)).changed();
        });

        // LVQ2/LVQ3 specific parameters
        if matches!(self.variant, LvqVariant::LVQ2 | LvqVariant::LVQ3) {
            ui.horizontal(|ui| {
                ui.label("Window:");
                changed |= ui.add(egui::Slider::new(&mut self.window, 0.1..=0.5)).changed();
            });
        }

        // LVQ3 specific parameters
        if matches!(self.variant, LvqVariant::LVQ3) {
            ui.horizontal(|ui| {
                ui.label("Epsilon:");
                changed |= ui.add(egui::Slider::new(&mut self.epsilon, 0.1..=1.0)).changed();
            });
        }

        // Seed for reproducibility
        ui.horizontal(|ui| {
            ui.label("Seed:");
            changed |= ui.add(egui::DragValue::new(&mut self.seed)).changed();
        });

        changed
    }

    fn update_data(&mut self, data: &[DataPoint<String, f64>]) {
        if data.is_empty() {
            self.model = None;
            self.prototypes.clear();
            return;
        }

        // Create prototype count map
        let unique_classes: std::collections::HashSet<String> = data.iter().map(|d| d.label.clone()).collect();
        let mut n_prototypes_per_class = HashMap::new();
        for class in unique_classes {
            n_prototypes_per_class.insert(class, self.n_prototypes_per_class);
        }

        // Convert initialization strategy
        let init = match self.init_strategy {
            LvqInitStrategy::Random => PrototypeInit::Random,
            LvqInitStrategy::ClassMean => PrototypeInit::ClassMean,
            LvqInitStrategy::KMeans => PrototypeInit::KMeans { max_iter: 100, tol: 1e-4 },
        };

        // Train based on variant and distance metric
        let result = match (&self.variant, &self.distance_metric) {
            (LvqVariant::LVQ1, DistanceMetric::L2) => {
                lvq::fit_with_init_and_seed(data, &n_prototypes_per_class, self.learning_rate, self.epochs, L2Dist, self.decay, init, self.seed)
                    .map(|model| Box::new(model) as Box<dyn LvqModel>)
            }
            (LvqVariant::LVQ1, DistanceMetric::L1) => {
                lvq::fit_with_init_and_seed(data, &n_prototypes_per_class, self.learning_rate, self.epochs, L1Dist, self.decay, init, self.seed)
                    .map(|model| Box::new(model) as Box<dyn LvqModel>)
            }
            (LvqVariant::LVQ1, DistanceMetric::LInf) => {
                lvq::fit_with_init_and_seed(data, &n_prototypes_per_class, self.learning_rate, self.epochs, LInfDist, self.decay, init, self.seed)
                    .map(|model| Box::new(model) as Box<dyn LvqModel>)
            }
            (LvqVariant::LVQ2, DistanceMetric::L2) => {
                lvq::fit_lvq2_with_init_and_seed(data, &n_prototypes_per_class, self.learning_rate, self.epochs, L2Dist, self.decay, self.window, init, self.seed)
                    .map(|model| Box::new(model) as Box<dyn LvqModel>)
            }
            (LvqVariant::LVQ2, DistanceMetric::L1) => {
                lvq::fit_lvq2_with_init_and_seed(data, &n_prototypes_per_class, self.learning_rate, self.epochs, L1Dist, self.decay, self.window, init, self.seed)
                    .map(|model| Box::new(model) as Box<dyn LvqModel>)
            }
            (LvqVariant::LVQ2, DistanceMetric::LInf) => {
                lvq::fit_lvq2_with_init_and_seed(data, &n_prototypes_per_class, self.learning_rate, self.epochs, LInfDist, self.decay, self.window, init, self.seed)
                    .map(|model| Box::new(model) as Box<dyn LvqModel>)
            }
            (LvqVariant::LVQ3, DistanceMetric::L2) => {
                lvq::fit_lvq3_with_init_and_seed(data, &n_prototypes_per_class, self.learning_rate, self.epochs, L2Dist, self.decay, self.window, self.epsilon, init, self.seed)
                    .map(|model| Box::new(model) as Box<dyn LvqModel>)
            }
            (LvqVariant::LVQ3, DistanceMetric::L1) => {
                lvq::fit_lvq3_with_init_and_seed(data, &n_prototypes_per_class, self.learning_rate, self.epochs, L1Dist, self.decay, self.window, self.epsilon, init, self.seed)
                    .map(|model| Box::new(model) as Box<dyn LvqModel>)
            }
            (LvqVariant::LVQ3, DistanceMetric::LInf) => {
                lvq::fit_lvq3_with_init_and_seed(data, &n_prototypes_per_class, self.learning_rate, self.epochs, LInfDist, self.decay, self.window, self.epsilon, init, self.seed)
                    .map(|model| Box::new(model) as Box<dyn LvqModel>)
            }
        };

        match result {
            Ok(model) => {
                self.prototypes = model.get_prototypes();
                self.model = Some(model);
            }
            Err(e) => {
                eprintln!("Failed to fit LVQ: {:?}", e);
                self.model = None;
                self.prototypes.clear();
            }
        }
    }

    // LVQ-specific visualization methods
    fn cluster_centers(&self) -> Option<Vec<(f64, f64)>> {
        if self.prototypes.is_empty() {
            None
        } else {
            Some(self.prototypes.iter().map(|(x, y, _)| (*x, *y)).collect())
        }
    }

    fn get_prototypes(&self) -> Option<Vec<(f64, f64, String)>> {
        if self.prototypes.is_empty() {
            None
        } else {
            Some(self.prototypes.clone())
        }
    }
}

impl LvqAdapter {
    pub fn new() -> Self {
        Self {
            model: None,
            variant: LvqVariant::LVQ1,
            distance_metric: DistanceMetric::L2,
            n_prototypes_per_class: 2, // Show multiple prototypes per class
            learning_rate: 0.1,
            epochs: 200,
            decay: 0.95,
            window: 0.3,
            epsilon: 0.5,
            init_strategy: LvqInitStrategy::Random,
            seed: rand::random::<u64>(), // Random seed for reproducibility
            prototypes: Vec::new(),
        }
    }

    /// Get the prototypes for visualization
    pub fn get_prototypes(&self) -> &[(f64, f64, String)] {
        &self.prototypes
    }
}
