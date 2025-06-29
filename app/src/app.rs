use crate::classifier::{Classifier, KnnAdapter, KMeansAdapter, LvqAdapter};
// When you add k-means, you'll uncomment this:
// use crate::classifier::KMeansAdapter; 
use crate::ui;


use eframe::egui::{self, Color32, Pos2, Rect};
use ecolor::Hsva;
use eframe::{App, Frame};
use ndarray::array;
use prec::DataPoint;
use std::collections::HashSet;
use std::error::Error;

/// An enum to select which algorithm we are visualizing.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Algorithm {
    Knn,
    KMeans,
    LVQ,
}

/// An enum to control what happens when the user clicks on the plot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InteractionMode {
    Classify,
    AddPoint,
    MultiPredict,
}

/// The main application struct.
/// It holds the high-level state and delegates logic to other modules.
pub struct MyApp {
    // --- Core State ---
    /// The active classifier, wrapped in a Box to allow for different types.
    pub classifier: Box<dyn Classifier>,
    /// The current dataset used for training.
    pub training_data: Vec<DataPoint<String, f64>>,
    /// A list of points the user has clicked to get predictions for.
    pub prediction_points: Vec<(Pos2, Result<String, Box<dyn Error + Send + Sync>>)>,

    // --- UI State ---
    /// The algorithm currently selected in the UI.
    pub active_algorithm: Algorithm,
    /// The current interaction mode.
    pub interaction_mode: InteractionMode,
    /// The pre-computed decision boundary map. Stored to avoid regenerating every frame.
    pub classification_map: Option<(egui::TextureHandle, (usize, usize))>,
    /// Defines the coordinate system of the data space.
    pub data_rect: Rect,
    /// The single point for "Classify" mode.
    pub single_prediction_point: Pos2,
    /// The last prediction result for the single point.
    pub last_single_prediction: Result<String, Box<dyn Error + Send + Sync>>,

    // --- State for the 'AddPoint' mode UI ---
    pub available_classes: Vec<String>,
    pub selected_class_for_add: String,
    pub new_class_input: String,
}

impl Default for MyApp {
    /// Creates the application with some default sample data.
    fn default() -> Self {
        let training_data = vec![
            // Class A - Cluster around (2, 3) - 20 points
            DataPoint::new(array![1.5, 2.5], "A".to_string()),
            DataPoint::new(array![2.0, 3.0], "A".to_string()),
            DataPoint::new(array![2.5, 3.5], "A".to_string()),
            DataPoint::new(array![1.8, 2.8], "A".to_string()),
            DataPoint::new(array![2.2, 3.2], "A".to_string()),
            DataPoint::new(array![1.6, 3.1], "A".to_string()),
            DataPoint::new(array![2.4, 2.7], "A".to_string()),
            DataPoint::new(array![1.9, 3.4], "A".to_string()),
            DataPoint::new(array![2.1, 2.6], "A".to_string()),
            DataPoint::new(array![2.3, 3.3], "A".to_string()),
            DataPoint::new(array![1.7, 2.9], "A".to_string()),
            DataPoint::new(array![2.6, 3.1], "A".to_string()),
            DataPoint::new(array![1.4, 3.2], "A".to_string()),
            DataPoint::new(array![2.8, 2.8], "A".to_string()),
            DataPoint::new(array![1.3, 2.7], "A".to_string()),
            DataPoint::new(array![2.7, 3.4], "A".to_string()),
            DataPoint::new(array![1.1, 3.0], "A".to_string()),
            DataPoint::new(array![2.9, 2.5], "A".to_string()),
            DataPoint::new(array![1.2, 3.5], "A".to_string()),
            DataPoint::new(array![2.0, 2.4], "A".to_string()),
            
            // Class B - Cluster around (7, 6) - 20 points  
            DataPoint::new(array![6.5, 5.5], "B".to_string()),
            DataPoint::new(array![7.0, 6.0], "B".to_string()),
            DataPoint::new(array![7.5, 6.5], "B".to_string()),
            DataPoint::new(array![6.8, 5.8], "B".to_string()),
            DataPoint::new(array![7.2, 6.2], "B".to_string()),
            DataPoint::new(array![6.6, 6.1], "B".to_string()),
            DataPoint::new(array![7.4, 5.7], "B".to_string()),
            DataPoint::new(array![6.9, 6.4], "B".to_string()),
            DataPoint::new(array![7.1, 5.6], "B".to_string()),
            DataPoint::new(array![7.3, 6.3], "B".to_string()),
            DataPoint::new(array![6.7, 5.9], "B".to_string()),
            DataPoint::new(array![7.6, 6.1], "B".to_string()),
            DataPoint::new(array![6.4, 6.2], "B".to_string()),
            DataPoint::new(array![7.8, 5.8], "B".to_string()),
            DataPoint::new(array![6.3, 5.7], "B".to_string()),
            DataPoint::new(array![7.7, 6.4], "B".to_string()),
            DataPoint::new(array![6.1, 6.0], "B".to_string()),
            DataPoint::new(array![7.9, 5.5], "B".to_string()),
            DataPoint::new(array![6.2, 6.5], "B".to_string()),
            DataPoint::new(array![7.0, 5.4], "B".to_string()),
            
            // Class C - Cluster around (5, 8.5) - 15 points (making it slightly unbalanced)
            DataPoint::new(array![4.5, 8.0], "C".to_string()),
            DataPoint::new(array![5.0, 8.5], "C".to_string()),
            DataPoint::new(array![5.5, 9.0], "C".to_string()),
            DataPoint::new(array![4.8, 8.3], "C".to_string()),
            DataPoint::new(array![5.2, 8.7], "C".to_string()),
            DataPoint::new(array![4.6, 8.6], "C".to_string()),
            DataPoint::new(array![5.4, 8.2], "C".to_string()),
            DataPoint::new(array![4.9, 8.9], "C".to_string()),
            DataPoint::new(array![5.1, 8.1], "C".to_string()),
            DataPoint::new(array![5.3, 8.8], "C".to_string()),
            DataPoint::new(array![4.7, 8.4], "C".to_string()),
            DataPoint::new(array![5.6, 8.6], "C".to_string()),
            DataPoint::new(array![4.4, 8.7], "C".to_string()),
            DataPoint::new(array![5.8, 8.3], "C".to_string()),
            DataPoint::new(array![4.3, 8.1], "C".to_string()),
        ];

        let available_classes = Self::extract_unique_classes(&training_data);
        let selected_class_for_add = available_classes.first().cloned().unwrap_or_default();
        let single_prediction_point = Pos2::new(5.0, 5.0);

        let mut app = Self {
            classifier: Box::new(KnnAdapter::new()), // Start with a k-NN classifier
            training_data,
            prediction_points: Vec::new(),
            active_algorithm: Algorithm::Knn,
            interaction_mode: InteractionMode::Classify,
            classification_map: None,
            data_rect: Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(10.0, 10.0)),
            single_prediction_point,
            last_single_prediction: Err("No prediction yet".into()),
            available_classes,
            selected_class_for_add,
            new_class_input: String::new(),
        };

        // Perform the initial training and prediction
        app.classifier.update_data(&app.training_data);
        app.recalculate_single_prediction();

        app
    }
}

impl App for MyApp {
    /// The main update loop, called by eframe on every frame.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        // This loop is now very clean. It just calls the drawing functions
        // from the `ui` module, passing its state to them.
        ui::draw_side_panel(self, ctx);
        ui::draw_central_panel(self, ctx);
    }
}

impl MyApp {
    /// Forces a recalculation of all predictions in "Multi Predict" mode.
    /// This should be called whenever the model or its parameters change.
    pub fn recalculate_multi_predictions(&mut self) {
        for (pos, prediction) in &mut self.prediction_points {
            let features = array![pos.x as f64, pos.y as f64];
            *prediction = self.classifier.predict(features.view());
        }
    }

    /// Recalculates the prediction for the single point in "Classify" mode.
    pub fn recalculate_single_prediction(&mut self) {
        let features = array![
            self.single_prediction_point.x as f64,
            self.single_prediction_point.y as f64
        ];
        self.last_single_prediction = self.classifier.predict(features.view());
    }

    /// Replaces the entire training dataset and retrains the classifier.
    pub fn set_training_data(&mut self, data: Vec<DataPoint<String, f64>>) {
        self.training_data = data;
        self.update_available_classes();
        self.classifier.update_data(&self.training_data);
        // Force regeneration of the map and update all predictions
        self.classification_map = None;
        self.recalculate_single_prediction();
        self.recalculate_multi_predictions();
    }
    
    /// Adds a single new point to the training data and updates the classifier.
    pub fn add_training_point(&mut self, point: DataPoint<String, f64>) {
        self.training_data.push(point);
        self.update_available_classes();
        self.classifier.update_data(&self.training_data);
        // Force regeneration and updates
        self.classification_map = None;
        self.recalculate_single_prediction();
        self.recalculate_multi_predictions();
    }

    /// Scans the training data to find all unique class labels.
    fn extract_unique_classes(training_data: &[DataPoint<String, f64>]) -> Vec<String> {
        let classes: HashSet<String> = training_data.iter().map(|p| p.label.clone()).collect();
        let mut sorted_classes: Vec<String> = classes.into_iter().collect();
        sorted_classes.sort();
        sorted_classes
    }

    /// Updates the list of available classes for the UI dropdown.
    pub fn update_available_classes(&mut self) {
        self.available_classes = Self::extract_unique_classes(&self.training_data);
        if !self.available_classes.contains(&self.selected_class_for_add) {
            self.selected_class_for_add = self
                .available_classes
                .first()
                .cloned()
                .unwrap_or_else(|| "Class1".to_string());
        }
    }
    
    /// Generates a consistent color for a class label using a simple hash.
    /// This can be used across the app to ensure 'Class A' is always the same color.
    pub fn get_class_color(class_name: &str) -> Color32 {
        let hash = class_name
            .bytes()
            .fold(0u32, |acc, byte| acc.wrapping_mul(31).wrapping_add(byte as u32));
        
        let golden_ratio_conjugate = 0.61803398875;
        let hue = (hash as f32 * golden_ratio_conjugate).fract();
    
        let hsva = Hsva { h: hue, s: 0.85, v: 0.9, a: 1.0 };
        Color32::from(hsva)
    }
}