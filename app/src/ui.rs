use crate::app::{Algorithm, InteractionMode, MyApp};
use crate::classifier::{Classifier, KMeansAdapter, KnnAdapter, LvqAdapter};

use eframe::egui::{self, ecolor::Hsva, Color32, Pos2, Rect, Sense, Stroke, Ui};
use ndarray::{array, ArrayView1};
use prec::DataPoint;
use std::error::Error;

/// Draws the entire left-side panel with all the controls.
pub fn draw_side_panel(app: &mut MyApp, ctx: &egui::Context) {
    egui::SidePanel::left("controls_panel").show(ctx, |ui| {
        ui.heading("ML Visualizer");
        ui.separator();

        draw_algorithm_selector(app, ui);
        ui.separator();

        // Let the current classifier draw its own specific UI controls.
        // The `ui` method returns `true` if any parameter was changed.
        if app.classifier.ui(ui) {
            // THE FIX IS HERE: If the classifier's UI reports a change,
            // we must update the classifier logic and invalidate the visual map.
            app.classifier.update_data(&app.training_data);
            app.classification_map = None; // This forces the map to be regenerated.
            
            // Also, recalculate any existing predictions.
            app.recalculate_single_prediction();
            app.recalculate_multi_predictions();
        }

        ui.separator();
        draw_interaction_mode_controls(app, ui);
    });
}

/// Draws the central panel containing the main visualization plot.
pub fn draw_central_panel(app: &mut MyApp, ctx: &egui::Context) {
    egui::CentralPanel::default().show(ctx, |ui| {
        // Regenerate the map if it has been invalidated (is None).
        if app.classification_map.is_none() && !app.training_data.is_empty() {
            app.classification_map = Some(generate_classification_map(&*app.classifier, ctx));
        }
        let point_assignments = app.classifier.point_assignments();
        // Allocate painter and handle coordinate transformations.
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click());
        let to_screen = egui::emath::RectTransform::from_to(app.data_rect, response.rect);
        let to_data = to_screen.inverse();

        // --- Drawing Layers ---
        if let Some((texture, _)) = &app.classification_map {
            painter.image(
                texture.id(),
                response.rect,
                Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }

    for (i, point) in app.training_data.iter().enumerate() {
        let color = match (app.active_algorithm, point_assignments) {
            // If we are in KMeans mode AND we have valid assignments...
            (Algorithm::KMeans, Some(assignments)) => {
                // ...then color the point based on its assigned cluster index with stable colors.
                let cluster_idx = assignments[i];
                get_stable_cluster_color(&*app.classifier, cluster_idx)
            }
            // Otherwise (for k-NN or if k-means failed), fall back to the original label color.
            _ => MyApp::get_class_color(&point.label),
        };

        let center = to_screen * Pos2::new(point.features[0] as f32, point.features[1] as f32);
        painter.circle_filled(center, 5.0, color);
        painter.circle_stroke(center, 5.0, Stroke::new(1.0, Color32::BLACK));
    }

        // Draw cluster centers for clustering algorithms (e.g., K-Means)
        if let Some(cluster_centers) = app.classifier.cluster_centers() {
            for (i, (x, y)) in cluster_centers.iter().enumerate() {
                let center_pos = to_screen * Pos2::new(*x as f32, *y as f32);
                
                // Draw a distinctive marker for cluster centers
                // Large white circle with black border
                painter.circle_filled(center_pos, 12.0, Color32::WHITE);
                painter.circle_stroke(center_pos, 12.0, Stroke::new(3.0, Color32::BLACK));
                
                // Draw a smaller colored circle inside based on stable cluster color
                let cluster_color = get_stable_cluster_color(&*app.classifier, i);
                painter.circle_filled(center_pos, 8.0, cluster_color);
                
                // Draw an 'X' mark to make it even more distinctive
                let x_size = 6.0;
                painter.line_segment(
                    [
                        Pos2::new(center_pos.x - x_size, center_pos.y - x_size),
                        Pos2::new(center_pos.x + x_size, center_pos.y + x_size),
                    ],
                    Stroke::new(2.0, Color32::BLACK),
                );
                painter.line_segment(
                    [
                        Pos2::new(center_pos.x - x_size, center_pos.y + x_size),
                        Pos2::new(center_pos.x + x_size, center_pos.y - x_size),
                    ],
                    Stroke::new(2.0, Color32::BLACK),
                );
            }
        }

        // Draw LVQ prototypes with labels
        if let Some(prototypes) = app.classifier.get_prototypes() {
            for (x, y, label) in prototypes.iter() {
                let proto_pos = to_screen * Pos2::new(*x as f32, *y as f32);
                
                // Draw prototype as a diamond shape with class color
                let class_color = MyApp::get_class_color(label);
                let size = 10.0;
                
                // Draw diamond shape
                let diamond_points = [
                    Pos2::new(proto_pos.x, proto_pos.y - size),      // top
                    Pos2::new(proto_pos.x + size, proto_pos.y),      // right
                    Pos2::new(proto_pos.x, proto_pos.y + size),      // bottom
                    Pos2::new(proto_pos.x - size, proto_pos.y),      // left
                ];
                
                // Fill diamond
                painter.add(egui::Shape::convex_polygon(
                    diamond_points.to_vec(),
                    class_color,
                    Stroke::new(2.0, Color32::BLACK),
                ));
                
                // Add a smaller white center
                painter.circle_filled(proto_pos, 4.0, Color32::WHITE);
                
                // Add label text below the prototype
                painter.text(
                    Pos2::new(proto_pos.x, proto_pos.y + size + 5.0),
                    egui::Align2::CENTER_TOP,
                    label,
                    egui::FontId::default(),
                    Color32::BLACK,
                );
            }
        }

        match app.interaction_mode {
            InteractionMode::Classify => {
                let screen_pos = to_screen * app.single_prediction_point;
                painter.circle_filled(screen_pos, 7.0, Color32::YELLOW);
                painter.circle_stroke(screen_pos, 7.0, Stroke::new(2.0, Color32::BLACK));
            }
            InteractionMode::MultiPredict => {
                for (pos, _prediction) in &app.prediction_points {
                    let screen_pos = to_screen * *pos;
                    // Use the same color mapping as the background for consistency
                    let features = array![pos.x as f64, pos.y as f64];
                    let color = map_prediction_to_color_with_features(features.view(), false, &*app.classifier);
                    painter.circle_filled(screen_pos, 6.0, color);
                    painter.circle_stroke(screen_pos, 6.0, Stroke::new(2.0, Color32::BLACK));
                }
            }
            InteractionMode::AddPoint => {}
        }

        // --- Interaction Handling ---
        if let Some(hover_pos) = response.hover_pos() {
            if response.clicked() {
                let clicked_pos_data = to_data * hover_pos;
                handle_plot_click(app, clicked_pos_data);
            }
        }
    });
}

// ... The rest of your ui.rs functions (draw_algorithm_selector, etc.) remain the same ...
// Make sure this `map_prediction_to_color` function is also present.

/// Helper to map a prediction result to a color for drawing.
fn map_prediction_to_color(prediction: &Result<String, Box<dyn Error + Send + Sync>>, is_background: bool, classifier: &dyn Classifier) -> Color32 {
    let base_color = match prediction {
        Ok(class_name) => {
            // Check if this is a K-means cluster (has cluster centers)
            if let Some(cluster_centers) = classifier.cluster_centers() {
                // For K-means, try to extract cluster index from class name
                if let Some(cluster_idx) = extract_cluster_index(class_name) {
                    get_stable_cluster_color(classifier, cluster_idx)
                } else {
                    // No cluster index found, try to map by position in cluster_centers
                    // This handles cases where labels are just "A", "B", etc.
                    get_stable_cluster_color_for_class(class_name, &cluster_centers, classifier)
                }
            } else {
                // Non-clustering algorithm, use original colors
                MyApp::get_class_color(class_name)
            }
        },
        Err(_) => Color32::from_gray(80),
    };

    if is_background {
        base_color.linear_multiply(0.3)
    } else {
        base_color
    }
}

/// Helper to map predictions with direct cluster index access
fn map_prediction_to_color_with_features(features: ArrayView1<f64>, is_background: bool, classifier: &dyn Classifier) -> Color32 {
    let base_color = if let Some(cluster_idx) = classifier.predict_cluster_index(features) {
        // Use stable cluster color for accurate color mapping
        get_stable_cluster_color(classifier, cluster_idx)
    } else {
        // Fallback to regular prediction-based color mapping
        let prediction = classifier.predict(features);
        match prediction {
            Ok(class_name) => {
                if let Some(_cluster_centers) = classifier.cluster_centers() {
                    let hash = class_name.bytes().fold(0u32, |acc, byte| acc.wrapping_mul(31).wrapping_add(byte as u32));
                    let cluster_idx = (hash as usize) % _cluster_centers.len();
                    get_stable_cluster_color(classifier, cluster_idx)
                } else {
                    MyApp::get_class_color(&class_name)
                }
            },
            Err(_) => Color32::from_gray(80),
        }
    };

    if is_background {
        base_color.linear_multiply(0.3)
    } else {
        base_color
    }
}

/// Extract cluster index from class names like "A-C1", "B-C2", etc.
fn extract_cluster_index(class_name: &str) -> Option<usize> {
    if let Some(pos) = class_name.find("-C") {
        if let Ok(idx) = class_name[pos + 2..].parse::<usize>() {
            return Some(idx.saturating_sub(1)); // Convert from 1-based to 0-based
        }
    }
    None
}

/// Get cluster color for a class name when no cluster index is available
/// This uses a deterministic mapping so the same class always gets the same cluster color
fn get_stable_cluster_color_for_class(class_name: &str, cluster_centers: &[(f64, f64)], classifier: &dyn Classifier) -> Color32 {
    // Create a deterministic but well-distributed hash
    let hash = class_name.bytes().fold(0u32, |acc, byte| acc.wrapping_mul(31).wrapping_add(byte as u32));
    let cluster_idx = (hash as usize) % cluster_centers.len();
    get_stable_cluster_color(classifier, cluster_idx)
}

fn draw_algorithm_selector(app: &mut MyApp, ui: &mut Ui) {
    ui.heading("Algorithm");
    let mut selected_algo = app.active_algorithm;

    egui::ComboBox::from_label("")
        .selected_text(format!("{:?}", app.active_algorithm))
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut selected_algo, Algorithm::Knn, "k-Nearest Neighbors");
            ui.selectable_value(&mut selected_algo, Algorithm::KMeans, "K-Means Clustering");
            ui.selectable_value(&mut selected_algo, Algorithm::LVQ, "Learning Vector Quantization");
        });

    if selected_algo != app.active_algorithm {
        app.active_algorithm = selected_algo;
        let new_classifier: Box<dyn Classifier> = match selected_algo {
            Algorithm::Knn => Box::new(KnnAdapter::new()),
            Algorithm::KMeans => Box::new(KMeansAdapter::new()),
            Algorithm::LVQ => Box::new(LvqAdapter::new()),
        };
        app.classifier = new_classifier;
        app.classifier.update_data(&app.training_data);
        app.classification_map = None;
        app.recalculate_single_prediction();
        app.recalculate_multi_predictions();
    }
}

fn draw_interaction_mode_controls(app: &mut MyApp, ui: &mut Ui) {
    ui.heading("Interaction Mode");
    ui.horizontal(|ui| {
        ui.radio_value(&mut app.interaction_mode, InteractionMode::Classify, "Classify");
        ui.radio_value(&mut app.interaction_mode, InteractionMode::AddPoint, "Add Data");
        ui.radio_value(&mut app.interaction_mode, InteractionMode::MultiPredict, "Multi-Predict");
    });
    ui.separator();

    match app.interaction_mode {
        InteractionMode::Classify => {
            ui.label("Click on the plot to classify a point.");
            ui.label(format!("Point: [{:.1}, {:.1}]", app.single_prediction_point.x, app.single_prediction_point.y));
            let prediction_text = match &app.last_single_prediction {
                Ok(class) => format!("Prediction: {}", class),
                Err(e) => format!("Error: {}", e),
            };
            // Use the same color mapping as the background for consistency
            let features = array![app.single_prediction_point.x as f64, app.single_prediction_point.y as f64];
            let color = map_prediction_to_color_with_features(features.view(), false, &*app.classifier);
            ui.colored_label(color, prediction_text);
        }
        InteractionMode::AddPoint => {
            ui.label("Click on the plot to add a training point.");
            ui.horizontal(|ui| {
                ui.label("Class:");
                egui::ComboBox::from_label("").selected_text(&app.selected_class_for_add).show_ui(ui, |ui| {
                    for class in &app.available_classes {
                        ui.selectable_value(&mut app.selected_class_for_add, class.clone(), class);
                    }
                });
            });
            ui.horizontal(|ui| {
                ui.label("New:");
                if ui.text_edit_singleline(&mut app.new_class_input).lost_focus() && !app.new_class_input.is_empty() {
                    let new_class = app.new_class_input.trim().to_string();
                    if !app.available_classes.contains(&new_class) {
                        app.available_classes.push(new_class.clone());
                    }
                    app.selected_class_for_add = new_class;
                    app.new_class_input.clear();
                }
            });
            ui.separator();
            if ui.button("Clear All Data").clicked() {
                app.set_training_data(Vec::new());
            }
        }
        InteractionMode::MultiPredict => {
            ui.label("Click on the plot to add prediction points.");
            if ui.button("Clear Prediction Points").clicked() {
                app.prediction_points.clear();
            }
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                for (pos, prediction) in &app.prediction_points {
                    let text = format!("[{:.1}, {:.1}] -> {:?}", pos.x, pos.y, prediction.as_ref().map(|s| s.as_str()).unwrap_or("Error"));
                    // Use the same color mapping as the background for consistency
                    let features = array![pos.x as f64, pos.y as f64];
                    let color = map_prediction_to_color_with_features(features.view(), false, &*app.classifier);
                    ui.colored_label(color, text);
                }
            });
        }
    }
}

fn handle_plot_click(app: &mut MyApp, pos: Pos2) {
    match app.interaction_mode {
        InteractionMode::Classify => {
            app.single_prediction_point = pos;
            app.recalculate_single_prediction();
        }
        InteractionMode::AddPoint => {
            let new_point = DataPoint::new(
                array![pos.x as f64, pos.y as f64],
                app.selected_class_for_add.clone(),
            );
            app.add_training_point(new_point);
        }
        InteractionMode::MultiPredict => {
            let features = array![pos.x as f64, pos.y as f64];
            let prediction = app.classifier.predict(features.view());
            app.prediction_points.push((pos, prediction));
        }
    }
}

fn generate_classification_map(
    classifier: &dyn Classifier,
    ctx: &egui::Context,
) -> (egui::TextureHandle, (usize, usize)) {
    let resolution = (200, 200);
    let mut pixels = Vec::with_capacity(resolution.0 * resolution.1);
    let data_rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(10.0, 10.0));

    for y in 0..resolution.1 {
        for x in 0..resolution.0 {
            let data_x = egui::remap(x as f32, 0.0..=(resolution.0 - 1) as f32, data_rect.x_range());
            let data_y = egui::remap(y as f32, 0.0..=(resolution.1 - 1) as f32, data_rect.y_range());
            let features = array![data_x as f64, data_y as f64];
            // Use direct cluster index access for better color accuracy
            pixels.push(map_prediction_to_color_with_features(features.view(), true, classifier));
        }
    }

    let image = egui::ColorImage { size: [resolution.0, resolution.1], pixels };
    let handle = ctx.load_texture("classification-map", image, egui::TextureOptions::LINEAR);
    (handle, resolution)
}

/// Generate a distinct color for each cluster index using stable color mapping
fn get_cluster_color(cluster_index: usize) -> Color32 {
    let colors = [
        Color32::RED,
        Color32::BLUE,
        Color32::GREEN,
        Color32::YELLOW,
        Color32::from_rgb(255, 0, 255),
        Color32::from_rgb(0, 255, 255),
        Color32::from_rgb(255, 165, 0),
        Color32::from_rgb(128, 0, 128),
        Color32::from_rgb(255, 192, 203),
        Color32::from_rgb(165, 42, 42),
    ];
    colors[cluster_index % colors.len()]
}

/// Get stable color for a cluster using the classifier's stable color mapping
fn get_stable_cluster_color(classifier: &dyn Classifier, cluster_index: usize) -> Color32 {
    let stable_color_index = classifier
        .get_stable_cluster_color(cluster_index)
        .unwrap_or(cluster_index);
    get_cluster_color(stable_color_index)
}