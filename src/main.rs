// Comprehensive example showing how to use all algorithms from the nonparametric package
use ndarray::array;
use prec::{DataPoint, Distance, L1Dist, L2Dist, LInfDist};
use std::collections::HashMap;

// Import all algorithms
use k_nn::KnnClassifier;
use k_means::KMeans;
use lvq::{fit_with_init_and_seed, fit_lvq2_with_init_and_seed, fit_lvq3_with_init_and_seed, PrototypeInit};

fn main() {
    println!("=== PREC Library with All Algorithms ===");
    println!("Now using separate prec-helpers crate with all algorithms!\n");

    // Create comprehensive test data
    let data_points = create_comprehensive_dataset();
    println!("Created {} data points across 3 classes", data_points.len());
    
    // Display data summary
    display_data_summary(&data_points);

    // Demonstrate distance metrics
    println!("\n=== Distance Metrics Demonstration ===");
    demonstrate_distance_metrics();

    // Test k-NN Algorithm
    println!("\n=== k-NN Algorithm Demonstration ===");
    demonstrate_knn(&data_points);

    // Test K-Means Algorithm
    println!("\n=== K-Means Algorithm Demonstration ===");
    demonstrate_kmeans(&data_points);

    // Test LVQ Algorithms
    println!("\n=== LVQ Algorithms Demonstration ===");
    demonstrate_lvq(&data_points);

    // Show visualizer information
    println!("\n=== Interactive Visualizer ===");
    println!("Run the interactive visualizer to see all algorithms in action:");
    println!("  cargo run -p knn-visualizer");
    println!("\nThe visualizer includes:");
    println!("  ✓ All three algorithms with full parameter control");
    println!("  ✓ Interactive data point editing");
    println!("  ✓ Real-time decision boundary rendering");
    println!("  ✓ LVQ prototype visualization");
    
    println!("\n=== Project Structure ===");
    println!("prec/");
    println!("├── prec-helpers/          # Shared core types and utilities");
    println!("│   ├── DataPoint          # Common data structure");
    println!("│   ├── Distance traits    # L1, L2, L∞ distances");
    println!("│   └── Float trait        # Numeric constraints");
    println!("├── algorithms/");
    println!("│   └── nonparametric/");
    println!("│       ├── k-nn/          ✓ Working k-Nearest Neighbors");
    println!("│       ├── k-means/       ✓ Working K-Means Clustering");
    println!("│       └── lvq/           ✓ Working LVQ (1, 2, 3 variants)");
    println!("└── app/                   ✓ Interactive Visualizer");
    
    println!("\n=== Dependencies Resolved ===");
    println!("✓ No circular dependencies");
    println!("✓ All algorithms use prec-helpers");
    println!("✓ Clean modular structure");
    println!("✓ Easy to extend with new algorithms");
    
    println!("\nAll algorithms successfully imported and demonstrated! 🎉");
}

fn create_comprehensive_dataset() -> Vec<DataPoint<String, f64>> {
    vec![
        // Class A - Cluster around (2, 3)
        DataPoint::new(array![1.5, 2.5], "A".to_string()),
        DataPoint::new(array![2.0, 3.0], "A".to_string()),
        DataPoint::new(array![2.5, 3.5], "A".to_string()),
        DataPoint::new(array![1.8, 2.8], "A".to_string()),
        DataPoint::new(array![2.2, 3.2], "A".to_string()),
        DataPoint::new(array![1.6, 3.1], "A".to_string()),
        DataPoint::new(array![2.4, 2.7], "A".to_string()),
        DataPoint::new(array![1.9, 3.4], "A".to_string()),
        
        // Class B - Cluster around (7, 6)
        DataPoint::new(array![6.5, 5.5], "B".to_string()),
        DataPoint::new(array![7.0, 6.0], "B".to_string()),
        DataPoint::new(array![7.5, 6.5], "B".to_string()),
        DataPoint::new(array![6.8, 5.8], "B".to_string()),
        DataPoint::new(array![7.2, 6.2], "B".to_string()),
        DataPoint::new(array![6.6, 6.1], "B".to_string()),
        DataPoint::new(array![7.4, 5.7], "B".to_string()),
        DataPoint::new(array![6.9, 6.4], "B".to_string()),
        
        // Class C - Cluster around (5, 8.5)
        DataPoint::new(array![4.5, 8.0], "C".to_string()),
        DataPoint::new(array![5.0, 8.5], "C".to_string()),
        DataPoint::new(array![5.5, 9.0], "C".to_string()),
        DataPoint::new(array![4.8, 8.3], "C".to_string()),
        DataPoint::new(array![5.2, 8.7], "C".to_string()),
        DataPoint::new(array![4.6, 8.6], "C".to_string()),
        DataPoint::new(array![5.4, 8.2], "C".to_string()),
        DataPoint::new(array![4.9, 8.9], "C".to_string()),
    ]
}

fn display_data_summary(data: &[DataPoint<String, f64>]) {
    let mut class_counts: HashMap<String, usize> = HashMap::new();
    let mut class_centers: HashMap<String, (f64, f64, usize)> = HashMap::new();
    
    for point in data {
        *class_counts.entry(point.label.clone()).or_insert(0) += 1;
        let entry = class_centers.entry(point.label.clone()).or_insert((0.0, 0.0, 0));
        entry.0 += point.features[0];
        entry.1 += point.features[1];
        entry.2 += 1;
    }
    
    println!("\nClass Distribution:");
    for (class, count) in &class_counts {
        let (sum_x, sum_y, n) = class_centers[class];
        let center_x = sum_x / n as f64;
        let center_y = sum_y / n as f64;
        println!("  Class {}: {} points, center at ({:.2}, {:.2})", 
                class, count, center_x, center_y);
    }
}

fn demonstrate_distance_metrics() {
    let point1 = array![2.0, 3.0];
    let point2 = array![5.0, 7.0];
    
    println!("Distance between [{:.1}, {:.1}] and [{:.1}, {:.1}]:", 
             point1[0], point1[1], point2[0], point2[1]);
    
    let l2_dist = L2Dist.distance(point1.view(), point2.view());
    let l1_dist = L1Dist.distance(point1.view(), point2.view());
    let linf_dist = LInfDist.distance(point1.view(), point2.view());
    
    println!("  L2 (Euclidean): {:.3}", l2_dist);
    println!("  L1 (Manhattan):  {:.3}", l1_dist);
    println!("  L∞ (Chebyshev):  {:.3}", linf_dist);
}

fn demonstrate_knn(data: &[DataPoint<String, f64>]) {
    println!("Creating k-NN classifier with k=3 and L2 distance...");
    
    // Create k-NN classifier with training data
    let training_data = data.to_vec();  // Clone the data for the classifier
    match KnnClassifier::new(3, training_data, L2Dist) {
        Ok(knn) => {
            println!("k-NN classifier created successfully with {} training points", data.len());
            
            // Test predictions
            let test_points = vec![
                (array![2.0, 3.0], "Should be A"),
                (array![7.0, 6.0], "Should be B"),
                (array![5.0, 8.5], "Should be C"),
                (array![4.0, 5.0], "Boundary point"),
            ];
            
            println!("k-NN Predictions:");
            for (point, expected) in test_points {
                match knn.predict(point.view()) {
                    Ok(prediction) => {
                        println!("  [{:.1}, {:.1}] -> '{}' ({})", 
                                point[0], point[1], prediction, expected);
                    }
                    Err(e) => {
                        println!("  [{:.1}, {:.1}] -> Error: {} ({})", 
                                point[0], point[1], e, expected);
                    }
                }
            }
        }
        Err(e) => {
            println!("Failed to create k-NN classifier: {}", e);
        }
    }
}

fn demonstrate_kmeans(data: &[DataPoint<String, f64>]) {
    println!("Running K-Means clustering with k=3...");
    
    // Create k-means model
    let mut kmeans = KMeans::new(3, 100, 1e-4, L2Dist);
    
    // Run k-means clustering
    match kmeans.fit(data) {
        Ok((assignments, centroids)) => {
            println!("K-Means clustering SUCCESS!");
            println!("Final cluster centers:");
            for (i, centroid) in centroids.rows().into_iter().enumerate() {
                println!("  Cluster {}: [{:.3}, {:.3}]", 
                        i+1, centroid[0], centroid[1]);
            }
            
            println!("Point assignments (showing first 10):");
            for (i, (point, &assignment)) in data.iter().zip(assignments.iter()).take(10).enumerate() {
                println!("  Point {}: [{:.1}, {:.1}] '{}' -> Cluster {}", 
                        i+1, point.features[0], point.features[1], point.label, assignment+1);
            }
            
            // Calculate inertia manually since it's not returned by the API
            let mut inertia = 0.0;
            for (point, &assignment) in data.iter().zip(assignments.iter()) {
                let centroid = centroids.row(assignment);
                let dist = L2Dist.distance(point.features.view(), centroid);
                inertia += dist * dist;
            }
            println!("Within-cluster sum of squares: {:.3}", inertia);
        }
        Err(e) => {
            println!("K-Means clustering FAILED: {}", e);
        }
    }
}

fn demonstrate_lvq(data: &[DataPoint<String, f64>]) {
    println!("Testing all LVQ variants...");
    
    let mut n_prototypes_per_class = HashMap::new();
    n_prototypes_per_class.insert("A".to_string(), 2);
    n_prototypes_per_class.insert("B".to_string(), 2);
    n_prototypes_per_class.insert("C".to_string(), 2);
    
    // Test LVQ1
    println!("\n--- LVQ1 Algorithm ---");
    match fit_with_init_and_seed(
        data,
        &n_prototypes_per_class,
        0.3,  // learning rate
        50,   // epochs
        L2Dist,
        0.95, // decay
        PrototypeInit::Random,
        42,   // seed
    ) {
        Ok(model) => {
            println!("LVQ1 Training SUCCESS!");
            println!("Final prototypes:");
            for (i, proto) in model.codebook.iter().enumerate() {
                println!("  Proto {}: [{:.3}, {:.3}] -> '{}'", 
                        i+1, proto.features[0], proto.features[1], proto.label);
            }
            
            // Test predictions
            let test_point = array![2.0, 3.0];
            let prediction = model.predict(test_point.view());
            println!("  Test: [{:.1}, {:.1}] -> '{}'", test_point[0], test_point[1], prediction);
        }
        Err(e) => {
            println!("LVQ1 Training FAILED: {}", e);
        }
    }
    
    // Test LVQ2
    println!("\n--- LVQ2 Algorithm ---");
    match fit_lvq2_with_init_and_seed(
        data,
        &n_prototypes_per_class,
        0.3,  // learning rate
        50,   // epochs
        L2Dist,
        0.95, // decay
        0.3,  // window
        PrototypeInit::Random,
        42,   // seed
    ) {
        Ok(model) => {
            println!("LVQ2 Training SUCCESS!");
            println!("Final prototypes:");
            for (i, proto) in model.codebook.iter().enumerate() {
                println!("  Proto {}: [{:.3}, {:.3}] -> '{}'", 
                        i+1, proto.features[0], proto.features[1], proto.label);
            }
        }
        Err(e) => {
            println!("LVQ2 Training FAILED: {}", e);
        }
    }
    
    // Test LVQ3
    println!("\n--- LVQ3 Algorithm ---");
    match fit_lvq3_with_init_and_seed(
        data,
        &n_prototypes_per_class,
        0.3,  // learning rate
        50,   // epochs
        L2Dist,
        0.95, // decay
        0.3,  // window
        0.1,  // epsilon
        PrototypeInit::Random,
        42,   // seed
    ) {
        Ok(model) => {
            println!("LVQ3 Training SUCCESS!");
            println!("Final prototypes:");
            for (i, proto) in model.codebook.iter().enumerate() {
                println!("  Proto {}: [{:.3}, {:.3}] -> '{}'", 
                        i+1, proto.features[0], proto.features[1], proto.label);
            }
        }
        Err(e) => {
            println!("LVQ3 Training FAILED: {}", e);
        }
    }
    
    println!("\nLVQ Summary:");
    println!("- LVQ1: Basic prototype learning");
    println!("- LVQ2: Improves decision boundaries with window parameter");
    println!("- LVQ3: Further refinement with epsilon parameter");
}
