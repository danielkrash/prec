use ndarray::array;
use prec_helpers::{DataPoint, L2Dist};
use std::collections::HashMap;
use lvq::{fit_lvq2_with_init_and_seed, PrototypeInit};

fn main() {
    // Create test data - same as in the app but smaller for debugging
    let data = vec![
        // Class A - around (2, 3)
        DataPoint::new(array![1.5, 2.5], "A".to_string()),
        DataPoint::new(array![2.0, 3.0], "A".to_string()),
        DataPoint::new(array![2.5, 3.5], "A".to_string()),
        DataPoint::new(array![1.8, 2.8], "A".to_string()),
        DataPoint::new(array![2.2, 3.2], "A".to_string()),
        
        // Class B - around (7, 6)
        DataPoint::new(array![6.5, 5.5], "B".to_string()),
        DataPoint::new(array![7.0, 6.0], "B".to_string()),
        DataPoint::new(array![7.5, 6.5], "B".to_string()),
        DataPoint::new(array![6.8, 5.8], "B".to_string()),
        DataPoint::new(array![7.2, 6.2], "B".to_string()),
        
        // Class C - around (5, 8.5)
        DataPoint::new(array![4.5, 8.0], "C".to_string()),
        DataPoint::new(array![5.0, 8.5], "C".to_string()),
        DataPoint::new(array![5.5, 9.0], "C".to_string()),
        DataPoint::new(array![4.8, 8.3], "C".to_string()),
        DataPoint::new(array![5.2, 8.7], "C".to_string()),
    ];

    let mut n_prototypes_per_class = HashMap::new();
    n_prototypes_per_class.insert("A".to_string(), 2);
    n_prototypes_per_class.insert("B".to_string(), 2);
    n_prototypes_per_class.insert("C".to_string(), 2);

    println!("=== Testing LVQ2 Algorithm ===");
    println!("Data points: {}", data.len());
    println!("Prototypes per class: 2");
    
    // First, let's test with a simple configuration and see the initial vs final prototypes
    println!("\n=== Testing Initial vs Final Prototype Positions ===");
    
    // Test with Random initialization to see movement
    println!("\n--- Random Initialization ---");
    let result = fit_lvq2_with_init_and_seed(
        &data,
        &n_prototypes_per_class,
        0.3,  // learning rate
        100,  // epochs
        L2Dist,
        0.95, // decay
        0.3,  // window
        PrototypeInit::Random,
        42,   // Fixed seed
    );
    
    match result {
        Ok(model) => {
            println!("Training SUCCESS!");
            println!("Final prototypes after {} epochs:", 100);
            for (i, proto) in model.codebook.iter().enumerate() {
                println!("  Proto {}: [{:.3}, {:.3}] -> '{}'", 
                    i, proto.features[0], proto.features[1], proto.label);
            }
            
            // Test prediction accuracy
            println!("\nPrediction tests:");
            let test_points = vec![
                (array![2.0, 3.0], "A"),
                (array![7.0, 6.0], "B"), 
                (array![5.0, 8.5], "C"),
                (array![4.0, 5.0], "?"),
            ];
            
            for (point, expected) in test_points {
                let prediction = model.predict(point.view());
                println!("  [{:.1}, {:.1}] -> '{}' (expected: '{}')", 
                    point[0], point[1], prediction, expected);
            }
        }
        Err(e) => {
            println!("Training FAILED: {}", e);
        }
    }
    
    // Test with ClassMean initialization
    println!("\n--- ClassMean Initialization ---");
    let result_classmean = fit_lvq2_with_init_and_seed(
        &data,
        &{
            let mut map = HashMap::new();
            map.insert("A".to_string(), 1);
            map.insert("B".to_string(), 1);
            map.insert("C".to_string(), 1);
            map
        },
        0.3,
        100,
        L2Dist,
        0.95,
        0.3,
        PrototypeInit::ClassMean,
        42,
    );
    
    match result_classmean {
        Ok(model) => {
            println!("ClassMean Training SUCCESS!");
            println!("Final prototypes (started from class means):");
            for (i, proto) in model.codebook.iter().enumerate() {
                println!("  Proto {}: [{:.3}, {:.3}] -> '{}'", 
                    i, proto.features[0], proto.features[1], proto.label);
            }
        }
        Err(e) => {
            println!("ClassMean Training FAILED: {}", e);
        }
    }
    
    // Test different learning rates to see if prototypes move more
    println!("\n=== Testing Different Learning Rates ===");
    let learning_rates = vec![0.1, 0.3, 0.5, 0.8];
    
    for lr in learning_rates {
        println!("\n--- Learning Rate: {} ---", lr);
        let result = fit_lvq2_with_init_and_seed(
            &data,
            &n_prototypes_per_class,
            lr,
            50,
            L2Dist,
            0.98,
            0.3,
            PrototypeInit::Random,
            42,
        );
        
        match result {
            Ok(model) => {
                println!("  SUCCESS! Final prototypes:");
                for (i, proto) in model.codebook.iter().enumerate() {
                    println!("    Proto {}: [{:.3}, {:.3}] -> '{}'", 
                        i, proto.features[0], proto.features[1], proto.label);
                }
            }
            Err(e) => {
                println!("  FAILED: {}", e);
            }
        }
    }
    
    println!("\n=== LVQ2 Algorithm Explanation ===");
    println!("In LVQ2, prototypes should move in the following ways:");
    println!("1. If closest and second-closest prototypes have DIFFERENT classes:");
    println!("   - Move the correct one CLOSER to the data point");
    println!("   - Move the incorrect one AWAY from the data point");
    println!("2. If they have the SAME class, no update happens");
    println!("3. Updates only happen if both prototypes are within the 'window'");
    println!("4. Learning rate decreases over time (with decay)");
    println!("\nIf prototypes aren't moving much, it could mean:");
    println!("- Learning rate is too low");
    println!("- Window parameter is too restrictive");
    println!("- Data is already well-separated");
    println!("- Not enough epochs for convergence");
}
