//! Example demonstrating proper error handling with the k-NN classifier.
//!
//! This example shows how to handle various error conditions that can occur
//! when using the k-NN classifier, replacing panics with proper Result-based
//! error handling.

use k_nn::{KnnClassifier, KnnError};
use ndarray::array;
use prec::{DataPoint, L2Dist};

fn main() {
    println!("k-NN Classifier Error Handling Examples");
    println!("=======================================");

    // Example 1: Handle invalid k value
    println!("\n1. Handling invalid k value (k=0):");
    let training_data = vec![
        DataPoint::new(array![1.0, 1.0], "A"),
        DataPoint::new(array![2.0, 2.0], "A"),
    ];

    match KnnClassifier::new(0, training_data.clone(), L2Dist) {
        Ok(_) => println!("   Classifier created successfully"),
        Err(KnnError::InvalidK) => println!("   ✓ Caught expected error: {}", KnnError::InvalidK),
        Err(e) => println!("   ✗ Unexpected error: {}", e),
    }

    // Example 2: Handle empty training set
    println!("\n2. Handling empty training set:");
    let empty_data: Vec<DataPoint<&str, f64>> = vec![];

    match KnnClassifier::new(3, empty_data, L2Dist) {
        Ok(classifier) => {
            println!("   Classifier created with empty training set");
            let test_point = array![1.0, 1.0];

            match classifier.predict(test_point.view()) {
                Ok(label) => println!("   Predicted label: {}", label),
                Err(KnnError::EmptyTrainingSet) => {
                    println!("   ✓ Caught expected error: {}", KnnError::EmptyTrainingSet)
                }
                Err(e) => println!("   ✗ Unexpected error: {}", e),
            }
        }
        Err(e) => println!("   Error creating classifier: {}", e),
    }

    // Example 3: Successful prediction with proper error handling
    println!("\n3. Successful prediction with error handling:");
    let valid_training_data = vec![
        DataPoint::new(array![1.0, 1.0], "Class A"),
        DataPoint::new(array![2.0, 2.0], "Class A"),
        DataPoint::new(array![1.0, 2.0], "Class A"),
        DataPoint::new(array![8.0, 8.0], "Class B"),
        DataPoint::new(array![9.0, 8.0], "Class B"),
        DataPoint::new(array![8.0, 9.0], "Class B"),
    ];

    match KnnClassifier::new(3, valid_training_data, L2Dist) {
        Ok(classifier) => {
            println!("   ✓ Classifier created successfully");

            let test_points = vec![
                (array![2.5, 2.5], "should be Class A"),
                (array![7.5, 8.5], "should be Class B"),
            ];

            for (point, expected) in test_points {
                match classifier.predict(point.view()) {
                    Ok(predicted_label) => {
                        println!(
                            "   ✓ Prediction successful: {} ({})",
                            predicted_label, expected
                        );
                    }
                    Err(e) => {
                        println!("   ✗ Prediction failed: {}", e);
                    }
                }
            }
        }
        Err(e) => println!("   ✗ Failed to create classifier: {}", e),
    }

    // Example 4: Demonstrate error propagation in a function
    println!("\n4. Error propagation in functions:");

    fn classify_with_error_handling() -> Result<String, KnnError> {
        let training_data = vec![
            DataPoint::new(array![1.0], "positive"),
            DataPoint::new(array![-1.0], "negative"),
        ];

        let classifier = KnnClassifier::new(1, training_data, L2Dist)?;
        let test_point = array![0.5];
        let result = classifier.predict(test_point.view())?;
        Ok(result.to_string())
    }

    match classify_with_error_handling() {
        Ok(result) => println!("   ✓ Classification result: {}", result),
        Err(e) => println!("   ✗ Classification failed: {}", e),
    }

    println!("\n5. Error types and their meanings:");
    println!("   - InvalidK: k cannot be zero for a k-NN classifier");
    println!("   - EmptyTrainingSet: Cannot predict with an empty training set");
    println!("   - InvalidDistance: Invalid distance comparison (likely due to NaN values)");
    println!("   - NoMajorityClass: Could not determine a majority class among neighbors");

    println!("\nAll examples completed successfully!");
}
