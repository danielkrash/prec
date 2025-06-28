# k-Nearest Neighbors (k-NN) Classifier

A robust implementation of the k-Nearest Neighbors algorithm with proper error handling.

## Features

- **Type-safe**: Generic over label types, float types, and distance metrics
- **Robust error handling**: No panics - all errors are handled gracefully via `Result` types
- **Flexible distance metrics**: Support for L1, L2, L∞, and custom Lp distances
- **Memory efficient**: Uses relative distances to avoid unnecessary square root calculations

## Error Handling

This implementation uses proper Rust error handling patterns instead of panics. All functions that can fail return `Result` types:

### Error Types

- `KnnError::InvalidK` - k cannot be zero for a k-NN classifier
- `KnnError::EmptyTrainingSet` - Cannot predict with an empty training set
- `KnnError::InvalidDistance` - Invalid distance comparison (likely due to NaN values in data)
- `KnnError::NoMajorityClass` - Could not determine a majority class among neighbors

### Basic Usage

```rust
use k_nn::{KnnClassifier, KnnError};
use ndarray::array;
use prec::{DataPoint, L2Dist};

// Create training data
let training_data = vec![
    DataPoint::new(array![1.0, 1.0], "Class A"),
    DataPoint::new(array![2.0, 2.0], "Class A"),
    DataPoint::new(array![8.0, 8.0], "Class B"),
    DataPoint::new(array![9.0, 8.0], "Class B"),
];

// Create classifier with error handling
let classifier = match KnnClassifier::new(3, training_data, L2Dist) {
    Ok(classifier) => classifier,
    Err(KnnError::InvalidK) => {
        eprintln!("Error: k must be greater than 0");
        return;
    }
    Err(e) => {
        eprintln!("Unexpected error: {}", e);
        return;
    }
};

// Make predictions with error handling
let test_point = array![2.5, 2.5];
match classifier.predict(test_point.view()) {
    Ok(predicted_label) => println!("Predicted: {}", predicted_label),
    Err(KnnError::EmptyTrainingSet) => eprintln!("No training data available"),
    Err(KnnError::InvalidDistance) => eprintln!("Invalid data (contains NaN)"),
    Err(e) => eprintln!("Prediction error: {}", e),
}
```

### Error Propagation

The error types implement the standard `Error` trait, making them compatible with error propagation using the `?` operator:

```rust
fn classify_point() -> Result<String, KnnError> {
    let training_data = vec![
        DataPoint::new(array![1.0], "positive"),
        DataPoint::new(array![-1.0], "negative"),
    ];
    
    let classifier = KnnClassifier::new(1, training_data, L2Dist)?;
    let test_point = array![0.5];
    let result = classifier.predict(test_point.view())?;
    Ok(result.to_string())
}
```

## Examples

Run the error handling example to see all error cases in action:

```bash
cargo run --example error_handling -p k-nn
```

## Migration from Panic-based Code

If you were previously using a version that used panics, here's how to migrate:

### Before (with panics)
```rust
// This would panic on invalid input
let classifier = KnnClassifier::new(0, data, distance);
let result = classifier.predict(features);
```

### After (with proper error handling)
```rust
// This returns a Result and handles errors gracefully
let classifier = KnnClassifier::new(0, data, distance)?;
let result = classifier.predict(features)?;
```

## Testing

All error conditions are thoroughly tested:

```bash
cargo test -p k-nn
```

The test suite includes:
- Valid classifications
- Error handling for invalid k values
- Error handling for empty training sets
- Edge cases like k larger than dataset size