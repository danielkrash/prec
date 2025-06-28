// This is a simple example showing how to use the prec library
use prec::{DataPoint, L2Dist, Distance};
use ndarray::array;

fn main() {
    println!("prec library example");
    
    // Create a simple data point
    let point = DataPoint::new(array![1.0, 2.0], "example".to_string());
    println!("Created data point: {:?}", point);
    
    // Example of distance calculation
    let dist = L2Dist;
    let a = array![0.0, 0.0];
    let b = array![3.0, 4.0];
    let distance = dist.distance(a.view(), b.view());
    println!("Distance between {:?} and {:?}: {}", a, b, distance);
}
