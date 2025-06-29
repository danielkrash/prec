use k_means::KMeans;
use ndarray::{Array1, ArrayView1};
use prec_helpers::{DataPoint, Distance, Float};
use rand::prelude::SliceRandom;
use rand::seq::index;
use rand::{Rng, RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

/// Errors that can occur during LVQ training.
#[derive(Debug, Clone, PartialEq)]
pub enum LvqError {
    /// The training data is empty.
    EmptyDataSet,
    /// Not enough data points to initialize the requested number of prototypes for a class.
    NotEnoughDataForPrototypes(String),
    /// The prototype configuration references a class not present in the data.
    UnknownClassInConfig,
    /// Invalid configuration (e.g., requesting more than one prototype with ClassMean)
    InvalidConfig(String),
}

impl Display for LvqError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            LvqError::EmptyDataSet => write!(f, "Training data is empty"),
            LvqError::NotEnoughDataForPrototypes(cls) => write!(
                f,
                "Not enough data points to initialize requested number of prototypes for class {}",
                cls
            ),
            LvqError::UnknownClassInConfig => write!(f, "Unknown class in prototype configuration"),
            LvqError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
        }
    }
}
impl Error for LvqError {}

/// Prototype initialization strategies for LVQ algorithms.
#[derive(Debug, Clone)]
pub enum PrototypeInit<F: Float> {
    /// Randomly sample data points from each class (default/original LVQ).
    Random,
    /// Use the mean of each class as the prototype (if only one prototype per class).
    ClassMean,
    /// Use K-means clustering within each class to initialize prototypes.
    KMeans { max_iter: u32, tol: F },
}

/// The trained LVQ model, containing the finalized codebook.
#[derive(Debug, Clone)]
pub struct Lvq<L, F, D>
where
    L: Clone + Eq + Hash + Debug,
    F: Float,
    D: Distance<F>,
{
    /// The learned prototypes (codebook vectors).
    pub codebook: Vec<DataPoint<L, F>>,
    distance: D,
}

impl<L, F, D> Lvq<L, F, D>
where
    L: Clone + Eq + Hash + Debug,
    F: Float,
    D: Distance<F>,
{
    /// Predicts the class label for a new data point.
    ///
    /// Finds the single closest prototype in the codebook and returns its label (by value).
    pub fn predict(&self, features: ArrayView1<F>) -> L {
        let (best_index, _) = self
            .codebook
            .iter()
            .enumerate()
            .map(|(i, proto)| (i, self.distance.rdistance(features, proto.features.view())))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .expect("Codebook should not be empty");

        self.codebook[best_index].label.clone()
    }
}

/// Helper function to initialize prototypes according to the chosen strategy.
pub fn initialize_prototypes<L, F, D, R>(
    data: &[DataPoint<L, F>],
    n_prototypes_per_class: &HashMap<L, usize>,
    distance: &D,
    init: PrototypeInit<F>,
    rng: &mut R,
) -> Result<Vec<DataPoint<L, F>>, LvqError>
where
    L: Clone + Eq + Hash + Debug + Ord,
    F: Float,
    D: Distance<F> + Clone,
    R: RngCore + Rng,
{
    if data.is_empty() {
        return Err(LvqError::EmptyDataSet);
    }

    let data_by_class = group_data_by_class(data)?;
    let n_features = data[0].features.len();
    let mut labels: Vec<_> = n_prototypes_per_class.keys().cloned().collect();
    labels.sort();
    let mut codebook = Vec::new();

    for label in labels {
        let &n_proto = n_prototypes_per_class
            .get(&label)
            .ok_or(LvqError::UnknownClassInConfig)?;
        let indices = data_by_class.get(&label).unwrap();

        if n_proto > indices.len() {
            return Err(LvqError::NotEnoughDataForPrototypes(format!("{:?}", label)));
        }

        match &init {
            PrototypeInit::Random => {
                let picks = index::sample(rng, indices.len(), n_proto).into_vec();
                for &i in &picks {
                    codebook.push(data[indices[i]].clone());
                }
            }
            PrototypeInit::ClassMean => {
                if n_proto != 1 {
                    return Err(LvqError::InvalidConfig(
                        "ClassMean supports exactly one prototype per class".into(),
                    ));
                }
                let mut sum = Array1::zeros(n_features);
                for &i in indices {
                    sum += &data[i].features;
                }
                let mean = sum / F::from(indices.len()).unwrap();
                codebook.push(DataPoint::new(mean, label.clone()));
            }
            PrototypeInit::KMeans { max_iter, tol } => {
                // Use K-means clustering within each class to initialize prototypes
                if n_proto == 1 {
                    // If only one prototype per class, use class mean for efficiency
                    let mut sum = Array1::zeros(n_features);
                    for &i in indices {
                        sum += &data[i].features;
                    }
                    let mean = sum / F::from(indices.len()).unwrap();
                    codebook.push(DataPoint::new(mean, label.clone()));
                } else {
                    // Collect class data points
                    let class_data: Vec<DataPoint<L, F>> =
                        indices.iter().map(|&i| data[i].clone()).collect();

                    // Run K-means on this class's data to get n_proto centroids
                    let mut kmeans = KMeans::new(n_proto, *max_iter, *tol, distance.clone());
                    match kmeans.fit(&class_data) {
                        Ok((_assignments, centroids)) => {
                            // Convert centroids to DataPoints with the correct label
                            for i in 0..n_proto {
                                if i < centroids.nrows() {
                                    let centroid_features = centroids.row(i).to_owned();
                                    codebook.push(DataPoint::new(centroid_features, label.clone()));
                                }
                            }
                        }
                        Err(_) => {
                            // K-means failed, fall back to random sampling
                            println!(
                                "K-means failed for class {:?}, falling back to random initialization",
                                label
                            );
                            let picks = index::sample(rng, indices.len(), n_proto).into_vec();
                            for &i in &picks {
                                codebook.push(data[indices[i]].clone());
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(codebook)
}

/// Trains an LVQ1 model with random initialization (backward compatibility).
///
/// # Arguments
///
/// * `data`: The labeled training data.
/// * `n_prototypes_per_class`: A map from class label to the number of prototypes for that class.
/// * `learning_rate`: The initial learning rate (alpha).
/// * `epochs`: The number of times to iterate over the training data.
/// * `distance`: The distance metric to use.
/// * `decay`: The multiplicative decay factor for the learning rate (e.g., 0.99 for 1% decay per epoch).
///
/// # Returns
/// A trained `Lvq` model ready for prediction.
pub fn fit<L, F, D>(
    data: &[DataPoint<L, F>],
    n_prototypes_per_class: &HashMap<L, usize>,
    learning_rate: F,
    epochs: u32,
    distance: D,
    decay: F,
) -> Result<Lvq<L, F, D>, LvqError>
where
    L: Clone + Eq + Hash + Debug + Ord,
    F: Float,
    D: Distance<F> + Clone,
{
    // Use a random seed for backward compatibility
    fit_with_seed(
        data,
        n_prototypes_per_class,
        learning_rate,
        epochs,
        distance,
        decay,
        rand::random(),
    )
}
/// Trains an LVQ1 model with a specific seed for reproducibility.
///
/// # Arguments
///
/// * `data`: The labeled training data.
/// * `n_prototypes_per_class`: A map from class label to the number of prototypes for that class.
/// * `learning_rate`: The initial learning rate (alpha).
/// * `epochs`: The number of times to iterate over the training data.
/// * `distance`: The distance metric to use.
/// * `decay`: The multiplicative decay factor for the learning rate (e.g., 0.99 for 1% decay per epoch).
/// * `seed`: The seed for the random number generator for reproducible results.
///
/// # Returns
/// A trained `Lvq` model ready for prediction.
pub fn fit_with_seed<L, F, D>(
    data: &[DataPoint<L, F>],
    n_prototypes_per_class: &HashMap<L, usize>,
    learning_rate: F,
    epochs: u32,
    distance: D,
    decay: F,
    seed: u64,
) -> Result<Lvq<L, F, D>, LvqError>
where
    L: Clone + Eq + Hash + Debug + Ord,
    F: Float,
    D: Distance<F> + Clone,
{
    fit_with_init_and_seed(
        data,
        n_prototypes_per_class,
        learning_rate,
        epochs,
        distance,
        decay,
        PrototypeInit::Random,
        seed,
    )
}

/// Trains an LVQ1 model with custom initialization and a specific seed for reproducibility.
///
/// # Arguments
///
/// * `data`: The labeled training data.
/// * `n_prototypes_per_class`: A map from class label to the number of prototypes for that class.
/// * `learning_rate`: The initial learning rate (alpha).
/// * `epochs`: The number of times to iterate over the training data.
/// * `distance`: The distance metric to use.
/// * `decay`: The multiplicative decay factor for the learning rate (e.g., 0.99 for 1% decay per epoch).
/// * `init`: The prototype initialization strategy.
/// * `seed`: The seed for the random number generator for reproducible results.
///
/// # Returns
/// A trained `Lvq` model ready for prediction.
pub fn fit_with_init_and_seed<L, F, D>(
    data: &[DataPoint<L, F>],
    n_prototypes_per_class: &HashMap<L, usize>,
    learning_rate: F,
    epochs: u32,
    distance: D,
    decay: F,
    init: PrototypeInit<F>,
    seed: u64,
) -> Result<Lvq<L, F, D>, LvqError>
where
    L: Clone + Eq + Hash + Debug + Ord,
    F: Float,
    D: Distance<F> + Clone,
{
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    fit_with_init_and_xoshiro_rng(
        data,
        n_prototypes_per_class,
        learning_rate,
        epochs,
        distance,
        decay,
        init,
        &mut rng,
    )
}

/// Trains an LVQ1 model with Xoshiro256PlusPlus RNG (internal function).
fn fit_with_init_and_xoshiro_rng<L, F, D>(
    data: &[DataPoint<L, F>],
    n_prototypes_per_class: &HashMap<L, usize>,
    learning_rate: F,
    epochs: u32,
    distance: D,
    decay: F,
    init: PrototypeInit<F>,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<Lvq<L, F, D>, LvqError>
where
    L: Clone + Eq + Hash + Debug + Ord,
    F: Float,
    D: Distance<F> + Clone,
{
    if data.is_empty() {
        return Err(LvqError::EmptyDataSet);
    }

    // --- 1. Initialization using specified strategy ---
    let mut codebook = initialize_prototypes(data, n_prototypes_per_class, &distance, init, rng)?;

    // --- 2. Training Loop ---
    let mut current_learning_rate = learning_rate;
    let mut training_indices: Vec<usize> = (0..data.len()).collect();

    for _epoch in 0..epochs {
        training_indices.shuffle(rng);

        for &data_idx in &training_indices {
            let data_point = &data[data_idx];

            // a. Find the Best Matching Unit (BMU)
            let (bmu_index, _) = codebook
                .iter()
                .enumerate()
                .map(|(i, proto)| {
                    (
                        i,
                        distance.rdistance(data_point.features.view(), proto.features.view()),
                    )
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                .unwrap();

            let bmu = &mut codebook[bmu_index];
            let update_vec = &data_point.features - &bmu.features;

            // b. Update the BMU's position based on label match
            if data_point.label == bmu.label {
                // Labels match: move prototype closer
                bmu.features += &(update_vec * current_learning_rate);
            } else {
                // Labels don't match: push prototype away
                bmu.features -= &(update_vec * current_learning_rate);
            }
        }

        // c. Decay the learning rate for the next epoch
        current_learning_rate = current_learning_rate * decay;
    }

    Ok(Lvq { codebook, distance })
}

/// Trains an LVQ1 model with seeded RNG.
///
/// # Arguments
///
/// * `data`: The labeled training data.
/// * `n_prototypes_per_class`: A map from class label to the number of prototypes for that class.
/// * `learning_rate`: The initial learning rate (alpha).
/// * `epochs`: The number of times to iterate over the training data.
/// * `distance`: The distance metric to use.
/// * `decay`: The multiplicative decay factor for the learning rate (e.g., 0.99 for 1% decay per epoch).
/// * `init`: The prototype initialization strategy.
/// * `rng`: A mutable reference to a random number generator.
///
/// # Returns
/// A trained `Lvq` model ready for prediction.
pub fn fit_with_init_and_rng<L, F, D, R>(
    data: &[DataPoint<L, F>],
    n_prototypes_per_class: &HashMap<L, usize>,
    learning_rate: F,
    epochs: u32,
    distance: D,
    decay: F,
    init: PrototypeInit<F>,
    rng: &mut R,
) -> Result<Lvq<L, F, D>, LvqError>
where
    L: Clone + Eq + Hash + Debug + Ord,
    F: Float,
    D: Distance<F> + Clone,
    R: RngCore + Rng,
{
    if data.is_empty() {
        return Err(LvqError::EmptyDataSet);
    }

    // --- 1. Initialization using specified strategy ---
    let mut codebook = initialize_prototypes(data, n_prototypes_per_class, &distance, init, rng)?;

    // --- 2. Training Loop ---
    let mut current_learning_rate = learning_rate;
    let mut training_indices: Vec<usize> = (0..data.len()).collect();

    for _epoch in 0..epochs {
        training_indices.shuffle(rng);

        for &data_idx in &training_indices {
            let data_point = &data[data_idx];

            // a. Find the Best Matching Unit (BMU)
            let (bmu_index, _) = codebook
                .iter()
                .enumerate()
                .map(|(i, proto)| {
                    (
                        i,
                        distance.rdistance(data_point.features.view(), proto.features.view()),
                    )
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                .unwrap();

            let bmu = &mut codebook[bmu_index];
            let update_vec = &data_point.features - &bmu.features;

            // b. Update the BMU's position based on label match
            if data_point.label == bmu.label {
                // Labels match: move prototype closer
                bmu.features += &(update_vec * current_learning_rate);
            } else {
                // Labels don't match: push prototype away
                bmu.features -= &(update_vec * current_learning_rate);
            }
        }

        // c. Decay the learning rate for the next epoch
        current_learning_rate = current_learning_rate * decay;
    }

    Ok(Lvq { codebook, distance })
}

/// Trains an LVQ2 model.
///
/// # Arguments
///
/// * `data`: The labeled training data.
/// * `n_prototypes_per_class`: A map from class label to the number of prototypes for that class.
/// * `learning_rate`: The initial learning rate (alpha).
/// * `epochs`: The number of times to iterate over the training data.
/// * `distance`: The distance metric to use.
/// * `decay`: The multiplicative decay factor for the learning rate (e.g., 0.99 for 1% decay per epoch).
/// * `window`: The window width parameter (e.g., 0.3).
///
/// # Returns
/// A trained `Lvq` model ready for prediction.
pub fn fit_lvq2<L, F, D>(
    data: &[DataPoint<L, F>],
    n_prototypes_per_class: &HashMap<L, usize>,
    learning_rate: F,
    epochs: u32,
    distance: D,
    decay: F,
    window: F,
) -> Result<Lvq<L, F, D>, LvqError>
where
    L: Clone + Eq + Hash + Debug + Ord,
    F: Float,
    D: Distance<F> + Clone,
{
    fit_lvq2_with_seed(
        data,
        n_prototypes_per_class,
        learning_rate,
        epochs,
        distance,
        decay,
        window,
        rand::random(),
    )
}

/// Trains an LVQ2 model with a specific seed for reproducibility.
///
/// # Arguments
///
/// * `data`: The labeled training data.
/// * `n_prototypes_per_class`: A map from class label to the number of prototypes for that class.
/// * `learning_rate`: The initial learning rate (alpha).
/// * `epochs`: The number of times to iterate over the training data.
/// * `distance`: The distance metric to use.
/// * `decay`: The multiplicative decay factor for the learning rate (e.g., 0.99 for 1% decay per epoch).
/// * `window`: The window width parameter (e.g., 0.3).
/// * `seed`: The seed for the random number generator for reproducible results.
///
/// # Returns
/// A trained `Lvq` model ready for prediction.
pub fn fit_lvq2_with_seed<L, F, D>(
    data: &[DataPoint<L, F>],
    n_prototypes_per_class: &HashMap<L, usize>,
    learning_rate: F,
    epochs: u32,
    distance: D,
    decay: F,
    window: F,
    seed: u64,
) -> Result<Lvq<L, F, D>, LvqError>
where
    L: Clone + Eq + Hash + Debug + Ord,
    F: Float,
    D: Distance<F> + Clone,
{
    fit_lvq2_with_init_and_seed(
        data,
        n_prototypes_per_class,
        learning_rate,
        epochs,
        distance,
        decay,
        window,
        PrototypeInit::Random,
        seed,
    )
}

/// Trains an LVQ2 model with custom initialization and a specific seed for reproducibility.
///
/// # Arguments
///
/// * `data`: The labeled training data.
/// * `n_prototypes_per_class`: A map from class label to the number of prototypes for that class.
/// * `learning_rate`: The initial learning rate (alpha).
/// * `epochs`: The number of times to iterate over the training data.
/// * `distance`: The distance metric to use.
/// * `decay`: The multiplicative decay factor for the learning rate (e.g., 0.99 for 1% decay per epoch).
/// * `window`: The window width parameter (e.g., 0.3).
/// * `init`: The prototype initialization strategy.
/// * `seed`: The seed for the random number generator for reproducible results.
///
/// # Returns
/// A trained `Lvq` model ready for prediction.
/// Trains an LVQ2 model with custom initialization and a specific seed for reproducibility.
pub fn fit_lvq2_with_init_and_seed<L, F, D>(
    data: &[DataPoint<L, F>],
    n_prototypes_per_class: &HashMap<L, usize>,
    learning_rate: F,
    epochs: u32,
    distance: D,
    decay: F,
    window: F,
    init: PrototypeInit<F>,
    seed: u64,
) -> Result<Lvq<L, F, D>, LvqError>
where
    L: Clone + Eq + Hash + Debug + Ord,
    F: Float,
    D: Distance<F> + Clone,
{
    if data.is_empty() {
        return Err(LvqError::EmptyDataSet);
    }

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    // --- 1. Initialization using specified strategy ---
    let mut codebook =
        initialize_prototypes(data, n_prototypes_per_class, &distance, init, &mut rng)?;

    // --- 2. Training Loop (LVQ2) ---
    // End of Debug
    for _epoch in 0..epochs {
        let mut current_learning_rate = learning_rate;
        let mut training_indices: Vec<usize> = (0..data.len()).collect();
        //Debug
        let mut total_movement = F::zero();
        let codebook_before = codebook.clone();
        let mut updates_count = 0;
        let mut window_hits = 0;
        println!(
            "Epoch {}: learning_rate = {:.6}",
            _epoch, current_learning_rate
        );
        training_indices.shuffle(&mut rng);

        for &data_idx in &training_indices {
            let data_point = &data[data_idx];

            // a. Find the two closest prototypes
            if codebook.len() < 2 {
                // Cannot perform LVQ2 with fewer than 2 prototypes, fall back to LVQ1 logic
                // (or skip, as LVQ2 conditions would not be met anyway)
                continue;
            }

            let mut dists: Vec<(usize, F)> = codebook
                .iter()
                .enumerate()
                .map(|(i, proto)| {
                    (
                        i,
                        distance.rdistance(data_point.features.view(), proto.features.view()),
                    )
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            let (bmu1_idx, bmu1_dist) = dists[0];
            let (bmu2_idx, bmu2_dist) = dists[1];

            let bmu1_label = &codebook[bmu1_idx].label;
            let bmu2_label = &codebook[bmu2_idx].label;

            // b. Only update if the two closest prototypes belong to different classes
            //    and one of them matches the data point's label
            // println!("=== Window Debug Info ===");
            // println!(
            //     "BMU1 distance: {:.6}, BMU2 distance: {:.6}",
            //     bmu1_dist, bmu2_dist
            // );
            // println!(
            //     "BMU1 label: {:?}, BMU2 label: {:?}, Data label: {:?}",
            //     bmu1_label, bmu2_label, data_point.label
            // );
            if bmu1_label != bmu2_label
                && (&data_point.label == bmu1_label || &data_point.label == bmu2_label)
            {
                // c. Check if the sample is within the window (CORRECTED LOGIC)
                // The condition is that the data point falls in a 'window' around the
                // midplane of the two prototypes. This is expressed as a ratio of the distances.
                // d1/d2 > (1 - window) / (1 + window)
                // Since we are using squared distances (rdistance), the check becomes:
                // (d1/d2)^2 > ((1 - window) / (1 + window))^2

                // Avoid division by zero if prototypes are identical
                println!("=== Window Debug Info ===");
                println!(
                    "BMU1 distance: {:.6}, BMU2 distance: {:.6}",
                    bmu1_dist, bmu2_dist
                );
                println!(
                    "BMU1 label: {:?}, BMU2 label: {:?}, Data label: {:?}",
                    bmu1_label, bmu2_label, data_point.label
                );
                if bmu2_dist > F::epsilon() {
                    let dist_ratio_sq = bmu1_dist / bmu2_dist; // This is (d1/d2)^2
                    println!("Distance ratio squared (d1/d2): {:.6}", dist_ratio_sq);
                    let window_threshold_num = F::one() - window;
                    let window_threshold_den = F::one() + window;
                    println!("Window parameter: {:.3}", window);
                    println!("Threshold numerator (1-w): {:.6}", window_threshold_num);
                    println!("Threshold denominator (1+w): {:.6}", window_threshold_den);
                    // Check for valid window parameter to avoid division by zero
                    if window_threshold_den > F::epsilon() {
                        let window_threshold = window_threshold_num / window_threshold_den;
                        let window_threshold_sq = window_threshold * window_threshold;
                        println!("Window threshold: {:.6}", window_threshold);
                        println!("Window threshold squared: {:.6}", window_threshold_sq);
                        println!(
                            "Condition: {:.6} > {:.6} ? {}",
                            dist_ratio_sq,
                            window_threshold_sq,
                            dist_ratio_sq > window_threshold_sq
                        );

                        if dist_ratio_sq > window_threshold_sq {
                            // d. Update both prototypes
                            let (winner_idx, loser_idx) = if &data_point.label == bmu1_label {
                                (bmu1_idx, bmu2_idx)
                            } else {
                                (bmu2_idx, bmu1_idx)
                            };

                            // Update winner (move closer) using a mutable split to borrow twice
                            // This complex block is to satisfy the borrow checker
                            let (codebook_part1, codebook_part2) =
                                codebook.split_at_mut(winner_idx.max(loser_idx));
                            let (winner, loser) = if winner_idx < loser_idx {
                                (&mut codebook_part1[winner_idx], &mut codebook_part2[0])
                            } else {
                                (&mut codebook_part2[0], &mut codebook_part1[loser_idx])
                            };

                            let update_vec_winner = &data_point.features - &winner.features;
                            winner.features += &(update_vec_winner * current_learning_rate);

                            // Update loser (move away)
                            let update_vec_loser = &data_point.features - &loser.features;
                            loser.features -= &(update_vec_loser * current_learning_rate);

                            // Debug: Count updates and window hits
                            window_hits += 1;
                            updates_count += 1;
                        }
                    }
                } else {
                    println!("BMU2 distance too small: {:.10}", bmu2_dist);
                }
                println!("========================");
            }
        }
        // e. Decay the learning rate for the next epoch
        current_learning_rate = current_learning_rate * decay;
        // Debug: Calculate total movement of prototypes
        for (i, (before, after)) in codebook_before.iter().zip(codebook.iter()).enumerate() {
            let movement = distance
                .rdistance(before.features.view(), after.features.view())
                .sqrt();
            total_movement += movement;
            println!("Prototype {}: moved {:.6}", i, movement);
        }
        println!("Total movement this epoch: {:.6}", total_movement);
        println!(
            "Window hits: {}, Actual updates: {}",
            window_hits, updates_count
        );
        println!("--- Prototype positions at end of epoch {} ---", _epoch);
        for (i, proto) in codebook.iter().enumerate() {
            println!(
                "  Proto {}: Label='{:?}', Position={:?}",
                i, proto.label, proto.features
            );
        }
        println!("----------------------------------------------");
    }

    Ok(Lvq { codebook, distance })
}

/// Trains an LVQ3 model.
///
/// # Arguments
///
/// * `data`: The labeled training data.
/// * `n_prototypes_per_class`: A map from class label to the number of prototypes for that class.
/// * `learning_rate`: The initial learning rate (alpha).
/// * `epochs`: The number of times to iterate over the training data.
/// * `distance`: The distance metric to use.
/// * `decay`: The multiplicative decay factor for the learning rate (e.g., 0.99 for 1% decay per epoch).
/// * `window`: The window width parameter (e.g., 0.3).
/// * `epsilon`: The factor for same-class prototype updates (e.g., 0.5).
///
/// # Returns
/// A trained `Lvq` model ready for prediction.
pub fn fit_lvq3<L, F, D>(
    data: &[DataPoint<L, F>],
    n_prototypes_per_class: &HashMap<L, usize>,
    learning_rate: F,
    epochs: u32,
    distance: D,
    decay: F,
    window: F,
    epsilon: F,
) -> Result<Lvq<L, F, D>, LvqError>
where
    L: Clone + Eq + Hash + Debug + Ord,
    F: Float,
    D: Distance<F> + Clone,
{
    fit_lvq3_with_seed(
        data,
        n_prototypes_per_class,
        learning_rate,
        epochs,
        distance,
        decay,
        window,
        epsilon,
        rand::random(),
    )
}

/// Trains an LVQ3 model with a specific seed for reproducibility.
///
/// # Arguments
///
/// * `data`: The labeled training data.
/// * `n_prototypes_per_class`: A map from class label to the number of prototypes for that class.
/// * `learning_rate`: The initial learning rate (alpha).
/// * `epochs`: The number of times to iterate over the training data.
/// * `distance`: The distance metric to use.
/// * `decay`: The multiplicative decay factor for the learning rate (e.g., 0.99 for 1% decay per epoch).
/// * `window`: The window width parameter (e.g., 0.3).
/// * `epsilon`: The factor for same-class prototype updates (e.g., 0.5).
/// * `seed`: The seed for the random number generator for reproducible results.
///
/// # Returns
/// A trained `Lvq` model ready for prediction.
pub fn fit_lvq3_with_seed<L, F, D>(
    data: &[DataPoint<L, F>],
    n_prototypes_per_class: &HashMap<L, usize>,
    learning_rate: F,
    epochs: u32,
    distance: D,
    decay: F,
    window: F,
    epsilon: F,
    seed: u64,
) -> Result<Lvq<L, F, D>, LvqError>
where
    L: Clone + Eq + Hash + Debug + Ord,
    F: Float,
    D: Distance<F> + Clone,
{
    fit_lvq3_with_init_and_seed(
        data,
        n_prototypes_per_class,
        learning_rate,
        epochs,
        distance,
        decay,
        window,
        epsilon,
        PrototypeInit::Random,
        seed,
    )
}

/// Trains an LVQ3 model with custom initialization and a specific seed for reproducibility.
///
/// # Arguments
///
/// * `data`: The labeled training data.
/// * `n_prototypes_per_class`: A map from class label to the number of prototypes for that class.
/// * `learning_rate`: The initial learning rate (alpha).
/// * `epochs`: The number of times to iterate over the training data.
/// * `distance`: The distance metric to use.
/// * `decay`: The multiplicative decay factor for the learning rate (e.g., 0.99 for 1% decay per epoch).
/// * `window`: The window width parameter (e.g., 0.3).
/// * `epsilon`: The factor for same-class prototype updates (e.g., 0.5).
/// * `init`: The prototype initialization strategy.
/// * `seed`: The seed for the random number generator for reproducible results.
///
/// # Returns
/// A trained `Lvq` model ready for prediction.
pub fn fit_lvq3_with_init_and_seed<L, F, D>(
    data: &[DataPoint<L, F>],
    n_prototypes_per_class: &HashMap<L, usize>,
    learning_rate: F,
    epochs: u32,
    distance: D,
    decay: F,
    window: F,
    epsilon: F,
    init: PrototypeInit<F>,
    seed: u64,
) -> Result<Lvq<L, F, D>, LvqError>
where
    L: Clone + Eq + Hash + Debug + Ord,
    F: Float,
    D: Distance<F> + Clone,
{
    if data.is_empty() {
        return Err(LvqError::EmptyDataSet);
    }

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    // --- 1. Initialization using specified strategy ---
    let mut codebook =
        initialize_prototypes(data, n_prototypes_per_class, &distance, init, &mut rng)?;

    // --- 2. Training Loop (LVQ3) ---
    let mut current_learning_rate = learning_rate;
    let mut training_indices: Vec<usize> = (0..data.len()).collect();

    for _epoch in 0..epochs {
        training_indices.shuffle(&mut rng);

        for &data_idx in &training_indices {
            let data_point = &data[data_idx];

            if codebook.len() < 2 {
                continue;
            }

            // a. Find the two closest prototypes
            let mut dists: Vec<(usize, F)> = codebook
                .iter()
                .enumerate()
                .map(|(i, proto)| {
                    (
                        i,
                        distance.rdistance(data_point.features.view(), proto.features.view()),
                    )
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            let (bmu1_idx, bmu1_dist) = dists[0];
            let (bmu2_idx, bmu2_dist) = dists[1];

            let bmu1_label = &codebook[bmu1_idx].label;
            let bmu2_label = &codebook[bmu2_idx].label;

            // --- CORRECTED WINDOW LOGIC ---
            if bmu2_dist > F::epsilon() {
                let dist_ratio_sq = bmu1_dist / bmu2_dist;

                let window_threshold_num = F::one() - window;
                let window_threshold_den = F::one() + window;

                if window_threshold_den > F::epsilon() {
                    let window_threshold = window_threshold_num / window_threshold_den;
                    let window_threshold_sq = window_threshold * window_threshold;

                    // Check if data point is inside the window
                    if dist_ratio_sq > window_threshold_sq {
                        // This block handles updates inside the window
                        if bmu1_label != bmu2_label
                            && (&data_point.label == bmu1_label || &data_point.label == bmu2_label)
                        {
                            // Standard LVQ2 update for different-class prototypes
                            let (winner_idx, loser_idx) = if &data_point.label == bmu1_label {
                                (bmu1_idx, bmu2_idx)
                            } else {
                                (bmu2_idx, bmu1_idx)
                            };

                            // Mutable split to satisfy borrow checker
                            let (part1, part2) = codebook.split_at_mut(winner_idx.max(loser_idx));
                            let (winner, loser) = if winner_idx < loser_idx {
                                (&mut part1[winner_idx], &mut part2[0])
                            } else {
                                (&mut part2[0], &mut part1[loser_idx])
                            };

                            let update_vec = &data_point.features - &winner.features;
                            winner.features += &(update_vec * current_learning_rate);

                            let update_vec = &data_point.features - &loser.features;
                            loser.features -= &(update_vec * current_learning_rate);
                        } else if bmu1_label == bmu2_label && &data_point.label == bmu1_label {
                            // LVQ3 specific update: both prototypes are correct class
                            let update = current_learning_rate * epsilon;

                            // Mutable split to satisfy borrow checker
                            let (part1, part2) = codebook.split_at_mut(bmu1_idx.max(bmu2_idx));
                            let (proto1, proto2) = if bmu1_idx < bmu2_idx {
                                (&mut part1[bmu1_idx], &mut part2[0])
                            } else {
                                (&mut part2[0], &mut part1[bmu2_idx])
                            };

                            let update_vec1 = &data_point.features - &proto1.features;
                            proto1.features += &(update_vec1 * update);

                            let update_vec2 = &data_point.features - &proto2.features;
                            proto2.features += &(update_vec2 * update);
                        }
                    }
                }
            }
        }
        // e. Decay the learning rate for the next epoch
        current_learning_rate = current_learning_rate * decay;
    }

    Ok(Lvq { codebook, distance })
}
/// Trains an LVQ3 model with custom initialization.
///
/// # Arguments
///
/// * `data`: The labeled training data.
/// * `n_prototypes_per_class`: A map from class label to the number of prototypes for that class.
/// * `learning_rate`: The initial learning rate (alpha).
/// * `epochs`: The number of times to iterate over the training data.
/// * `distance`: The distance metric to use.
/// * `decay`: The multiplicative decay factor for the learning rate (e.g., 0.99 for 1% decay per epoch).
/// * `window`: The window width parameter (e.g., 0.3).
/// * `epsilon`: The factor for same-class prototype updates (e.g., 0.5).
/// * `init`: The prototype initialization strategy.
///
/// # Returns
/// A trained `Lvq` model ready for prediction.
pub fn fit_lvq3_with_init<L, F, D>(
    data: &[DataPoint<L, F>],
    n_prototypes_per_class: &HashMap<L, usize>,
    learning_rate: F,
    epochs: u32,
    distance: D,
    decay: F,
    window: F,
    epsilon: F,
    init: PrototypeInit<F>,
) -> Result<Lvq<L, F, D>, LvqError>
where
    L: Clone + Eq + Hash + Debug + Ord,
    F: Float,
    D: Distance<F> + Clone,
{
    fit_lvq3_with_init_and_seed(
        data,
        n_prototypes_per_class,
        learning_rate,
        epochs,
        distance,
        decay,
        window,
        epsilon,
        init,
        rand::random(),
    )
}

/// Helper function to group data indices by their class label.
fn group_data_by_class<L, F>(data: &[DataPoint<L, F>]) -> Result<HashMap<L, Vec<usize>>, LvqError>
where
    L: Clone + Eq + Hash + Debug + Ord,
    F: Float,
{
    let mut map = HashMap::new();
    for (i, dp) in data.iter().enumerate() {
        map.entry(dp.label.clone()).or_insert_with(Vec::new).push(i);
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use prec_helpers::L2Dist;

    fn make_simple_data() -> Vec<DataPoint<&'static str, f64>> {
        vec![
            DataPoint::new(array![0.0, 0.0], "A"),
            DataPoint::new(array![0.1, 0.2], "A"),
            DataPoint::new(array![0.2, 0.1], "A"),
            DataPoint::new(array![1.0, 1.0], "B"),
            DataPoint::new(array![1.1, 1.2], "B"),
            DataPoint::new(array![1.2, 1.1], "B"),
        ]
    }

    fn proto_map() -> std::collections::HashMap<&'static str, usize> {
        let mut m = std::collections::HashMap::new();
        m.insert("A", 1);
        m.insert("B", 1);
        m
    }

    #[test]
    fn test_lvq1_fit_and_predict() {
        let data = make_simple_data();
        let proto = proto_map();
        let model = fit(&data, &proto, 0.3, 30, L2Dist, 0.98).expect("LVQ1 fit should succeed");

        // Should predict "A" for a point near cluster A, "B" for cluster B
        let pred_a = model.predict(array![0.05, 0.05].view());
        let pred_b = model.predict(array![1.1, 1.05].view());
        assert_eq!(pred_a, "A");
        assert_eq!(pred_b, "B");
    }

    #[test]
    fn test_lvq2_fit_and_predict() {
        let data = make_simple_data();
        let proto = proto_map();
        let model =
            fit_lvq2(&data, &proto, 0.3, 30, L2Dist, 0.98, 0.3).expect("LVQ2 fit should succeed");

        let pred_a = model.predict(array![0.05, 0.05].view());
        let pred_b = model.predict(array![1.1, 1.05].view());
        assert_eq!(pred_a, "A");
        assert_eq!(pred_b, "B");
    }

    #[test]
    fn test_lvq3_fit_and_predict() {
        let data = make_simple_data();
        let proto = proto_map();
        let model = fit_lvq3(&data, &proto, 0.3, 30, L2Dist, 0.98, 0.3, 0.5)
            .expect("LVQ3 fit should succeed");

        let pred_a = model.predict(array![0.05, 0.05].view());
        let pred_b = model.predict(array![1.1, 1.05].view());
        assert_eq!(pred_a, "A");
        assert_eq!(pred_b, "B");
    }

    #[test]
    fn test_prototype_init_random() {
        let data = make_simple_data();
        let proto = proto_map();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let codebook = initialize_prototypes(
            &data,
            &proto,
            &L2Dist,
            PrototypeInit::<f64>::Random,
            &mut rng,
        )
        .expect("Random init should succeed");
        assert_eq!(codebook.len(), 2); // One prototype per class
    }

    #[test]
    fn test_prototype_init_class_mean() {
        let data = make_simple_data();
        let proto = proto_map();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let codebook = initialize_prototypes(
            &data,
            &proto,
            &L2Dist,
            PrototypeInit::<f64>::ClassMean,
            &mut rng,
        )
        .expect("ClassMean init should succeed");
        assert_eq!(codebook.len(), 2); // One prototype per class

        // Check that prototypes are roughly at class centers
        let proto_a = codebook.iter().find(|p| p.label == "A").unwrap();
        let proto_b = codebook.iter().find(|p| p.label == "B").unwrap();

        // Class A center should be around [0.1, 0.1]
        assert!((proto_a.features[0] - 0.1).abs() < 0.1);
        assert!((proto_a.features[1] - 0.1).abs() < 0.1);

        // Class B center should be around [1.1, 1.1]
        assert!((proto_b.features[0] - 1.1).abs() < 0.1);
        assert!((proto_b.features[1] - 1.1).abs() < 0.1);
    }

    #[test]
    fn test_error_on_class_mean_multiple_prototypes() {
        let data = make_simple_data();
        let mut proto = HashMap::new();
        proto.insert("A", 2); // Request 2 prototypes for ClassMean (should fail)
        proto.insert("B", 1);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let result = initialize_prototypes(
            &data,
            &proto,
            &L2Dist,
            PrototypeInit::<f64>::ClassMean,
            &mut rng,
        );
        assert!(matches!(result, Err(LvqError::InvalidConfig(_))));
    }

    #[test]
    fn test_prototype_init_kmeans() {
        let data = make_simple_data();
        let mut proto = HashMap::new();
        proto.insert("A", 2); // Request 2 prototypes per class using K-means
        proto.insert("B", 2);

        let kmeans_init = PrototypeInit::KMeans {
            max_iter: 100,
            tol: 1e-4,
        };

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let codebook = initialize_prototypes(&data, &proto, &L2Dist, kmeans_init, &mut rng)
            .expect("K-means init should succeed");

        assert_eq!(codebook.len(), 4); // Two prototypes per class

        // Check that we have 2 prototypes for each class
        let proto_a_count = codebook.iter().filter(|p| p.label == "A").count();
        let proto_b_count = codebook.iter().filter(|p| p.label == "B").count();
        assert_eq!(proto_a_count, 2);
        assert_eq!(proto_b_count, 2);
    }

    #[test]
    fn test_prototype_init_kmeans_single_prototype() {
        let data = make_simple_data();
        let proto = proto_map(); // 1 prototype per class

        let kmeans_init = PrototypeInit::KMeans {
            max_iter: 100,
            tol: 1e-4,
        };

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let codebook = initialize_prototypes(&data, &proto, &L2Dist, kmeans_init, &mut rng)
            .expect("K-means init with single prototype should succeed");

        assert_eq!(codebook.len(), 2); // One prototype per class

        // Should use class mean for single prototype case
        let proto_a = codebook.iter().find(|p| p.label == "A").unwrap();
        let proto_b = codebook.iter().find(|p| p.label == "B").unwrap();

        // Class A center should be around [0.1, 0.1]
        assert!((proto_a.features[0] - 0.1).abs() < 0.1);
        assert!((proto_a.features[1] - 0.1).abs() < 0.1);

        // Class B center should be around [1.1, 1.1]
        assert!((proto_b.features[0] - 1.1).abs() < 0.1);
        assert!((proto_b.features[1] - 1.1).abs() < 0.1);
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let data = make_simple_data();
        let proto = proto_map();

        // Train the same model twice with the same seed
        let model1 = fit_with_seed(&data, &proto, 0.3, 30, L2Dist, 0.98, 42)
            .expect("LVQ1 fit should succeed");
        let model2 = fit_with_seed(&data, &proto, 0.3, 30, L2Dist, 0.98, 42)
            .expect("LVQ1 fit should succeed");

        // The codebooks should be identical (same prototype positions)
        assert_eq!(model1.codebook.len(), model2.codebook.len());
        for (proto1, proto2) in model1.codebook.iter().zip(model2.codebook.iter()) {
            assert_eq!(proto1.label, proto2.label);
            // Check that feature vectors are very close (within floating point precision)
            for (f1, f2) in proto1.features.iter().zip(proto2.features.iter()) {
                assert!(
                    (f1 - f2).abs() < 1e-10,
                    "Features should be identical: {} vs {}",
                    f1,
                    f2
                );
            }
        }

        // Test predictions should also be identical
        let pred1_a = model1.predict(array![0.05, 0.05].view());
        let pred1_b = model1.predict(array![1.1, 1.05].view());
        let pred2_a = model2.predict(array![0.05, 0.05].view());
        let pred2_b = model2.predict(array![1.1, 1.05].view());

        assert_eq!(pred1_a, pred2_a);
        assert_eq!(pred1_b, pred2_b);
    }

    #[test]
    fn test_different_seeds_produce_different_results() {
        let data = make_simple_data();
        let proto = proto_map();

        // Train with different seeds
        let model1 = fit_with_seed(&data, &proto, 0.3, 30, L2Dist, 0.98, 42)
            .expect("LVQ1 fit should succeed");
        let model2 = fit_with_seed(&data, &proto, 0.3, 30, L2Dist, 0.98, 123)
            .expect("LVQ1 fit should succeed");

        // Models should have different prototypes (very likely with different seeds)
        let mut different_found = false;
        for (proto1, proto2) in model1.codebook.iter().zip(model2.codebook.iter()) {
            for (f1, f2) in proto1.features.iter().zip(proto2.features.iter()) {
                if (f1 - f2).abs() > 1e-6 {
                    different_found = true;
                    break;
                }
            }
            if different_found {
                break;
            }
        }

        assert!(
            different_found,
            "Different seeds should produce different results"
        );
    }

    #[test]
    fn debug_prototype_positions() {
        // Create simple test data with well-separated classes
        let data = vec![
            // Class A - bottom left
            DataPoint::new(array![1.0, 1.0], "A"),
            DataPoint::new(array![1.5, 1.2], "A"),
            DataPoint::new(array![1.2, 1.5], "A"),
            DataPoint::new(array![0.8, 1.3], "A"),
            // Class B - top right
            DataPoint::new(array![8.0, 8.0], "B"),
            DataPoint::new(array![8.5, 8.2], "B"),
            DataPoint::new(array![8.2, 8.5], "B"),
            DataPoint::new(array![7.8, 8.3], "B"),
        ];

        let mut n_prototypes_per_class = HashMap::new();
        n_prototypes_per_class.insert("A", 1);
        n_prototypes_per_class.insert("B", 1);

        println!("\n=== Debug Prototype Positions ===");
        println!("Training data:");
        for (i, point) in data.iter().enumerate() {
            println!(
                "  {}: [{:.1}, {:.1}] -> {}",
                i, point.features[0], point.features[1], point.label
            );
        }

        // Test different learning parameters
        let test_cases = vec![
            ("Conservative", 0.01, 0.99, 100),
            ("Moderate", 0.1, 0.99, 100),
            ("Aggressive", 0.3, 0.95, 200),
        ];

        for (name, lr, decay, epochs) in test_cases {
            println!(
                "\n--- {} (lr={}, decay={}, epochs={}) ---",
                name, lr, decay, epochs
            );

            match fit_with_init_and_seed(
                &data,
                &n_prototypes_per_class,
                lr,
                epochs,
                L2Dist,
                decay,
                PrototypeInit::Random,
                42,
            ) {
                Ok(model) => {
                    println!("Training successful!");
                    println!("Prototypes:");
                    for (i, proto) in model.codebook.iter().enumerate() {
                        println!(
                            "  {}: [{:.3}, {:.3}] -> {}",
                            i, proto.features[0], proto.features[1], proto.label
                        );
                    }

                    // Check if prototypes are reasonably positioned
                    let proto_a = model.codebook.iter().find(|p| p.label == "A").unwrap();
                    let proto_b = model.codebook.iter().find(|p| p.label == "B").unwrap();

                    // Calculate average positions of data points
                    let avg_a_x: f64 = 1.125; // (1.0 + 1.5 + 1.2 + 0.8) / 4
                    let avg_a_y: f64 = 1.25; // (1.0 + 1.2 + 1.5 + 1.3) / 4
                    let avg_b_x: f64 = 8.125; // (8.0 + 8.5 + 8.2 + 7.8) / 4
                    let avg_b_y: f64 = 8.25; // (8.0 + 8.2 + 8.5 + 8.3) / 4

                    println!("  Expected A around: [{:.3}, {:.3}]", avg_a_x, avg_a_y);
                    println!("  Expected B around: [{:.3}, {:.3}]", avg_b_x, avg_b_y);

                    // Check distance from expected positions
                    let dist_a = ((proto_a.features[0] - avg_a_x).powf(2.0)
                        + (proto_a.features[1] - avg_a_y).powf(2.0))
                    .sqrt();
                    let dist_b = ((proto_b.features[0] - avg_b_x).powf(2.0)
                        + (proto_b.features[1] - avg_b_y).powf(2.0))
                    .sqrt();

                    println!("  Distance from expected A: {:.3}", dist_a);
                    println!("  Distance from expected B: {:.3}", dist_b);
                }
                Err(e) => {
                    println!("Training failed: {:?}", e);
                }
            }
        }
    }
}
