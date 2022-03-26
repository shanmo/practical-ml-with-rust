use std::io;
use std::vec::Vec;
use std::error::Error;
use std::iter::repeat;
use std::collections::HashSet;
use std::cmp::Ordering;


use rusty_machine as rm;
use rm::linalg::{Matrix, BaseMatrix};
use rm::learning::gmm::{CovOption, GaussianMixtureModel};
use rm::learning::dbscan::DBSCAN;
use rm::learning::UnSupModel;
use csv;
use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;
use rm::linalg::Vector;

#[path = "./datasets.rs"] mod datasets;
use datasets::Flower;

// for classification
pub fn accuracy(y_preds: &[u32], y_test: &[u32]) -> f32 {
    let mut correct_hits = 0;
    for (predicted, actual) in y_preds.iter().zip(y_test.iter()) {
        if predicted == actual {
            correct_hits += 1;
        }
    }
    let acc: f32 = correct_hits as f32 / y_test.len() as f32;
    acc
}

fn main() -> Result<(), Box<dyn Error>> {
    // Get all the data
    let mut rdr = csv::Reader::from_reader(io::stdin());
    let mut data = Vec::new();
    for result in rdr.deserialize() {
        let r: Flower = result?;
        data.push(r); // data contains all the records
    }

    // shuffle the data.
    data.shuffle(&mut thread_rng());

    // separate out to train and test datasets.
    let test_size: f64 = 0.2;
    let test_size: f64 = data.len() as f64 * test_size;
    let test_size = test_size.round() as usize;
    let (test_data, train_data) = data.split_at(test_size);
    let train_size = train_data.len();
    let test_size = test_data.len();

    // differentiate the features and the labels.
    let flower_x_train: Vec<f64> = train_data.iter().flat_map(|r| {
        let features = r.into_feature_vector();
        let features: Vec<f64> = features.iter().map(|&x| x as f64).collect();
        features
    }).collect();
    let flower_y_train: Vec<usize> = train_data.iter().map(
        |r| r.into_int_labels() as usize).collect();

    let flower_x_test: Vec<f64> = test_data.iter().flat_map(|r| {
        let features = r.into_feature_vector();
        let features: Vec<f64> = features.iter().map(|&x| x as f64).collect();
        features
    }).collect();
    let flower_y_test: Vec<u32> = test_data.iter().map(|r| r.into_int_labels() as u32).collect();

    // COnvert the data into matrices for rusty machine
    let flower_x_train = Matrix::new(train_size, 4, flower_x_train);
    let flower_y_train = Vector::new(flower_y_train);
    let flower_x_test = Matrix::new(test_size, 4, flower_x_test);

    output_separator();

    let mut model = rm::learning::k_means::KMeansClassifier::new(3); 
    model.train(&flower_x_train)?;
    let centroids = model.centroids().as_ref().unwrap();
    println!("Model Centroids:\n{:.3}", centroids);

    println!("Predicting the samples...");
    let classes = model.predict(&flower_x_test).unwrap();
    println!("number of classes from kmeans: {:?}", classes.data().len());

    output_separator(); 

    // create gmm with k classes 
    let mut model = GaussianMixtureModel::new(2); 
    model.set_max_iters(1000);
    model.cov_option = CovOption::Diagonal; 

    println!("training the model"); 
    model.train(&flower_x_train)?; 

    println!("{:?}", model.means()); 
    println!("{:?}", model.covariances()); 

    println!("predicting the samples..."); 
    let classes = model.predict(&flower_x_test).unwrap();
    println!("number of classes from gmm: {:?}", classes.data().len()); 

    let predicted_clusters = flower_labels_clusters_gmm(classes.data()); 
    println!("predicted clusters from gmm: {:?}", predicted_clusters); 

    output_separator(); 

    Ok(())
}

fn output_separator() {
    let repeat_string = repeat("*********").take(10).collect::<String>(); 
    println!("{}", repeat_string); 
    println!(""); 
}

fn flower_labels_clusters_gmm(iris_data: &Vec<f64>) -> Vec<HashSet<u64>> {
    let mut setosa = HashSet::new(); 
    let mut versicolor = HashSet::new(); 
    let mut virginica = HashSet::new(); 

    for (index, flower) in iris_data.chunks(3).enumerate() {
        match max_index(&flower) {
            0 => setosa.insert(index as u64),
            1 => versicolor.insert(index as u64),
            2 => virginica.insert(index as u64),
            _ => panic!("cluster not found"),
        };
    }
    vec![setosa, versicolor, virginica]
}

fn max_index(array: &[f64]) -> usize {
    let mut i = 0; 
    for (j, value) in array.iter().enumerate() {
        match value.partial_cmp(&array[i]).unwrap() {
            Ordering::Greater => i = j, 
            _ => (), 
        };
    }
    i 
}