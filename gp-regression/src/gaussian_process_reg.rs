use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::fs::File;
use std::vec::Vec;
use std::error::Error;
use std::process::exit;
use std::env::args;

use rusty_machine;
use rusty_machine::linalg::Matrix;
// use rusty_machine::linalg::BaseMatrix;
use rusty_machine::linalg::Vector;
use rusty_machine::learning::gp::GaussianProcess;
use rusty_machine::learning::gp::ConstMean;
use rusty_machine::learning::toolkit::kernel;
use rusty_machine::analysis::score::neg_mean_squared_error;
use rusty_machine::learning::SupModel;

// use ndarray::{Array, arr1};
use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;

#[path = "./datasets.rs"] mod datasets;
use datasets::get_boston_records_from_file; 

// for regression
pub fn r_squared_score(y_test: &[f64], y_preds: &[f64]) -> f64 {
    let model_variance: f64 = y_test.iter().zip(y_preds.iter()).fold(
        0., |v, (y_i, y_i_hat)| {
            v + (y_i - y_i_hat).powi(2)
        }
    );

    // get the mean for the actual values to be used later
    let y_test_mean = y_test.iter().sum::<f64>() as f64
        / y_test.len() as f64;

    // finding the variance
    let variance =  y_test.iter().fold(
        0., |v, &x| {v + (x - y_test_mean).powi(2)}
    );
    let r2_calculated: f64 = 1.0 - (model_variance / variance);
    r2_calculated
}

pub fn run() -> Result<(), Box<dyn Error>> {
    // Get all the data
    let filename = "data/housing.csv";
    let mut data = get_boston_records_from_file(&filename); // file must be in the folder data

    // shuffle the data.
    data.shuffle(&mut thread_rng());

    // separate out to train and test datasets.
    let test_size: f64 = 0.2;
    let test_size: f64 = data.len() as f64 * test_size;
    let test_size = test_size.round() as usize;
    let (test_data, train_data) = data.split_at(test_size);
    let train_size = train_data.len();
    let test_size = test_data.len();

    // differentiate the features and the targets.
    let boston_x_train: Vec<f64> = train_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let boston_y_train: Vec<f64> = train_data.iter().map(|r| r.into_targets()).collect();

    let boston_x_test: Vec<f64> = test_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let boston_y_test: Vec<f64> = test_data.iter().map(|r| r.into_targets()).collect();

    // using ndarray
    // let boston_x_train = Array::from_shape_vec((train_size, 13), boston_x_train).unwrap();
    // let boston_y_train = Array::from_vec(boston_y_train);
    // let boston_x_test = Array::from_shape_vec((test_size, 13), boston_x_test).unwrap();
    // let boston_y_test = Array::from_vec(boston_y_test);

    // COnvert the data into matrices for rusty machine
    let boston_x_train = Matrix::new(train_size, 13, boston_x_train);
    let boston_y_train = Vector::new(boston_y_train);
    let boston_x_test = Matrix::new(test_size, 13, boston_x_test);
    // let boston_y_test = Vector::new(boston_y_test);
    let boston_y_test = Matrix::new(test_size, 1, boston_y_test);

    // Create a gaussian process regression
    // A squared exponential kernel with lengthscale 2 and amplitude 1
    let ker = kernel::SquaredExp::new(2., 1.); 
    // zero function as mean function
    let zero_mean = ConstMean::default(); 
    // defining the model with noise 10
    let mut gaus_model = GaussianProcess::new(ker, zero_mean, 10f64); 

    gaus_model.train(&boston_x_train, &boston_y_train); 

    let predictions = gaus_model.predict(&boston_x_test).unwrap(); 
    let predictions = Matrix::new(test_size, 1, predictions); 
    let acc = neg_mean_squared_error(&predictions, &boston_y_test); 
    println!("gaussian process regression error: {:?}", acc);
    println!("gaussian process regression R2 score: {:?}", r_squared_score(
        &boston_y_test.data(), &predictions.data()));

    Ok(())
}