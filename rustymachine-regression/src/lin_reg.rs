use std::io::prelude::*;
use std::io::BufReader; 
use std::path::Path; 
use std::fs::File; 
use std::vec::Vec; 
use std::error::Error; 
use rand::thread_rng; 
use rand::seq::SliceRandom; 

use rusty_machine; 
use rusty_machine::linalg::Matrix; 
use rusty_machine::linalg::Vector; 
use rusty_machine::analysis::score::neg_mean_squared_error;
use rusty_machine::learning::lin_reg::LinRegressor;
use rusty_machine::learning::SupModel;

#[path = "./datasets.rs"] mod datasets;

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
    let fl = "data/housing.csv";
    let mut data = datasets::get_boston_records_from_file(&fl); 

    // prepare train and test datasets 
    data.shuffle(&mut thread_rng()); 
    let test_size: f64 = 0.2; 
    let test_size: f64 = data.len() as f64 * test_size; 
    let test_size = test_size.round() as usize; 
    let (test_data, train_data) = data.split_at(test_size); 
    let train_size = train_data.len(); 
    let test_size = test_data.len(); 

    let boston_x_train: Vec<f64> = train_data.iter()
        .flat_map(|r| r.into_feature_vector())
        .collect(); 
    let boston_y_train: Vec<f64> = train_data.iter()
        .map(|r| r.into_targets()).collect();
    let boston_x_test: Vec<f64> = test_data.iter()
        .flat_map(|r| r.into_feature_vector()).collect(); 
    let boston_y_test: Vec<f64> = test_data.iter()
        .map(|r| r.into_targets()).collect(); 

    let boston_x_train = Matrix::new(train_size, 13, boston_x_train); 
    let boston_y_train = Vector::new(boston_y_train); 
    let boston_x_test = Matrix::new(test_size, 13, boston_x_test); 
    let boston_y_test = Matrix::new(test_size, 1, boston_y_test);  

    let mut lin_model = LinRegressor::default(); 
    lin_model.train(&boston_x_train, &boston_y_train)?; 

    let predictions = lin_model.predict(&boston_x_test).unwrap(); 
    let predictions = Matrix::new(test_size, 1, predictions); 
    let acc = neg_mean_squared_error(&predictions, &boston_y_test);
    println!("linear regression error: {:?}", acc); 
    println!("linear regression R2 score: {:?}", r_squared_score(&boston_y_test.data(), &predictions.data()));

    Ok(())
}



