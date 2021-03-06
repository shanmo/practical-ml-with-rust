#![allow(non_snake_case)]

use std::error::Error;
use std::result::Result;
use std::vec::Vec;

use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;
use transpose;

use tensorflow as tf;
use tf::{Graph, Tensor, DataType};
use tf::{Session, SessionOptions, SessionRunArgs};

#[path = "./datasets.rs"] mod datasets;
use datasets::get_boston_records_from_file; 

#[cfg_attr(feature="examples_system_alloc", global_allocator)]
#[cfg(feature="examples_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

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
    let filename = "./data/housing.csv"; 
    let mut data = get_boston_records_from_file(&filename); 

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

    // define graph 
    let mut graph = Graph::new(); 
    let dim = (boston_y_train.len() as u64, 13); 
    let test_dim = (boston_y_test.len()as u64, dim.1); 
    let x_train = <Tensor<f64>>::new(&[dim.0, dim.1]).with_values(&boston_x_train)?;
    let y_train = <Tensor<f64>>::new(&[dim.0, 1]).with_values(&boston_y_train)?;
    let x_test = <Tensor<f64>>::new(&[test_dim.0, test_dim.1]).with_values(&boston_x_test)?;   

    let mut output_array = vec![0.0; (dim.0 * dim.1) as usize];
    transpose::transpose(&boston_x_train, &mut output_array, dim.1 as usize, dim.0 as usize); 
    let XT =  <Tensor<f64>>::new(&[dim.1, dim.0]).with_values(&output_array[..])?;

    // theta = (X^T @ X)^-1 @ X^T @ y 
    // set variables 
    let XT_const = {
        let mut op = graph.new_operation("Const", "XT")?;
        op.set_attr_tensor("value", XT)?;
        op.set_attr_type("dtype", DataType::Double)?; // check the enums https://github.com/tensorflow/rust/blob/ddff61850be1c8044ac86350caeed5a55824ebe4/src/lib.rs#L297
        op.finish()?
    };
    let X_const = {
        let mut op = graph.new_operation("Const", "X_train")?;
        op.set_attr_tensor("value", X_train)?;
        op.set_attr_type("dtype", DataType::Double)?; // check the enums https://github.com/tensorflow/rust/blob/ddff61850be1c8044ac86350caeed5a55824ebe4/src/lib.rs#L297
        op.finish()?
    };
    let y_const = {
        let mut op = graph.new_operation("Const", "y_train")?;
        op.set_attr_tensor("value", y_train)?;
        op.set_attr_type("dtype", DataType::Double)?;
        op.finish()?
    };

    // set operators 
    let mul = {
        let mut op = graph.new_operation("MatMul", "mul")?;
        op.add_input(XT_const.clone());
        op.add_input(X_const.clone());
        op.finish()?
    };
    let inverse = {
        let mut op = graph.new_operation("MatrixInverse", "mul_inv")?;
        op.add_input(mul);
        op.finish()?
    };
    let mul2 = {
        let mut op = graph.new_operation("MatMul", "mul2")?;
        op.add_input(inverse);
        op.add_input(XT_const.clone());
        op.finish()?
    };
    let theta = {
        let mut op = graph.new_operation("MatMul", "theta")?;
        op.add_input(mul2);
        op.add_input(y_const);
        op.finish()?
    };

    // running predictions
    // y = X_test .* theta
    let X_test_const = {
        let mut op = graph.new_operation("Const", "X_test")?;
        op.set_attr_tensor("value", X_test)?;
        op.set_attr_type("dtype", DataType::Double)?;
        op.finish()?
    };
    let predictions = {
        let mut op = graph.new_operation("MatMul", "preds")?;
        op.add_input(X_test_const);
        op.add_input(theta);
        op.finish()?
    };

    // Run graph.
    let session = Session::new(&SessionOptions::new(), &graph)?;
    let mut args = SessionRunArgs::new();
    let preds_token = args.request_fetch(&predictions, 0);
    session.run(&mut args)?;
    let preds_token_res: Tensor<f64> = args.fetch::<f64>(preds_token)?;
    // println!("Now the preds", );
    // println!("{:?}", &preds_token_res[..]);
    println!("r-squared error score: {:?}", r_squared_score(&preds_token_res.to_vec(), &boston_y_test));

    Ok(())
}
