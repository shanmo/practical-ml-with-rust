use std::io;
use std::vec::Vec;
use std::error::Error;

use csv;
use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;

use tch;
use tch::{nn, kind, Kind, Tensor, no_grad, vision, Device};
use tch::{nn::Module, nn::OptimizerConfig};

#[path = "./datasets.rs"] mod datasets;
use datasets::Flower;

static FEATURE_DIM: i64 = 4;
static HIDDEN_NODES: i64 = 10;
static LABELS: i64 = 3;

pub fn run() -> Result<(), Box<dyn Error>> {
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
    let test_size: f64 = 0.5;
    let test_size: f64 = data.len() as f64 * test_size;
    let test_size = test_size.round() as usize;

    let (test_data, train_data) = data.split_at(test_size);
    let train_size = train_data.len();
    let test_size = test_data.len();
    assert_eq!(train_size, test_size);

    // differentiate the features and the labels.
    // torch needs vectors in f64
    let flower_x_train: Vec<f64> = train_data.iter().flat_map(|r| r.into_feature_vector()).map(|x| x as f64).collect();
    let flower_y_train: Vec<f64> = train_data.iter().map(|r| r.into_labels()).map(|x| x as f64).collect();

    let flower_x_test: Vec<f64> = test_data.iter().flat_map(|r| r.into_feature_vector()).map(|x| x as f64).collect();
    let flower_y_test: Vec<f64> = test_data.iter().map(|r| r.into_labels()).map(|x| x as f64).collect();

    let flower_x_train = Tensor::of_slice(flower_x_train.as_slice());
    let flower_y_train = Tensor::of_slice(flower_y_train.as_slice()).to_kind(Kind::Int64);
    let flower_x_test = Tensor::of_slice(flower_x_test.as_slice());
    let flower_y_test = Tensor::of_slice(flower_y_test.as_slice()).to_kind(Kind::Int64);

    // print shape of all the data.
    println!("Training data shape {:?}", flower_x_train.size());
    println!("Training flower_y_train data shape {:?}", flower_y_train.size());

    // reshaping examples
    // one way to reshape is using unsqueeze
    //let flower_x_train1 = flower_x_train.unsqueeze(0); // Training data shape [1, 360]
    //println!("Training data shape {:?}", flower_x_train1.size());
    let train_size = train_size as i64;
    let test_size = test_size as i64;
    let flower_x_train = flower_x_train.view([train_size, FEATURE_DIM]);
    let flower_x_test = flower_x_test.view([test_size, FEATURE_DIM]);
    let flower_y_train = flower_y_train.view([train_size]);
    let flower_y_test = flower_y_test.view([test_size]);

    // define the trainable parameters ourselves  
    let mut ws = Tensor::ones(&[FEATURE_DIM, 1], kind::FLOAT_CPU).set_requires_grad(true);
    let mut bs = Tensor::ones(&[train_size], kind::FLOAT_CPU).set_requires_grad(true);

    for epoch in 1..200 {
        let logits = flower_x_train.mm(&ws) + &bs;
        let loss = logits.squeeze().cross_entropy_for_logits(&flower_y_train); // since working on label encoded vectors.
        ws.zero_grad();
        bs.zero_grad();
        loss.backward();
        no_grad(|| {
            ws += ws.grad() * (-1);
            bs += bs.grad() * (-1);
        });
        let test_logits = flower_x_test.mm(&ws) + &bs;
        let test_accuracy = test_logits
            .argmax(-1, false)
            .eq_tensor(&flower_y_test)
            .to_kind(Kind::Float)
            .mean(Kind::Float)
            .double_value(&[]);
        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            loss.double_value(&[]),
            100. * test_accuracy
        );
    }

    Ok(())
}