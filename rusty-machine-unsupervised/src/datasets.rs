use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::fs::File;
use std::vec::Vec;
use serde_derive::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Flower {
    Id: i32, 
    SepalLengthCm: f32, 
    SepalWidthCm: f32, 
    PetalLengthCm: f32, 
    PetalWidthCm: f32, 
    Species: String, 
}

impl Flower {
    pub fn into_feature_vector(&self) -> Vec<f32> {
        vec![self.SepalLengthCm, self.SepalWidthCm, self.PetalLengthCm, self.PetalWidthCm]
    }

    pub fn into_labels(&self) -> f32 {
        match self.Species.as_str() {
            "Iris-setosa" => 0., 
            "Iris-versicolor" => 1., 
            "Iris-virginica" => 2., 
            some_other => panic!("Not able to parse the target. Some other target got passed. {:?}", some_other), 
        }
    }

    pub fn into_int_labels(&self) -> u64 {
        match self.Species.as_str() {
            "Iris-setosa" => 0, 
            "Iris-versicolor" => 1, 
            "Iris-virginica" => 2, 
            l => panic!("Not able to parse the target. Some other target got passed. {:?}", l),
        }
    }
}