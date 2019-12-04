extern crate csv;
use csv::ReaderBuilder;
use serde::Deserialize;
use std::error::Error;
use std::io;
// extern crate ndarray;
// use ndarray::Array;
use crate::types::Arr2d;

pub fn csv_to_array(filename: &str) -> Result<Arr2d, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(filename)?;
    let mut v = Vec::new();
    for result in rdr.deserialize() {
        let record: Vec<f32> = result?;
        v.push(record);
    }
    Ok(Array::from_shape_fn((v.len(), v[0].len()), |(i, j)| {
        v[i][j]
    }))
}

pub fn read_csv<T>(filename: &str) -> Result<Vec<T>, Box<dyn Error>>
where
    for<'de> T: Deserialize<'de>,
{
    println!("in read csv");
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(filename)?;
    let mut v = Vec::new();
    for result in rdr.deserialize() {
        let record: T = result?;
        v.push(record);
    }
    Ok(v)
}

extern crate ndarray;
use ndarray::Array;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn read_csv_test() {
        // let v: Vec<Vec<f32>> = read_csv("./data/spiral/x.csv").unwrap();
        // let arr = Array::from_shape_fn((v.len(), 2), |(i, j)| v[i][j]);
        // let arr = csv_to_array("./data/spiral/t.csv").unwrap();
        // println!("{:?}", arr);
        let c = read_csv::<(usize, String)>("./data/ptb/id.csv").expect("error!!!");
        putsl!(c);
    }
}
