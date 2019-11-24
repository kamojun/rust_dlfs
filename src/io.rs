extern crate csv;
use csv::ReaderBuilder;
use serde::Deserialize;
use std::error::Error;
use std::io;

type Record = [f32; 2];

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
        let v: Vec<[f32; 2]> = read_csv("./data/spiral/x.csv").unwrap();
        let arr = Array::from_shape_fn((v.len(), 2), |(i, j)| v[i]);
        println!("{:?}", arr);
    }
}
