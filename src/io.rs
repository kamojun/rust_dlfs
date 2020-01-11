extern crate csv;
use csv::{ReaderBuilder, WriterBuilder};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::io;
// extern crate ndarray;
use crate::model::rnn::RnnlmLSTMParams;
use crate::params::P1;
use crate::types::Arr2d;
use ndarray::{Array2, Dimension, Ix2};

pub trait Save {
    fn save_as_csv(&self, filename: &str) -> Result<(), Box<dyn Error>>;
}
impl<T: Clone + Serialize> Save for Array2<T> {
    fn save_as_csv(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        let mut wtr = WriterBuilder::new()
            .has_headers(false)
            .from_path(filename)?;
        // wtr.serialize()?;
        for row in self.outer_iter() {
            wtr.serialize(row.iter().cloned().collect::<Vec<T>>())?
        }
        wtr.flush()?;
        Ok(())
    }
}
use crate::model::{Model2, CBOW};
impl Save for CBOW {
    fn save_as_csv(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        let v = self.params_immut();
        v[0].save_as_csv(&format!("{}{}", filename, "/w_in.csv"))?;
        v[1].save_as_csv(&format!("{}{}", filename, "/w_out.csv"))?;
        Ok(())
    }
}

// impl<D: Dimension> Save for P1<Array<f32, D>> {
//     fn save_as_csv(&self, filename: &str) -> Result<(), Box<dyn Error>> {
//         self.p().save_as_csv("hello");
//     }
// }
impl Save for RnnlmLSTMParams {
    fn save_as_csv(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        unimplemented!();
    }
}

fn save_example(arr: Arr2d) -> Result<(), Box<dyn Error>> {
    let mut wtr = WriterBuilder::new()
        .has_headers(false)
        .from_path("foo.csv")?;
    // wtr.serialize()?;
    for row in arr.outer_iter() {
        wtr.serialize(row.iter().cloned().collect::<Vec<f32>>())?
    }
    wtr.flush()?;
    Ok(())
}

pub fn csv_to_arrayf(filename: &str) -> Result<Arr2d, Box<dyn Error>> {
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

pub fn csv_to_array<T: Copy>(filename: &str) -> Result<Array2<T>, Box<dyn Error>>
where
    for<'de> T: Deserialize<'de>,
{
    let v: Vec<Vec<T>> = read_csv(filename)?;
    Ok(Array::from_shape_fn((v.len(), v[0].len()), |(i, j)| {
        v[i][j]
    }))
}

pub fn read_csv<T>(filename: &str) -> Result<Vec<T>, Box<dyn Error>>
where
    for<'de> T: Deserialize<'de>,
{
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

pub fn read_csv_small<T>(filename: &str, n: usize) -> Result<Vec<T>, Box<dyn Error>>
where
    for<'de> T: Deserialize<'de>,
{
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(filename)?;
    let mut v = Vec::new();
    for result in rdr.deserialize().take(n) {
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
        let arr = csv_to_array::<f32>("./data/spiral/t.csv").unwrap();
        // println!("{:?}", arr);
        // let c = read_csv::<(usize, String)>("./data/ptb/id.csv").expect("error!!!");
        arr.save_as_csv("hoge.csv");
        // putsl!(c);
    }
}
