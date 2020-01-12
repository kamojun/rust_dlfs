extern crate csv;
use csv::{ReaderBuilder, WriterBuilder};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::io;
// extern crate ndarray;
use crate::model::rnn::{RnnlmLSTMParams, SavableParams};
use crate::params::{Param, P1};
use crate::types::{Arr1d, Arr2d};
use ndarray::{Array, Array1, Array2, Axis, Dimension, Ix1, Ix2, RemoveAxis};

pub trait Save {
    fn save_as_csv(&self, filename: &str) -> Result<(), Box<dyn Error>>;
}
impl<T: Clone + Serialize, D: RemoveAxis> Save for Array<T, D> {
    fn save_as_csv(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        let mut wtr = WriterBuilder::new()
            .has_headers(false)
            .from_path(filename)?;
        let n = self.shape().len();
        for row in self.lanes(Axis(n - 1)) {
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

impl<T: Save + Default> Save for P1<T> {
    fn save_as_csv(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        self.p().save_as_csv(filename)
    }
}
impl<SP: SavableParams> Save for SP {
    fn save_as_csv(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        for (p, name) in self.params_to_save() {
            p.save_as_csv(&format!("{}/{}", filename, name))?
        }
        Ok(())
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

pub trait Load {
    fn load_from_csv(filename: &str) -> Result<Self, Box<dyn Error>>
    where
        Self: std::marker::Sized;
}
/// Array<T, D>で実装したいところだが...
impl<T: Copy> Load for Array2<T>
where
    for<'de> T: Deserialize<'de>,
{
    fn load_from_csv(filename: &str) -> Result<Self, Box<dyn Error>> {
        csv_to_array(filename)
    }
}
impl<T: Copy> Load for Array1<T>
where
    for<'de> T: Deserialize<'de>,
{
    fn load_from_csv(filename: &str) -> Result<Self, Box<dyn Error>> {
        csv_to_array(filename).map(|res| res.outer_iter().next().unwrap().to_owned())
    }
}
impl<T: Load + Default> Load for P1<T> {
    fn load_from_csv(filename: &str) -> Result<Self, Box<dyn Error>> {
        T::load_from_csv(filename).map(P1::new) // まさにgeneric
    }
}
impl<SP: SavableParams> Load for SP {
    fn load_from_csv(filename: &str) -> Result<Self, Box<dyn Error>> {
        let (name1, name2) = SP::param_names();
        // generic なclosure作れず
        // let load_files = |names: Vec<&str>| {
        //     names
        //         .into_iter()
        //         .map(|name| P1::load_from_csv(&format!("{}/{}", filename, name)))
        //         .collect::<Result<Vec<_>, _>>()
        // };
        // let params1 = load_files(name1)?;
        // let params2 = load_files(name2)?;
        // ↓全く同じ3行を書かなければならないのだろうか...
        let params1 = name1
            .into_iter()
            .map(|name| P1::load_from_csv(&format!("{}/{}", filename, name)))
            .collect::<Result<Vec<_>, _>>()?;
        let params2 = name2
            .into_iter()
            .map(|name| P1::load_from_csv(&format!("{}/{}", filename, name)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SP::load_new(params1, params2))
    }
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
        use ndarray::Array1;
        Array1::<f32>::zeros((10,)).save_as_csv("zerozero.csv");
    }
}
