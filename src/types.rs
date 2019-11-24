extern crate ndarray;
use ndarray::{Array1, Array2};
pub type Arr1d = Array1<f32>;
pub type Arr2d = Array2<f32>;

pub enum Arr {
    Arr1d(Arr1d),
    Arr2d(Arr2d),
}

use std::marker::Sized;
use std::ops::{Deref, Mul, SubAssign};
pub trait Math: SubAssign + Mul<f32, Output = Self> + Sized + Clone {}

// impl Math for arr1d {}
