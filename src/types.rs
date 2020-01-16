extern crate ndarray;
use ndarray::{Array1, Array2, Array3};
pub type Arr1d = Array1<f32>;
pub type Arr2d = Array2<f32>;
pub type Arr3d = Array3<f32>;
pub type Seq1d = Array1<usize>;
pub type Seq = Array2<usize>;
pub type Seq3d = Array3<usize>;

use std::marker::Sized;
use std::ops::{Deref, Mul, SubAssign};
pub trait Math: SubAssign + Mul<f32, Output = Self> + Sized + Clone {}

// impl Math for arr1d {}
