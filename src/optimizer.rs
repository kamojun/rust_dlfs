extern crate ndarray;
use super::types::{Arr1d, Arr2d};
use ndarray::{arr1, Array1};

pub trait Optimizer {
    // fn update<'a, T: Math<'a>>(&self, mut params: Vec<&mut T>, grad: Vec<&'a T>) {}
    fn update1d(&self, mut params: Vec<&mut Arr1d>, grad: Vec<Arr1d>) {}
    fn update2d(&self, mut params: Vec<&mut Arr2d>, grad: Vec<Arr2d>) {}
}

pub struct SGD {
    pub lr: f32,
}

// impl SGD {
// arr1d, arr2dのそれぞれに対して呼び出せば良い。
// Math traitをどう設定するかが問題
// 無理だった...。  2019/11/23
//     fn update<'a, T: Math<'a>>(&self, mut params: Vec<&mut T>, grad: Vec<&'a T>) {
//         for i in (0..params.len()) {
//             // *params[i] -= &(grad[i].clone() * self.lr);
//             *params[i] -= &(grad[i].clone());
//         }
//     }
// }
impl Optimizer for SGD {
    fn update1d(&self, mut params: Vec<&mut Arr1d>, grad: Vec<Arr1d>) {
        for i in 0..params.len() {
            *params[i] -= &(&grad[i] * self.lr);
        }
    }
    fn update2d(&self, mut params: Vec<&mut Arr2d>, grad: Vec<Arr2d>) {
        for i in 0..params.len() {
            *params[i] -= &(&grad[i] * self.lr);
        }
    }
}

use std::marker::Sized;
use std::ops::{Mul, SubAssign};
// pub trait Math: SubAssign<Self> + Mul<f32, Output = Self> + Sized + Clone {}
pub trait Math<'a>: 'a + Sized + Clone + Mul<f32, Output = Self> + SubAssign<&'a Self> {}
impl Math<'_> for Arr1d {}

#[test]
fn main() {
    let s = SGD { lr: 0.02 };
    let mut a = arr1(&[1.0]);
    let b = arr1(&[1.0]);
    let p = vec![&mut a];
    let g = vec![&b];
    // s.update1d(p, g);
    println!("{}, {}", a, b);
}
