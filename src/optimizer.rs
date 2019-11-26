extern crate ndarray;
use super::types::{Arr1d, Arr2d};
use ndarray::{arr1, Array, Array1};

pub trait Optimizer {
    // fn update<'a, T: Math<'a>>(&self, mut params: Vec<&mut T>, grad: Vec<&'a T>) {}
    fn update1d(&mut self, mut params: Vec<&mut Arr1d>, grad: Vec<Arr1d>) {}
    fn update2d(&mut self, mut params: Vec<&mut Arr2d>, grad: Vec<Arr2d>) {}
}

pub struct SGD {
    pub lr: f32,
}
impl Optimizer for SGD {
    fn update1d(&mut self, mut params: Vec<&mut Arr1d>, grad: Vec<Arr1d>) {
        for i in 0..params.len() {
            *params[i] -= &(&grad[i] * self.lr);
        }
    }
    fn update2d(&mut self, mut params: Vec<&mut Arr2d>, grad: Vec<Arr2d>) {
        for i in 0..params.len() {
            *params[i] -= &(&grad[i] * self.lr);
        }
    }
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

pub struct AdaGrad {
    lr: f32,
    h1d: Vec<Arr1d>,
    h2d: Vec<Arr2d>,
}
impl AdaGrad {
    pub fn new(lr: f32) -> Self {
        AdaGrad {
            lr,
            h1d: Vec::new(),
            h2d: Vec::new(),
        }
    }
}
impl Optimizer for AdaGrad {
    fn update1d(&mut self, mut params: Vec<&mut Arr1d>, grad: Vec<Arr1d>) {
        if self.h1d.len() == 0 {
            self.h1d = grad.iter().map(|g| Array::zeros(g.dim())).collect();
        }
        for i in 0..params.len() {
            self.h1d[i] += &(&grad[i] * &grad[i]);
            *params[i] -= &(&grad[i] * self.lr / (self.h1d[i].mapv(f32::sqrt) + 1e-7));
        }
    }
    fn update2d(&mut self, mut params: Vec<&mut Arr2d>, grad: Vec<Arr2d>) {
        if self.h2d.len() == 0 {
            self.h2d = grad.iter().map(|g| Array::zeros(g.dim())).collect();
        }
        for i in 0..params.len() {
            self.h2d[i] += &(&grad[i] * &grad[i]);
            *params[i] -= &(&grad[i] * self.lr / (self.h2d[i].mapv(f32::sqrt) + 1e-7));
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
