extern crate ndarray;
use super::types::{Arr1d, Arr2d};
use itertools::izip;
use ndarray::{arr1, Array, Array1, Dimension, Ix1};

pub trait Optimizer {
    // fn update<'a, T: Math<'a>>(&self, mut params: Vec<&mut T>, grad: Vec<&'a T>) {}
    fn update1d(&mut self, mut params: Vec<&mut Arr1d>, grad: Vec<Arr1d>) {}
    fn update2d(&mut self, mut params: Vec<&mut Arr2d>, grad: Vec<Arr2d>) {}
    fn update<D: Dimension>(
        &mut self,
        mut params: Vec<&mut Array<f32, D>>,
        grad: Vec<Array<f32, D>>,
    ) {
    }
}

pub struct SGD {
    pub lr: f32,
}
impl Optimizer for SGD {
    fn update1d(&mut self, mut params: Vec<&mut Arr1d>, grad: Vec<Arr1d>) {
        self.update(params, grad);
    }
    fn update2d(&mut self, mut params: Vec<&mut Arr2d>, grad: Vec<Arr2d>) {
        self.update(params, grad);
    }
    fn update<D: Dimension>(
        &mut self,
        mut params: Vec<&mut Array<f32, D>>,
        grad: Vec<Array<f32, D>>,
    ) {
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
fn updates<D: Dimension>(
    lr: f32,
    mut params: Vec<&mut Array<f32, D>>,
    grad: Vec<Array<f32, D>>,
    mut h: Vec<Array<f32, D>>,
) {
    if h.len() == 0 {
        h = grad.iter().map(|g| Array::zeros(g.dim())).collect();
    }
    for i in 0..params.len() {
        h[i] += &(&grad[i] * &grad[i]);
        *params[i] -= &(&grad[i] * lr / (h[i].mapv(f32::sqrt) + 1e-7));
    }
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
        // updates(self.lr, params, grad, self.h1d);
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

pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    iter: i32,
    m2d: Vec<Arr2d>,
    v2d: Vec<Arr2d>,
    m1d: Vec<Arr1d>,
    v1d: Vec<Arr1d>,
}
impl Adam {
    pub fn new(lr: f32, beta1: f32, beta2: f32) -> Self {
        // use default lr=0.001, beta1=0.9, beta2=0.999
        Adam {
            lr,
            beta1,
            beta2,
            iter: 0,
            m2d: vec![],
            v2d: vec![],
            m1d: vec![],
            v1d: vec![],
        }
    }
    fn update<D: Dimension>(
        mut params: Vec<&mut Array<f32, D>>,
        grad: Vec<Array<f32, D>>,
        m: &mut Vec<Array<f32, D>>,
        v: &mut Vec<Array<f32, D>>,
    ) {
    } // これを使ってDRYしたかったが...
}
impl Optimizer for Adam {
    fn update2d(&mut self, mut params: Vec<&mut Arr2d>, grad: Vec<Arr2d>) {
        if self.m2d.len() == 0 {
            self.m2d = grad.iter().map(|g| Array::zeros(g.dim())).collect();
            self.v2d = grad.iter().map(|g| Array::zeros(g.dim())).collect();
        }
        self.iter += 1;
        let lr_t = self.lr * (1.0 - self.beta2.powi(self.iter)).sqrt()
            / (1.0 - self.beta1.powi(self.iter));
        for i in 0..params.len() {
            self.m2d[i] += &((&grad[i] - &self.m2d[i]) * (1.0 - self.beta1));
            self.v2d[i] += &((&grad[i].mapv(|x| x.powi(2)) - &self.m2d[i]) * (1.0 - self.beta1));
            *params[i] -= &(&grad[i] * lr_t / (self.m2d[i].mapv(f32::sqrt) + 1e-7));
        }
        // for (p,g,m,v ) in izip!(params.iter_mut(), grad.iter(), self.m2d.iter_mut(), self.v2d.iter_mut()){}
    }
    fn update1d(&mut self, mut params: Vec<&mut Arr1d>, grad: Vec<Arr1d>) {
        if self.m1d.len() == 0 {
            self.m1d = grad.iter().map(|g| Array::zeros(g.dim())).collect();
            self.v1d = grad.iter().map(|g| Array::zeros(g.dim())).collect();
        }
        self.iter += 1;
        let lr_t = self.lr * (1.0 - self.beta2.powi(self.iter)).sqrt()
            / (1.0 - self.beta1.powi(self.iter));
        for i in 0..params.len() {
            self.m1d[i] += &((&grad[i] - &self.m1d[i]) * (1.0 - self.beta1));
            self.v1d[i] += &((&grad[i].mapv(|x| x.powi(2)) - &self.m1d[i]) * (1.0 - self.beta1));
            *params[i] -= &(&grad[i] * lr_t / (self.m1d[i].mapv(f32::sqrt) + 1e-7));
        }
        // for (p,g,m,v ) in izip!(params.iter_mut(), grad.iter(), self.m1d.iter_mut(), self.v1d.iter_mut()){}
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
