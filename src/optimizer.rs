extern crate ndarray;
use super::types::{Arr1d, Arr2d};
use crate::params::Update;
use itertools::izip;
use ndarray::{arr1, Array, Array1, Dimension, Ix1};

pub fn update(params: Vec<&Update>, clip: f32, lr: f32) {
    let norm = params
        .iter()
        .map(|x| x.grads_norm_squared())
        .sum::<f32>()
        .sqrt();
    let rate = (clip / (norm + 1e-6)).min(1.0) * lr;
    update_lr(params, rate);
}

fn update_lr(params: Vec<&Update>, lr: f32) {
    for param in params.iter() {
        param.update_lr(lr);
    }
}

pub struct NewSGD<'a> {
    lr: f32,
    clip: f32,
    params: Vec<&'a Update>,
}

impl<'a> NewSGD<'a> {
    pub fn new(lr: f32, clip: f32, params: Vec<&'a Update>) -> Self {
        Self { lr, clip, params }
    }
    pub fn update(&self) {
        self.update_lr(self.lr);
    }
    pub fn update_lr(&self, lr: f32) {
        for param in self.params.iter() {
            param.update_lr(self.lr);
        }
    }
    pub fn update_clip_lr(&self) {
        for param in self.params.iter() {
            param.update_clip_lr(self.clip, self.lr);
        }
    }
    pub fn update_clipgrads(&self) {
        let norm = self
            .params
            .iter()
            .map(|x| x.grads_norm_squared())
            .sum::<f32>()
            .sqrt();
        let rate = (self.clip / (norm + 1e-6)).min(1.0) * self.lr;
        self.update_lr(rate);
    }
}

pub trait Optimizer {
    // fn update<'a, T: Math<'a>>(&self, mut params: Vec<&mut T>, grad: Vec<&'a T>) {}
    fn update1d(&mut self, mut params: Vec<&mut Arr1d>, grads: Vec<Arr1d>) {
        self.update(params, grads);
    }
    fn update2d(&mut self, mut params: Vec<&mut Arr2d>, grads: Vec<Arr2d>) {
        self.update(params, grads);
    }
    fn update<D: Dimension>(
        &mut self,
        mut params: Vec<&mut Array<f32, D>>,
        grads: Vec<Array<f32, D>>,
    ) {
        unimplemented!()
    }
}

pub struct SGD {
    pub lr: f32,
}
impl Optimizer for SGD {
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

#[derive(Default)]
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
            ..AdaGrad::default()
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

#[derive(Default)]
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
    /// use default lr=0.001, beta1=0.9, beta2=0.999
    pub fn new(lr: f32, beta1: f32, beta2: f32) -> Self {
        Adam {
            lr,
            beta1,
            beta2,
            ..Adam::default()
        }
    }
    fn updatex<D: Dimension>(
        mut params: Vec<&mut Array<f32, D>>,
        grad: Vec<Array<f32, D>>,
        m: &mut Vec<Array<f32, D>>,
        v: &mut Vec<Array<f32, D>>,
    ) {
    } // これを使ってDRYしたかったが...
}
impl Optimizer for Adam {
    fn update2d(&mut self, mut params: Vec<&mut Arr2d>, grads: Vec<Arr2d>) {
        if self.m2d.len() == 0 {
            self.m2d = grads.iter().map(|g| Array::zeros(g.dim())).collect();
            self.v2d = grads.iter().map(|g| Array::zeros(g.dim())).collect();
        }
        self.iter += 1;
        let lr_t = self.lr * (1.0 - self.beta2.powi(self.iter)).sqrt()
            / (1.0 - self.beta1.powi(self.iter));
        for i in 0..params.len() {
            self.m2d[i] += &((&grads[i] - &self.m2d[i]) * (1.0 - self.beta1));
            self.v2d[i] += &((&grads[i].mapv(|x| x.powi(2)) - &self.v2d[i]) * (1.0 - self.beta2));
            *params[i] -= &(&self.m2d[i] * lr_t / (self.v2d[i].mapv(f32::sqrt) + 1e-7));
        }
    }
    fn update1d(&mut self, mut params: Vec<&mut Arr1d>, grads: Vec<Arr1d>) {
        if self.m1d.len() == 0 {
            self.m1d = grads.iter().map(|g| Array::zeros(g.dim())).collect();
            self.v1d = grads.iter().map(|g| Array::zeros(g.dim())).collect();
        }
        self.iter += 1;
        let lr_t = self.lr * (1.0 - self.beta2.powi(self.iter)).sqrt()
            / (1.0 - self.beta1.powi(self.iter));
        for i in 0..params.len() {
            self.m1d[i] += &((&grads[i] - &self.m1d[i]) * (1.0 - self.beta1));
            self.v1d[i] += &((&grads[i].mapv(|x| x.powi(2)) - &self.v1d[i]) * (1.0 - self.beta2));
            *params[i] -= &(&self.m1d[i] * lr_t / (self.v1d[i].mapv(f32::sqrt) + 1e-7));
        }
    }
}

use std::marker::Sized;
use std::ops::{Mul, SubAssign};
// pub trait Math: SubAssign<Self> + Mul<f32, Output = Self> + Sized + Clone {}
pub trait Math<'a>: 'a + Sized + Clone + Mul<f32, Output = Self> + SubAssign<&'a Self> {}
impl Math<'_> for Arr1d {}

fn main() {
    let s = SGD { lr: 0.02 };
    let mut a = arr1(&[1.0]);
    let b = arr1(&[1.0]);
    let p = vec![&mut a];
    let g = vec![&b];
    // s.update1d(p, g);
    println!("{}, {}", a, b);
}
