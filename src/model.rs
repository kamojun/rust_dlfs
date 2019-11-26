use super::layers::*;
use super::types::*;
// extern crate ndarray;
use itertools::concat;
use ndarray::{Array, Axis};
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;

pub trait Model {
    /// 最後のloss_layerについて、勾配計算は目的とせず、ただ学習に基づく予想を出力させる
    fn predict(&mut self, mut x: Arr2d) -> Arr2d;
    /// 最後まで進める
    fn forward(&mut self, x: Arr2d, t: &Arr2d) -> f32;
    /// 誤差逆伝播させる
    fn backward(&mut self, batch_size: usize) -> Arr2d;
    fn params1d(&mut self) -> Vec<&mut Arr1d>;
    fn params2d(&mut self) -> Vec<&mut Arr2d>;
    fn grads1d(&self) -> Vec<Arr1d>;
    fn grads2d(&self) -> Vec<Arr2d>;
}

pub struct TwoLayerNet {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    layers: [Box<dyn Layer>; 3],
    loss_layer: Box<dyn LayerWithLoss>,
}

fn randarr2d(m: usize, n: usize) -> Arr2d {
    Array::<f32, _>::random((m, n), StandardNormal)
}
impl TwoLayerNet {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let w1 = randarr2d(input_size, hidden_size) * 0.01;
        let b1 = Array::zeros(hidden_size);
        let w2 = randarr2d(hidden_size, output_size);
        let b2 = Array::zeros(output_size);
        let affine1 = Affine::new(w1, b1);
        let sigmoid = Sigmoid::new();
        let affine2 = Affine::new(w2, b2);
        let layers: [Box<dyn Layer>; 3] = [Box::new(affine1), Box::new(sigmoid), Box::new(affine2)];
        Self {
            input_size,
            hidden_size,
            output_size,
            layers,
            loss_layer: Box::new(SoftMaxWithLoss::new()),
        }
    }
}
impl Model for TwoLayerNet {
    fn predict(&mut self, mut x: Arr2d) -> Arr2d {
        for layer in self.layers.iter_mut() {
            x = layer.forward(x);
        }
        self.loss_layer.predict(x)
    }
    fn forward(&mut self, mut x: Arr2d, t: &Arr2d) -> f32 {
        for layer in self.layers.iter_mut() {
            x = layer.forward(x);
        }
        self.loss_layer.forward(x, &t)
    }
    fn backward(&mut self, batch_size: usize) -> Arr2d {
        let mut dx = self.loss_layer.backward(batch_size);
        for layer in self.layers.iter_mut().rev() {
            dx = layer.backward(dx);
        }
        dx
    }
    fn params1d(&mut self) -> Vec<&mut Arr1d> {
        concat(self.layers.iter_mut().map(|l| l.params1d()))
    }
    fn params2d(&mut self) -> Vec<&mut Arr2d> {
        concat(self.layers.iter_mut().map(|l| l.params2d()))
    }
    fn grads1d(&self) -> Vec<Arr1d> {
        concat(self.layers.iter().map(|l| l.grads1d()))
    }
    fn grads2d(&self) -> Vec<Arr2d> {
        concat(self.layers.iter().map(|l| l.grads2d()))
    }
}
