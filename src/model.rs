use super::layers::*;
use super::types::*;
// extern crate ndarray;
use ndarray::{Array, Axis};
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;

pub struct TwoLayerNet {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    layers: [Box<dyn Layer>; 3],
    loss_layer: Box<dyn LayerWithLoss>,
}

fn randarr2d(m: usize, n: usize) -> arr2d {
    Array::<f32, _>::random((m, n), StandardNormal)
}
impl TwoLayerNet {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let W1 = randarr2d(input_size, hidden_size) * 0.01;
        let b1 = Array::zeros(hidden_size);
        let W2 = randarr2d(hidden_size, output_size);
        let b2 = Array::zeros(output_size);
        let affine1 = Affine::new(W1, b1);
        let sigmoid = Sigmoid::new();
        let affine2 = Affine::new(W2, b2);
        // let layers: [Box<Layer>; 3] = [Box::new(affine1), Box::new(sigmoid), Box::new(affine2)];
        Self {
            input_size,
            hidden_size,
            output_size,
            layers: [Box::new(affine1), Box::new(sigmoid), Box::new(affine2)],
            loss_layer: Box::new(SoftMaxWithLoss::new()),
        }
    }
    pub fn predict(&mut self, mut x: arr2d) -> arr2d {
        for layer in self.layers.iter_mut() {
            x = layer.forward(x);
        }
        x
    }
    pub fn forward(&mut self, x: arr2d, t: &arr2d) -> arr1d {
        let score = self.predict(x);
        self.loss_layer.forward(score, &t)
    }
    pub fn backward(&mut self, dout: arr1d) -> arr2d {
        let mut dx = self.loss_layer.backward(dout);
        for layer in self.layers.iter_mut() {
            dx = layer.backward(dx);
        }
        dx
    }
}
