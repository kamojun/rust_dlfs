use super::layers::*;
use super::types::*;
use crate::layers::loss_layer::*;
use crate::layers::negativ_sampling_layer::*;
use crate::util::*;
// extern crate ndarray;
use itertools::concat;
use ndarray::{Array, Array1, Axis, Dimension, Ix2, Ix3};
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;

pub trait Model2 {
    fn forward<D: Dimension>(&mut self, input: Array<usize, D>, target: Array1<usize>) -> f32 {
        unimplemented!();
    }
    fn backward(&mut self) {}
    fn params(&mut self) -> Vec<&mut Arr2d> {
        vec![]
    }
    fn grads(&self) -> Vec<Arr2d> {
        vec![]
    }
}

pub struct CBOW {
    in_layer: Embedding,
    loss_layer: NegativeSamplingLoss,
}
impl Model2 for CBOW {
    fn forward<D: Dimension>(&mut self, input: Array<usize, D>, target: Array1<usize>) -> f32 {
        let input = input
            .into_dimensionality::<Ix2>()
            .expect("CBOW: input size must be dim2");
        let h = self.in_layer.forward(input);
        self.loss_layer.forward2(h, target)
    }
    fn backward(&mut self) {
        let dx = self.loss_layer.backward(); // batch_sizeは使わない
        self.in_layer.backward(dx);
    }
    fn params(&mut self) -> Vec<&mut Arr2d> {
        concat(vec![self.in_layer.params(), self.loss_layer.params()])
    }
    fn grads(&self) -> Vec<Arr2d> {
        concat(vec![self.in_layer.grads(), self.loss_layer.grads()])
    }
}

use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;
impl InitWithSampler for CBOW {
    fn new(ws: &[Arr2d], sample_size: usize, distribution: WeightedIndex<f32>) -> Self {
        Self {
            in_layer: Embedding::new(ws[0].clone()),
            loss_layer: NegativeSamplingLoss::new(ws[1].clone(), sample_size, distribution),
        }
    }
}

pub trait Model {
    /// 最後のloss_layerについて、勾配計算は目的とせず、ただ学習に基づく予想を出力させる
    fn predict(&mut self, mut x: Arr2d) -> Arr2d {
        unimplemented!();
    }
    /// 最後まで進める
    /// ターゲットはone_hot
    fn forward(&mut self, x: Arr2d, t: &Arr2d) -> f32 {
        unimplemented!();
    }
    /// ターゲットはラベルベクトル
    fn forwardt(&mut self, x: Arr2d, t: &Array1<usize>) -> f32 {
        unimplemented!();
    }
    fn forwardx<D: Dimension>(&mut self, x: Array<f32, D>, t: Arr2d) -> f32;
    /// 誤差逆伝播させる
    fn backward(&mut self, batch_size: usize);
    fn params1d(&mut self) -> Vec<&mut Arr1d> {
        Vec::new()
    }
    fn params2d(&mut self) -> Vec<&mut Arr2d>;
    fn grads1d(&self) -> Vec<Arr1d> {
        Vec::new()
    }
    fn grads2d(&self) -> Vec<Arr2d>;
}

pub struct TwoLayerNet<L: LayerWithLoss + Default> {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    layers: [Box<dyn Layer>; 3],
    // loss_layer: Box<dyn LayerWithLoss>,
    loss_layer: L,
}
impl<L: LayerWithLoss + Default> TwoLayerNet<L> {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        const PARAM_INIT_SCALE: f32 = 1.0; // これ変えると微妙に学習効率変わるのだが、よくわからん
        let w1 = randarr2d(input_size, hidden_size) * PARAM_INIT_SCALE;
        let b1 = randarr1d(hidden_size) * PARAM_INIT_SCALE;
        let w2 = randarr2d(hidden_size, output_size) * PARAM_INIT_SCALE;
        let b2 = randarr1d(output_size) * PARAM_INIT_SCALE;
        let affine1 = Affine::new(w1, b1);
        let sigmoid: Sigmoid = Default::default();
        let affine2 = Affine::new(w2, b2);
        let layers: [Box<dyn Layer>; 3] = [Box::new(affine1), Box::new(sigmoid), Box::new(affine2)];
        Self {
            input_size,
            hidden_size,
            output_size,
            layers,
            loss_layer: L::default(),
        }
    }
}
impl<L: LayerWithLoss + Default> Model for TwoLayerNet<L> {
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
    fn forwardx<D: Dimension>(&mut self, x: Array<f32, D>, t: Arr2d) -> f32 {
        let mut x = x
            .into_dimensionality::<Ix2>()
            .expect("failed in converting to arr2d");
        for layer in self.layers.iter_mut() {
            x = layer.forward(x);
        }
        self.loss_layer.forward(x, &t)
    }
    fn backward(&mut self, batch_size: usize) {
        let mut dx = self.loss_layer.backward();
        for layer in self.layers.iter_mut().rev() {
            dx = layer.backward(dx);
        }
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

pub struct SimpleCBOW<L: LayerWithLoss + Default> {
    vocab_size: usize,
    hidden_size: usize,
    // layers: [Box<dyn Layer>; 3],
    loss_layer: L,
    in_layer_1: MatMul,
    in_layer_2: MatMul,
    out_layer: MatMul,
}
impl<L: LayerWithLoss + Default> SimpleCBOW<L> {
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        let (_v, _h) = (vocab_size, hidden_size);
        let scale = Some(0.01);
        const PARAM_INIT_SCALE: f32 = 0.01;
        let in_layer_1 = MatMul::new_from_size(_v, _h, scale);
        let in_layer_2 = MatMul::new_from_size(_v, _h, scale);
        let out_layer = MatMul::new_from_size(_h, _v, scale);
        Self {
            vocab_size,
            hidden_size,
            loss_layer: L::default(),
            in_layer_1,
            in_layer_2,
            out_layer,
        }
    }
    pub fn layers(&mut self) -> Vec<&mut dyn Layer> {
        vec![
            &mut self.in_layer_1,
            &mut self.in_layer_2,
            &mut self.out_layer,
        ]
    }
    pub fn word_vecs(&self) -> Arr2d {
        self.in_layer_1.w.clone()
    }
}
impl<L: LayerWithLoss + Default> Model for SimpleCBOW<L> {
    fn forwardx<D: Dimension>(&mut self, contexts: Array<f32, D>, target: Arr2d) -> f32 {
        let x = contexts
            .into_dimensionality::<Ix3>()
            .expect("contexts array must be dim3");
        let h0 = self.in_layer_1.forward(x.index_axis(Axis(1), 0).to_owned());
        let h1 = self.in_layer_2.forward(x.index_axis(Axis(1), 1).to_owned());
        let h = (h0 + h1) * 0.5;
        let score = self.out_layer.forward(h);
        self.loss_layer.forward(score, &target)
    }
    fn backward(&mut self, batch_size: usize) {
        let mut dx = self.loss_layer.backward();
        dx = self.out_layer.backward(dx);
        dx *= 0.5; // in1, in2の入力を平均する設定になっているためforwardx<D>参照
        self.in_layer_1.backward(dx.clone());
        self.in_layer_2.backward(dx);
    }
    fn params2d(&mut self) -> Vec<&mut Arr2d> {
        // イテレータ使えず...
        // concat(self.layers().iter_mut().map(|l| l.params2d()))
        // concat([&mut self.in_layer_1, &mut self.in_layer_2, &mut self.out_layer].into_iter().map(|l| l.grads2d()))
        // vec![self.in_layer_1.grads2d()[0]]
        // self.in_layer_1.grads2d()[0];
        let mut layers = self.layers();
        // concat(layers.iter_mut().map(|l| l.params2d()))
        // concat([&mut self.in_layer_1, &mut self.in_layer_2, &mut self.out_layer].iter_mut().map(|l| l.params2d()).collect::<Vec<Vec<&mut Arr2d>>>());
        // let c = concat(self.layers().iter_mut().map(|l| l.params2d()).collect::<Vec<_>>());
        concat(vec![
            self.in_layer_1.params2d(),
            self.in_layer_2.params2d(),
            self.out_layer.params2d(),
        ])
    }
    fn grads2d(&self) -> Vec<Arr2d> {
        concat(vec![
            self.in_layer_1.grads2d(),
            self.in_layer_2.grads2d(),
            self.out_layer.grads2d(),
        ])
    }
}
