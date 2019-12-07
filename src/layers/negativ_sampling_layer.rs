extern crate ndarray;
use super::loss_layer::{LayerWithLoss, SigmodWithLoss};
use crate::functions::*;
use crate::layers::{Embedding1d, Embedding2d};
use crate::model::Model;
use crate::types::{Arr1d, Arr2d, Arr3d};
use crate::util::*;
use itertools::izip;
use ndarray::{Array, Array1, Array2, Array3, Axis, Ix2};

pub struct EmbeddingDot {
    /// 単語の表現ベクトルの束(2次元行列)を持っていて、基本的にただそこから抜き出してくるだけ
    embed: Embedding1d,
    input: Arr2d,
    target_w: Arr2d,
}
impl EmbeddingDot {
    /// input: (bs, ch), idx: (bs,) <- 正解インデックスを想定
    fn forward(&mut self, input: Arr2d, idx: Array1<usize>) -> Arr1d {
        self.target_w = self.embed.forward(idx);
        self.input = input;
        (&self.input * &self.target_w).sum_axis(Axis(1)) // 要するに各列で内積を取る
    }
    fn backward(&mut self, dout: Arr1d) -> Arr2d {
        // let dout = dout.into_shape((dout.len(), 1)).unwrap();
        let dout = dout.insert_axis(Axis(1));
        let dtarget_w = &self.input * &dout; // target_wの方は、inputから勾配を得る
        self.embed.backward(dtarget_w); // 元のwに埋め込む
        &self.target_w * &dout // inputの勾配はtarget_wになる
    }
}

struct EmbeddingDot2d {
    /// (word_num, ch)
    embed: Embedding2d,
    /// (bs, 1, ch)
    input: Arr3d,
    /// (bs, sn, ch)
    target_w: Arr3d,
}
impl EmbeddingDot2d {
    /// input: (bs, ch), idx: (bs, samplnum), output: (bs, samplnum)
    /// 各行で、idxによりwからサンプリングして、inputと掛ける
    fn forward(&mut self, input: Arr2d, idx: Array2<usize>) -> Arr2d {
        let (batch_size, channel_num) = input.dim();
        self.target_w = self.embed.forward(idx); // (bs, sn, ch)
        self.input = input.into_shape((batch_size, 1, channel_num)).unwrap();
        let out = (&self.target_w * &self.input).sum_axis(Axis(2)); // 要するに各列で内積を取る
        out
    }
    fn backward(&mut self, dout: Arr2d) -> Arr2d {
        let (batch_size, sample_num) = dout.dim();
        let dout = dout.insert_axis(Axis(2));
        // let dtarget_w = self.input * dout; // target_wの方は、inputから勾配を得る
        let dtarget_w = Array::from_shape_fn(self.target_w.dim(), |(i, j, k)| {
            self.input[[i, 0, k]] * dout[[i, j, 0]]
        });
        self.embed.backward(dtarget_w); // 元のwに埋め込む
        (&self.target_w * &dout).sum_axis(Axis(1)) // sample_num方向に潰す
    }
    fn new(w: Arr2d) -> Self {
        Self {
            embed: Embedding2d::new(w),
            input: Default::default(),
            target_w: Default::default(),
        }
    }
    fn params(&mut self) -> Vec<&mut Arr2d> {
        self.embed.params()
    }
    fn grads(&self) -> Vec<Arr2d> {
        self.embed.grads()
    }
}

struct Sampler {
    sample_size: usize,
}
impl Sampler {
    fn sampling(&self, target: Array1<usize>) -> (Array2<usize>, Array2<bool>) {
        unimplemented!();
    }
    fn new() -> Self {
        unimplemented!();
    }
}

pub struct NegativeSamplingLoss {
    sample_size: usize,
    sampler: Sampler,
    loss_layer: SigmodWithLoss<Ix2>,
    embed: EmbeddingDot2d,
}
impl LayerWithLoss for NegativeSamplingLoss {
    fn forward2(&mut self, input: Arr2d, target: Array1<usize>) -> f32 {
        // let batch_size = target.shape()[0];
        let (target_and_negative_sample, label) = self.sampler.sampling(target);
        let out = self.embed.forward(input, target_and_negative_sample);
        self.loss_layer.forward(out, label)
    }
    fn backward(&mut self, unused: usize) -> Arr2d {
        let mut dx = self.loss_layer.backward();
        dx = self.embed.backward(dx);
        dx
    }
    fn new(ws: &[Arr2d]) -> Self {
        Self {
            sample_size: 0,
            sampler: Sampler::new(),
            loss_layer: Default::default(),
            embed: EmbeddingDot2d::new(ws[0].clone()),
        }
    }
    fn params(&mut self) -> Vec<&mut Arr2d> {
        self.embed.params()
    }
    fn grads(&self) -> Vec<Arr2d> {
        self.embed.grads()
    }
}
