use crate::functions::*;
use crate::types::{Arr1d, Arr2d, Arr3d};
use crate::util::*;
use itertools::izip;
use ndarray::{Array, Array1, Array2, Array3, Axis, Dimension};

pub trait LayerWithLoss {
    fn predict(&self, input: Arr2d) -> Arr2d {
        unimplemented!()
    }
    /// (バッチ次元、入力次元)の入力inputに対し、(バッチ次元、出力次元)を返す。
    fn forward(&mut self, input: Arr2d, one_hot_target: &Arr2d) -> f32 {
        self.forward2(input, reverse_one_hot(one_hot_target))
    }
    fn forward2(&mut self, input: Arr2d, target: Array1<usize>) -> f32 {
        unimplemented!()
    }
    /// (バッチ次元、出力次元)で伝播してきた誤差doutに対し、(バッチ次元、入力次元)
    /// の誤差を後ろに渡す。
    fn backward(&mut self, batch_size: usize) -> Arr2d;
    fn new(wvec: &[Arr2d]) -> Self;
    fn params(&mut self) -> Vec<&mut Arr2d> {
        vec![]
    }
    fn grads(&self) -> Vec<Arr2d> {
        vec![]
    }
}

#[derive(Default)]
pub struct SoftMaxWithLoss {
    /// softmaxの出力、すなわち、ラベルの予測確率
    pred: Arr2d,
    // /// 誤差関数の出力
    // out: Arr1d,
    /// 教師ラベル
    target: Array1<usize>,
}
impl SoftMaxWithLoss {}
impl LayerWithLoss for SoftMaxWithLoss {
    fn predict(&self, input: Arr2d) -> Arr2d {
        softmax(input)
    }
    fn forward2(&mut self, input: Arr2d, target: Array1<usize>) -> f32 {
        self.pred = self.predict(input);
        self.target = target;
        cross_entropy_error_target(&self.pred, &self.target)
    }
    fn backward(&mut self, batch_size: usize) -> Arr2d {
        // let batch_size = dout.Arr1d; // Arr2dッチで平均されるので、各バッチの寄与は1/batch_size か?
        let dout: Arr2d = Array::from_elem((batch_size, 1), 1.0 / batch_size as f32);
        let mut dx = self.pred.clone(); // これを使う
        for (i, t) in self.target.iter().enumerate() {
            dx[[i, *t]] -= 1.0; // 誤差(正解ラベルでの確率は1なのでそれを引く)
        }
        // dx * dout / batch_size
        dx * dout // doutはバッチ次元なので、(バッチサイズ, 1)にしてdxとかけれるようにする。
    }
    fn new(wvec: &[Arr2d]) -> Self {
        Default::default()
    }
}

#[derive(Default)]
pub struct SigmodWithLoss<D: Dimension> {
    // input: Arr2d,
    y: Array<f32, D>,
    target: Array<bool, D>,
}
impl<D: Dimension> SigmodWithLoss<D> {
    // targetは正解ラベル
    pub fn forward(&mut self, input: Array<f32, D>, target: Array<bool, D>) -> f32 {
        self.y = input.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let mut loss = Array::zeros(self.y.dim());
        for (mut l, y, b) in izip!(loss.iter_mut(), self.y.iter(), target.iter()) {
            *l = if *b {
                -(y + 1e-7).ln()
            } else {
                -(1.0 - y).ln()
            }
        }
        self.target = target;
        loss.mean().unwrap()
    }
    pub fn backward(&mut self) -> Array<f32, D> {
        let batch_sample_size = self.y.len() as f32;
        (&self.y - &self.target.mapv(|b| if b { 1.0 } else { 0.0 })) / batch_sample_size
    }
}
