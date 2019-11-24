extern crate ndarray;
use super::functions::*;
use super::types::{Arr1d, Arr2d};
use ndarray::{Array, Array1, Array2, Array3, Axis};

/// ANNを構成するレイヤー。入力は一つのものを考える。
pub trait Layer {
    /// (バッチ次元、入力次元)の入力inputに対し、(バッチ次元、出力次元)を返す。
    fn forward(&mut self, input: Arr2d) -> Arr2d;
    /// (バッチ次元、出力次元)で伝播してきた誤差doutに対し、(バッチ次元、入力次元)
    /// の誤差を後ろに渡す。
    fn backward(&mut self, dout: Arr2d) -> Arr2d;
    fn params1d(&mut self) -> Vec<&mut Arr1d> {
        Vec::new()
    }
    fn grads1d(&self) -> Vec<&Arr1d> {
        Vec::new()
    }
    fn params2d(&mut self) -> Vec<&mut Arr2d> {
        Vec::new()
    }
    fn grads2d(&self) -> Vec<&Arr2d> {
        Vec::new()
    }
}

pub struct Sigmoid {
    out: Arr2d,
}
impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid {
            out: Array::zeros((0, 0)),
        }
    }
}
pub struct Affine {
    w: Arr2d,
    b: Arr1d,
    x: Arr2d,
    dw: Arr2d,
    db: Arr1d,
}
impl Affine {
    pub fn new(w: Arr2d, b: Arr1d) -> Self {
        Self {
            w,
            b,
            x: Array::zeros((0, 0)),
            dw: Array::zeros((0, 0)),
            db: Array::zeros((0,)),
        }
    }
}
pub struct SoftMax {
    out: Arr2d,
}
impl SoftMax {
    pub fn new() -> Self {
        Self {
            out: Array::zeros((0, 0)),
        }
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: Arr2d) -> Arr2d {
        self.out = 1.0 / (1.0 + input.mapv(|a| (-a).exp()));
        self.out.clone()
    }
    fn backward(&mut self, dout: Arr2d) -> Arr2d {
        dout * (1.0 - &(self.out)) * &(self.out) // 自分が持っているものを計算に供出するには、参照渡しする?
    }
}
impl Layer for Affine {
    fn forward(&mut self, input: Arr2d) -> Arr2d {
        self.x = input;
        self.x.dot(&self.w) + &(self.b) // dotは自動的に参照渡し?
    }
    fn backward(&mut self, dout: Arr2d) -> Arr2d {
        // let dx = dout.dot(&self.w.t());
        self.dw = self.x.t().dot(&dout);
        self.db = dout.sum_axis(Axis(0));
        dout.dot(&self.w.t())
    }
    fn params1d(&mut self) -> Vec<&mut Arr1d> {
        vec![&mut self.b]
    }
    fn params2d(&mut self) -> Vec<&mut Arr2d> {
        vec![&mut self.w]
    }
    fn grads1d(&self) -> Vec<&Arr1d> {
        vec![&self.db]
    }
    fn grads2d(&self) -> Vec<&Arr2d> {
        vec![&self.dw]
    }
}
impl Layer for SoftMax {
    fn forward(&mut self, input: Arr2d) -> Arr2d {
        self.out = softmax(input);
        self.out.clone()
    }
    fn backward(&mut self, dout: Arr2d) -> Arr2d {
        let outdout = &(self.out) * &dout; // 演算の中間に出てくる値
        outdout.clone() - (&self.out * &(outdout.sum_axis(Axis(1))))
    }
}

pub trait LayerWithLoss {
    /// (バッチ次元、入力次元)の入力inputに対し、(バッチ次元、出力次元)を返す。
    fn forward(&mut self, input: Arr2d, one_hot_target: &Arr2d) -> f32;
    /// (バッチ次元、出力次元)で伝播してきた誤差doutに対し、(バッチ次元、入力次元)
    /// の誤差を後ろに渡す。
    fn backward(&mut self, batch_size: usize) -> Arr2d;
}
pub struct SoftMaxWithLoss {
    /// softmaxの出力、すなわち、ラベルの予測確率
    pred: Arr2d,
    // /// 誤差関数の出力
    // out: Arr1d,
    /// 教師ラベル
    target: Array1<usize>,
}
impl SoftMaxWithLoss {
    pub fn new() -> Self {
        Self {
            pred: Array::zeros((0, 0)),
            target: Array::zeros((0,)),
        }
    }
}
impl LayerWithLoss for SoftMaxWithLoss {
    fn forward(&mut self, input: Arr2d, one_hot_target: &Arr2d) -> f32 {
        self.pred = softmax(input);
        self.target = reverse_one_hot(one_hot_target);
        // self.out = cross_entropy_error_target(&self.pred, &self.target);
        // self.out.clone()
        cross_entropy_error_target(&self.pred, &self.target)
    }
    fn backward(&mut self, batch_size: usize) -> Arr2d {
        // let batch_size = dout.Arr1d; // Arr2dッチで平均されるので、各バッチの寄与は1/batch_size か?
        let dout: Arr2d = Array::from_elem((batch_size, 1), 1.0 / batch_size as f32);
        let mut dx = self.pred.clone(); // これを使う
        for (i, t) in self.target.iter().enumerate() {
            dx[[i, *t]] -= 1.0; // 誤差
        }
        // dx * dout / batch_size
        dx * dout // doutはバッチ次元なので、(バッチサイズ, 1)にしてdxとかけれるようにする。
    }
}

/// 行列による掛け算
pub struct MatMul {
    /// (入力次元, チャンネル数)
    W: Arr2d,
    /// (バッチ次元, 入力次元)
    x: Arr2d,
    dx: Arr2d,
    dW: Arr2d,
}
impl Layer for MatMul {
    fn forward(&mut self, input: Arr2d) -> Arr2d {
        self.x = input;
        self.x.dot(&self.W)
    }
    fn backward(&mut self, dout: Arr2d) -> Arr2d {
        self.dx = dout.dot(&self.W.t());
        self.dW = self.x.t().dot(&dout);
        self.dx.clone()
    }
}
