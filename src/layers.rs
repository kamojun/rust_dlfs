extern crate ndarray;
use super::functions::*;
use super::types::{arr1d, arr2d};
use ndarray::{Array, Array1, Array2, Array3, Axis};

/// ANNを構成するレイヤー。入力は一つのものを考える。
pub trait Layer {
    /// (バッチ次元、入力次元)の入力inputに対し、(バッチ次元、出力次元)を返す。
    fn forward(&mut self, input: arr2d) -> arr2d;
    /// (バッチ次元、出力次元)で伝播してきた誤差doutに対し、(バッチ次元、入力次元)
    /// の誤差を後ろに渡す。
    fn backward(&mut self, dout: arr2d) -> arr2d;
}

pub struct Sigmoid {
    out: arr2d,
}
impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid {
            out: Array::zeros((0, 0)),
        }
    }
}
pub struct Affine {
    W: arr2d,
    b: arr1d,
    x: arr2d,
    dW: arr2d,
    db: arr1d,
}
impl Affine {
    pub fn new(W: arr2d, b: arr1d) -> Self {
        Self {
            W,
            b,
            x: Array::zeros((0, 0)),
            dW: Array::zeros((0, 0)),
            db: Array::zeros((0,)),
        }
    }
}
pub struct SoftMax {
    out: arr2d,
}
impl SoftMax {
    pub fn new() -> Self {
        Self {
            out: Array::zeros((0, 0)),
        }
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: arr2d) -> arr2d {
        self.out = 1.0 / (1.0 + input.mapv(|a| (-a).exp()));
        self.out.clone()
    }
    fn backward(&mut self, dout: arr2d) -> arr2d {
        dout * (1.0 - &(self.out)) * &(self.out) // 自分が持っているものを計算に供出するには、参照渡しする?
    }
}
impl Layer for Affine {
    fn forward(&mut self, input: arr2d) -> arr2d {
        self.x = input;
        self.x.dot(&self.W) + &(self.b) // dotは自動的に参照渡し?
    }
    fn backward(&mut self, dout: arr2d) -> arr2d {
        // let dx = dout.dot(&self.W.t());
        self.dW = self.x.t().dot(&dout);
        self.db = dout.sum_axis(Axis(0));
        dout.dot(&self.W.t())
    }
}
impl Layer for SoftMax {
    fn forward(&mut self, input: arr2d) -> arr2d {
        self.out = softmax(input);
        self.out.clone()
    }
    fn backward(&mut self, dout: arr2d) -> arr2d {
        let outdout = &(self.out) * &dout; // 演算の中間に出てくる値
        outdout.clone() - (&self.out * &(outdout.sum_axis(Axis(1))))
    }
}

pub trait LayerWithLoss {
    /// (バッチ次元、入力次元)の入力inputに対し、(バッチ次元、出力次元)を返す。
    fn forward(&mut self, input: arr2d, one_hot_target: &arr2d) -> arr1d;
    /// (バッチ次元、出力次元)で伝播してきた誤差doutに対し、(バッチ次元、入力次元)
    /// の誤差を後ろに渡す。
    fn backward(&mut self, dout: arr1d) -> arr2d;
}
pub struct SoftMaxWithLoss {
    /// softmaxの出力、すなわち、ラベルの予測確率
    pred: arr2d,
    /// 誤差関数の出力
    out: arr1d,
    /// 教師ラベル
    target: Array1<usize>,
}
impl SoftMaxWithLoss {
    pub fn new() -> Self {
        Self {
            pred: Array::zeros((0, 0)),
            out: Array::zeros((0,)),
            target: Array::zeros((0,)),
        }
    }
}
impl LayerWithLoss for SoftMaxWithLoss {
    fn forward(&mut self, input: arr2d, one_hot_target: &arr2d) -> arr1d {
        self.pred = softmax(input);
        self.target = reverse_one_hot(one_hot_target);
        self.out = cross_entropy_error_target(&self.pred, &self.target);
        self.out.clone()
    }
    fn backward(&mut self, dout: arr1d) -> arr2d {
        let batch_size = dout.len(); // これいるかな??
        let mut dx = self.pred.clone(); // これを使う
        for (i, t) in self.target.iter().enumerate() {
            dx[[i, *t]] -= 1.0;
        }
        // dx * dout / batch_size
        dx * dout.insert_axis(Axis(1)) // doutはバッチ次元なので、(バッチサイズ, 1)にしてdxとかけれるようにする。
    }
}

/// 行列による掛け算
pub struct MatMul {
    /// (入力次元, チャンネル数)
    W: arr2d,
    /// (バッチ次元, 入力次元)
    x: arr2d,
    dx: arr2d,
    dW: arr2d,
}
impl Layer for MatMul {
    fn forward(&mut self, input: arr2d) -> arr2d {
        self.x = input;
        self.x.dot(&self.W)
    }
    fn backward(&mut self, dout: arr2d) -> arr2d {
        self.dx = dout.dot(&self.W.t());
        self.dW = self.x.t().dot(&dout);
        self.dx.clone()
    }
}
