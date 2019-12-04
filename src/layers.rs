extern crate ndarray;
use super::functions::*;
use super::types::{Arr1d, Arr2d};
use crate::util::*;
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
    fn grads1d(&self) -> Vec<Arr1d> {
        Vec::new()
    }
    fn params2d(&mut self) -> Vec<&mut Arr2d> {
        Vec::new()
    }
    fn grads2d(&self) -> Vec<Arr2d> {
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
    fn grads1d(&self) -> Vec<Arr1d> {
        vec![self.db.clone()]
    }
    fn grads2d(&self) -> Vec<Arr2d> {
        vec![self.dw.clone()]
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
    fn predict(&self, input: Arr2d) -> Arr2d;
    /// (バッチ次元、入力次元)の入力inputに対し、(バッチ次元、出力次元)を返す。
    fn forward(&mut self, input: Arr2d, one_hot_target: &Arr2d) -> f32;
    /// (バッチ次元、出力次元)で伝播してきた誤差doutに対し、(バッチ次元、入力次元)
    /// の誤差を後ろに渡す。
    fn backward(&mut self, batch_size: usize) -> Arr2d;
    fn new() -> Self;
}
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
    fn forward(&mut self, input: Arr2d, one_hot_target: &Arr2d) -> f32 {
        self.pred = self.predict(input);
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
            dx[[i, *t]] -= 1.0; // 誤差(正解ラベルでの確率は1なのでそれを引く)
        }
        // dx * dout / batch_size
        dx * dout // doutはバッチ次元なので、(バッチサイズ, 1)にしてdxとかけれるようにする。
    }
    fn new() -> Self {
        Self {
            pred: Array::zeros((0, 0)),
            target: Array::zeros((0,)),
        }
    }
}

struct SigmodWithLoss {}
struct EmbeddingDot {}

pub struct NegativeSamplingLoss {
    sample_size: usize,
    // sampler,
    loss_layers: [SigmodWithLoss; 6],
    embed_loss_layers: [EmbeddingDot; 6],
}

/// 行列による掛け算
pub struct MatMul {
    /// (入力次元, チャンネル数)
    pub w: Arr2d,
    /// (バッチ次元, 入力次元)
    x: Arr2d,
    dw: Arr2d,
}
impl MatMul {
    pub fn new(w: Arr2d) -> Self {
        Self {
            w,
            x: Array::zeros((0, 0)),
            dw: Array::zeros((0, 0)),
        }
    }
    pub fn new_from_size(in_size: usize, out_size: usize, scale: Option<f32>) -> Self {
        // let scale = scale.(1.0);
        let scale = scale.unwrap_or(1.0);
        Self {
            w: randarr2d(in_size, out_size) * scale,
            x: Array::zeros((0, 0)),
            dw: Array::zeros((0, 0)),
        }
    }
}
impl Layer for MatMul {
    fn forward(&mut self, input: Arr2d) -> Arr2d {
        self.x = input;
        self.x.dot(&self.w)
    }
    fn backward(&mut self, dout: Arr2d) -> Arr2d {
        self.dw = self.x.t().dot(&dout);
        dout.dot(&self.w.t())
    }
    fn params2d(&mut self) -> Vec<&mut Arr2d> {
        vec![&mut self.w]
    }
    fn grads2d(&self) -> Vec<Arr2d> {
        vec![self.dw.clone()]
    }
}

pub struct Embedding {
    /// 語彙数、hidden_size
    w: Arr2d,
    dw: Arr2d,
    // batch_size, context_size(単語id: usizeからなる配列)
    idx: Array2<usize>,
}
/// 出力はbatchsize, hiddensize
/// 入力idxは(batchsize, contextlen)で各行は、出現した単語のidである
/// 例えば[3,1,2,1,5] -> [2,1,1,0,5,0](語彙数6の場合)というような変形ができる
impl Embedding {
    pub fn forward(&mut self, idx: Array2<usize>) -> Arr2d {
        self.idx = idx;
        let mut out = Array2::zeros((self.idx.shape()[0], self.w.shape()[1]));
        // out, idx共にまずbatch方向に回す
        for (mut _o, _x) in out.outer_iter_mut().zip(self.idx.outer_iter()) {
            // _xにはどの単語が出現したが記録されている
            for __x in _x.iter() {
                // 単語__xが出現したら、そのベクトルを_oに加算する
                // 同じ単語が複数回出現していたら、例えば2回足さずに2倍して1回足す方が早いだろうが、
                // このループはせいぜい10回とかなので、意味ないだろう。
                // それより長さ10を走査する方が時間かかりそう,,,
                // でもないか??
                _o += &self.w.index_axis(Axis(0), *__x);
            }
        }
        out
    }
    pub fn backward(&mut self, dout: Arr2d) {
        self.dw = Array2::zeros(self.w.dim());
        // まずバッチ方向に回す
        for (_o, _x) in dout.outer_iter().zip(self.idx.outer_iter()) {
            // _xにはどの単語が出現したか記録されている。
            // その単語id__xを見て、self.wの行に_oを加算する。
            for __x in _x.iter() {
                self.dw.index_axis_mut(Axis(0), *__x).assign(&_o);
            }
        }
    }
}
