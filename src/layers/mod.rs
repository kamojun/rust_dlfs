extern crate ndarray;
use super::functions::*;
use super::types::{Arr1d, Arr2d, Arr3d};
use crate::util::*;
use itertools::izip;
use ndarray::{Array, Array1, Array2, Array3, Axis, Dimension};
pub mod loss_layer;
pub mod negativ_sampling_layer;

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

#[derive(Default)]
pub struct Sigmoid {
    out: Arr2d,
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
            x: Default::default(),
            dw: Default::default(),
            db: Default::default(),
        }
    }
}
#[derive(Default)]
pub struct SoftMax {
    out: Arr2d,
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
            x: Default::default(),
            dw: Default::default(),
        }
    }
    pub fn new_from_size(in_size: usize, out_size: usize, scale: Option<f32>) -> Self {
        // let scale = scale.(1.0);
        let scale = scale.unwrap_or(1.0);
        Self {
            w: randarr2d(in_size, out_size) * scale,
            x: Default::default(),
            dw: Default::default(),
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
        let (batch_size, context_len) = self.idx.dim();
        let mut out = Array2::zeros((batch_size, self.w.shape()[1]));
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
        out / (batch_size as f32) // 加算した後平均をとる
    }
    pub fn backward(&mut self, dout: Arr2d) {
        self.dw = Array2::zeros(self.w.dim());
        // まずバッチ方向に回す
        for (_o, _x) in dout.outer_iter().zip(self.idx.outer_iter()) {
            // _xにはどの単語が出現したか記録されている。
            // その単語id__xを見て、self.wの行に_oを加算する。
            for __x in _x.iter() {
                let mut row = self.dw.index_axis_mut(Axis(0), *__x);
                row += &_o;
            }
        }
        self.dw /= dout.shape()[0] as f32;
    }
    pub fn params(&mut self) -> Vec<&mut Arr2d> {
        vec![&mut self.w]
    }
    pub fn grads(&self) -> Vec<Arr2d> {
        vec![self.dw.clone()]
    }
    pub fn new(w: Arr2d) -> Self {
        Self {
            w,
            dw: Default::default(),
            idx: Default::default(),
        }
    }
}

pub struct Embedding1d {
    /// 語彙数、hidden_size
    w: Arr2d,
    dw: Arr2d,
    // batch_size (単語id: usizeからなる配列)
    idx: Array1<usize>,
}
/// 出力はbatchsize, hiddensize
/// 入力idxは(batchsize, contextlen)で各行は、出現した単語のidである
/// 例えば[3,1,2,1,5] -> [2,1,1,0,5,0](語彙数6の場合)というような変形ができる
impl Embedding1d {
    pub fn forward(&mut self, idx: Array1<usize>) -> Arr2d {
        self.idx = idx;
        pickup1(&self.w, Axis(0), self.idx.view())
    }
    pub fn backward(&mut self, dout: Arr2d) {
        self.dw = Array2::zeros(self.w.dim());
        // まずバッチ方向に回す
        for (_o, _x) in dout.outer_iter().zip(self.idx.iter()) {
            let mut row = self.dw.index_axis_mut(Axis(0), *_x);
            row += &_o;
        }
    }
}

pub struct Embedding2d {
    /// 語彙数, channel_num
    w: Arr2d,
    dw: Arr2d,
    // (batch_size, sample_num) (単語id: usizeからなる行列)
    idx: Array2<usize>,
}
/// 出力はbatchsize, sample_num, channel_num
/// 入力idx(batchsize, sampl_num)の各行は、抽出した単語のidである
impl Embedding2d {
    pub fn forward(&mut self, idx: Array2<usize>) -> Arr3d {
        let (batch_size, sample_num) = idx.dim();
        let channel_num = self.w.dim().1;
        let mut out: Arr3d = Array3::zeros((batch_size, sample_num, channel_num));
        for (mut _o, _x) in out.outer_iter_mut().zip(idx.outer_iter()) {
            _o.assign(&pickup1(&self.w, Axis(0), _x));
        }
        self.idx = idx;
        out
    }
    /// dout: (batch_size, sample_num, channel_num)
    pub fn backward(&mut self, dout: Arr3d) {
        self.dw = Array2::zeros(self.w.dim());
        // バッチ方向に回す
        for (_o, _x) in dout.outer_iter().zip(self.idx.outer_iter()) {
            for (__o, __x) in _o.outer_iter().zip(_x.iter()) {
                let mut row = self.dw.index_axis_mut(Axis(0), *__x); // TODO ADDASIGN?
                row += &__o;
            }
        }
    }
    fn params(&mut self) -> Vec<&mut Arr2d> {
        vec![&mut self.w]
    }
    fn grads(&self) -> Vec<Arr2d> {
        vec![self.dw.clone()]
    }
    fn new(w: Arr2d) -> Self {
        Self {
            w,
            dw: Default::default(),
            idx: Default::default(),
        }
    }
}
