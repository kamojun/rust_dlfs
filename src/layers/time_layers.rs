use crate::layers::*;
use crate::params::*;
use crate::types::*;
use ndarray::{Array1, Axis};

#[derive(Default)]
struct Cache {
    x: Arr2d,
    h_prev: Arr2d,
    h_next: Arr2d,
}

/// 外部入力 x: (b, c), wx: (c, h)
/// 相互入出力 h: (b, h), wh: (h, h)
/// 上位のTimeRNNではこれがtime_size個存在する
struct RNN<'a> {
    wx: &'a P1<Arr2d>,
    wh: &'a P1<Arr2d>,
    b: &'a P1<Arr1d>,
    cache: Cache,
}

impl<'a> RNN<'a> {
    pub fn forward(&mut self, x: Arr2d, h_prev: Arr2d) -> Arr2d {
        let t = h_prev.dot(&self.wh.p) + x.dot(&self.wx.p) + self.b.p.clone();
        let h_next = t.mapv(f32::tanh);
        self.cache = Cache {
            x,
            h_prev,
            h_next: h_next.clone(),
        };
        h_next
    }
    /// dx: (b, c)とdh_prev: (b, h)を返す
    pub fn backward(&mut self, dh_next: Arr2d) -> (Arr2d, Arr2d) {
        let dt = dh_next * self.cache.h_next.mapv(|y| 1.0 - y * y);
        let dh_prev = dt.dot(&self.wh.p.t());
        let dx = dt.dot(&self.wx.p.t());
        self.b.store(dt.sum_axis(Axis(0)));
        self.wh.store(self.cache.h_prev.t().dot(&dt));
        self.wx.store(self.cache.x.t().dot(&dt));
        (dx, dh_prev)
    }
}

/// (b,c)の外部入力と、(b,h)の内部相互入力を受け、(b,h)を出力する
/// time_sizeはlayerの数
pub struct TimeRNN<'a> {
    h: Arr2d,
    dh: Arr2d,
    stateful: bool,
    batch_size: usize,
    time_size: usize,
    channel_size: usize,
    hidden_size: usize,
    layers: Vec<RNN<'a>>,
}

impl<'a> TimeRNN<'a> {
    pub fn new(
        wx: &'a P1<Arr2d>,
        wh: &'a P1<Arr2d>,
        b: &'a P1<Arr1d>,
        batch_size: usize,
        time_size: usize,
    ) -> Self {
        let (channel_size, hidden_size) = wx.p.dim();
        let layers: Vec<_> = (0..time_size)
            .map(|_| RNN {
                wx: wx,
                wh: wh,
                b: b,
                cache: Default::default(),
            })
            .collect();
        Self {
            layers,
            h: Default::default(),
            dh: Default::default(),
            batch_size,
            time_size,
            channel_size,
            hidden_size,
            stateful: true,
        }
    }
    pub fn forward(&mut self, xs: Arr3d) -> Arr3d {
        let mut hs = Arr3d::zeros((self.batch_size, self.time_size, self.hidden_size));
        if !self.stateful || false {
            self.h = Arr2d::zeros((self.batch_size, self.hidden_size));
        }
        for (t, layer) in self.layers.iter_mut().enumerate() {
            self.h = layer.forward(xs.index_axis(Axis(1), t).to_owned(), self.h.clone());
            hs.index_axis_mut(Axis(1), t).assign(&self.h);
        }
        hs
    }
    /// dhs
    pub fn backward(&mut self, dhs: Arr3d) -> Arr3d {
        let mut dxs = Arr3d::zeros((self.batch_size, self.time_size, self.channel_size));
        /// 一番端っこのRNNではhはゼロ?
        let mut dh = Arr2d::zeros((self.batch_size, self.hidden_size));
        for (t, layer) in self.layers.iter_mut().enumerate().rev() {
            let (_dx, _dh) = layer.backward(dhs.index_axis(Axis(1), t).to_owned() + dh);
            dh = _dh;
            dxs.index_axis_mut(Axis(1), t).assign(&_dx);
        }
        self.dh = dh;
        dxs
    }
    pub fn set_state(&mut self, h: Arr2d) {
        self.h = h;
    }
    pub fn reset_state(&mut self) {
        self.h = Default::default();
    }
}

pub struct TimeEmbedding<'a> {
    w: &'a P1<Arr2d>,   // <- グローバルパラメータ(更新、学習が必要なもの)
    idx: Array2<usize>, // <- ローカルパラメータ(直接保持する)
}
impl<'a> TimeEmbedding<'a> {
    pub fn new(w: &'a P1<Arr2d>) -> Self {
        Self {
            w,
            idx: Default::default(),
        }
    }
    pub fn forward(&mut self, idx: Array2<usize>) -> Arr3d {
        let (batch_size, time_size) = idx.dim();
        let channel_num = self.w.p.dim().1;
        let mut out: Arr3d = Array3::zeros((batch_size, time_size, channel_num));
        for (mut _o, _x) in out.outer_iter_mut().zip(idx.outer_iter()) {
            _o.assign(&pickup1(&self.w.p, Axis(0), _x));
        }
        self.idx = idx;
        out
    }
    pub fn backward(&mut self, dout: Arr3d) {
        let mut dw = Array2::zeros(self.w.p.dim());
        // バッチ方向に回す
        for (_o, _x) in dout.outer_iter().zip(self.idx.outer_iter()) {
            for (__o, __x) in _o.outer_iter().zip(_x.iter()) {
                let mut row = dw.index_axis_mut(Axis(0), *__x);
                row += &__o;
            }
        }
        self.w.store(dw);
    }
}

pub struct TimeAffine<'a> {
    w: &'a P1<Arr2d>,
    b: &'a P1<Arr1d>,
    x: Arr3d,
}
fn dot(x: &Arr3d, w: &Arr2d) -> Arr3d {
    let (j, k, l) = x.dim();
    let (m, n) = w.dim();
    assert_eq!(l, m, "x.dim.2 and w.dim.0 must coinside!");
    x.to_owned()
        .into_shape((j * k, l))
        .unwrap()
        .dot(w)
        .into_shape((j, k, n))
        .unwrap()
}

impl<'a> TimeAffine<'a> {
    pub fn new(&mut self, w: &'a P1<Arr2d>, b: &'a P1<Arr1d>) -> Self {
        Self {
            w,
            b,
            x: Default::default(),
        }
    }
    pub fn forward(&mut self, x: Arr3d) -> Arr3d {
        self.x = x;
        dot(&self.x, &self.w.p) + &self.b.p
    }
    pub fn backward(&mut self, dout: Arr3d) -> Arr3d {
        let (b, t, c_out) = dout.dim(); // batch, time, channel
        let c_in = self.x.dim().2;
        let dout = dout.into_shape((b * t, c_out)).unwrap();
        let rx = self
            .x
            .clone()
            .into_shape((b * t, c_in))
            .expect("self.x and dout must have same dimention!");
        self.b.store(dout.sum_axis(Axis(0)));
        self.w.store(rx.t().dot(&dout));
        dout.dot(&self.w.p.t()).into_shape((b, t, c_in)).unwrap()
    }
}

pub struct TimeSoftmaxWithLoss {}
impl TimeSoftmaxWithLoss {
    fn forward(&mut self, xs: Arr3d, ts: Array2<usize>) -> f32 {
        0.0
    }
}
