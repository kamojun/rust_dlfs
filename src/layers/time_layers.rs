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
pub struct _RNN<'a> {
    _wx: &'a P1<Arr2d>,
    _wh: &'a P1<Arr2d>,
    _b: &'a P1<Arr1d>,
    cache: Cache,
}

impl<'a> _RNN<'a> {
    pub fn forward(&mut self, x: Arr2d, h_prev: Arr2d) -> Arr2d {
        let t = h_prev.dot(&self._wh.p) + x.dot(&self._wx.p) + self._b.p.clone();
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
        let dh_prev = dt.dot(&self._wh.p.t());
        let dx = dt.dot(&self._wx.p.t());
        self._b.store(dt.sum_axis(Axis(0)));
        self._wh.store(self.cache.h_prev.t().dot(&dt));
        self._wx.store(self.cache.x.t().dot(&dt));
        (dx, dh_prev)
    }
}

struct RNN<'a> {
    /// (D, H) x(入力)を変換する
    wx: &'a Arr2d,
    /// (H, H) h_prevを変換する
    wh: &'a Arr2d,
    /// (H, ) 定数項
    b: &'a Arr1d,
    dwx: Arr2d,
    dwh: Arr2d,
    db: Arr1d,
    cache: Cache,
}

impl<'a> RNN<'a> {
    /// x: (N, D), h_prev:  (N, H)
    /// N: バッチサイズ
    /// D: 入力次元
    /// H: 隠れ次元
    pub fn forward(&mut self, x: Arr2d, h_prev: Arr2d) -> Arr2d {
        let t = h_prev.dot(self.wh) + x.dot(self.wx) + self.b.clone();
        let h_next = t.mapv(f32::tanh);
        self.cache = Cache {
            x,
            h_prev,
            h_next: h_next.clone(),
        };
        h_next
    }
    pub fn backward(&mut self, dh_next: Arr2d) -> (Arr2d, Arr2d) {
        let dt = dh_next * self.cache.h_next.mapv(|y| 1.0 - y * y);
        let dh_prev = dt.dot(&self.wh.t());
        let dx = dt.dot(&self.wx.t());
        self.db = dt.sum_axis(Axis(0));
        self.dwh = self.cache.h_prev.t().dot(&dt);
        self.dwx = self.cache.x.t().dot(&dt);
        (dx, dh_prev)
    }
}

/// (b,c)の外部入力と、(b,h)の内部相互入力を受け、(b,h)を出力する
/// time_sizeはlayerの数
struct TimeRNN<'a> {
    h: Arr2d,
    dh: Arr2d,
    stateful: bool,
    batch_size: usize,
    time_size: usize,
    channel_size: usize,
    hidden_size: usize,
    layers: Vec<_RNN<'a>>,
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
            .map(|_| _RNN {
                _wx: wx,
                _wh: wh,
                _b: b,
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
        for (_t, layer) in self.layers.iter_mut().enumerate() {
            self.h = layer.forward(xs.index_axis(Axis(1), _t).to_owned(), self.h.clone());
            hs.index_axis_mut(Axis(1), _t).assign(&self.h);
        }
        self.layers = vec![];
        for (_t, layer) in self.layers.iter_mut().enumerate() {
            self.h = layer.forward(xs.index_axis(Axis(1), _t).to_owned(), self.h.clone());
            hs.index_axis_mut(Axis(1), _t).assign(&self.h);
        }
        hs
    }
    /// dhs
    pub fn backward(&mut self, dhs: Arr3d) -> Arr3d {
        let mut dxs = Arr3d::zeros((self.batch_size, self.time_size, self.channel_size));
        /// 一番端っこのRNNではhはゼロ?
        let mut dh = Arr2d::zeros((self.batch_size, self.hidden_size));
        for (_t, layer) in self.layers.iter_mut().enumerate().rev() {
            let (_dx, _dh) = layer.backward(dhs.index_axis(Axis(1), _t).to_owned() + dh);
            dh = _dh;
            dxs.index_axis_mut(Axis(1), _t).assign(&_dx);
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
