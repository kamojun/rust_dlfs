use crate::params::*;
use crate::types::*;
use ndarray::{Array1, Axis};

#[derive(Default)]
struct Cache {
    x: Arr2d,
    h_prev: Arr2d,
    h_next: Arr2d,
}

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

struct TimeRNN<'a> {
    wx: P1<Arr2d>,
    wh: P1<Arr2d>,
    b: P1<Arr1d>,
    h: Arr2d,
    stateful: bool,
    batch_size: usize,
    time_size: usize,
    channel_size: usize,
    hidden_size: usize,
    layers: Vec<_RNN<'a>>,
}

impl<'a> TimeRNN<'a> {
    // pub fn new(wx: Arr2d, wh: Arr2d, b: Arr1d, batch_size: usize, time_size: usize) -> Self {
    //     let (channel_size, hidden_size) = wx.dim();
    //     let wx = P1::new(wx);
    //     let wh = P1::new(wh);
    //     let b = P1::new(b);
    //     let layers = Vec::new();
    //     for _ in 0..time_size {
    //         layers.push(_RNN {
    //             _wx: &wx,
    //             _wh: &wh,
    //             _b: &b,
    //             cache: Default::default(),
    //         });
    //     }
    //     Self {
    //         wx,
    //         wh,
    //         b,
    //         layers,
    //         h: Default::default(),
    //         batch_size,
    //         time_size,
    //         channel_size,
    //         hidden_size,
    //         stateful: true,
    //     }
    // }
    pub fn forward(&mut self, xs: Arr3d) -> Arr3d {
        let mut hs = Arr3d::zeros((self.batch_size, self.time_size, self.hidden_size));
        if !self.stateful || false {
            self.h = Arr2d::zeros((self.batch_size, self.hidden_size));
        }
        for (_t, layer) in self.layers.iter_mut().enumerate() {
            self.h = layer.forward(xs.index_axis(Axis(1), _t).to_owned(), self.h.clone());
            hs.index_axis_mut(Axis(1), _t).assign(&self.h);
        }
        // self.layers = vec![];
        // for _t in 0..self.time_size {
        //     let mut layer = _RNN {
        //         _wx: &self.wx,
        //         _wh: &self.wh,
        //         _b: &self.b,
        //         cache: Default::default(),
        //     };
        //     self.h = layer.forward(xs.index_axis(Axis(1), _t).to_owned(), self.h.clone());
        //     hs.index_axis_mut(Axis(1), _t).assign(&self.h);
        //     // self.layers.push(layer);
        // }
        hs
    }

    pub fn backward(&mut self, dhs: Arr3d) {
        // let
    }
}
