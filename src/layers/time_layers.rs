use crate::types::*;
use ndarray::{Array1, Axis};

#[derive(Default)]
struct Cache {
    x: Arr2d,
    h_prev: Arr2d,
    h_next: Arr2d,
}

#[derive(Default)]
struct RNN {
    /// (D, H) x(入力)を変換する
    wx: Arr2d,
    /// (H, H) h_prevを変換する
    wh: Arr2d,
    /// (H, ) 定数項
    b: Arr1d,
    dwx: Arr2d,
    dwh: Arr2d,
    db: Arr1d,
    cache: Cache,
}

impl RNN {
    /// x: (N, D), h_prev:  (N, H)
    /// N: バッチサイズ
    /// D: 入力次元
    /// H: 隠れ次元
    pub fn forward(&mut self, x: Arr2d, h_prev: Arr2d) -> Arr2d {
        let t = h_prev.dot(&self.wh) + x.dot(&self.wx) + self.b.clone();
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

struct TimeRNN {
    wx: Arr2d,
    wh: Arr2d,
    b: Arr1d,
    h: Arr2d,
    stateful: bool,
}

impl TimeRNN {
    pub fn forward(&mut self, xs: Arr3d) -> Arr3d {
        let (n, t, d) = xs.dim();
        let (d, h) = self.wx.dim();

        let mut hs = Arr3d::zeros((n, t, h));
        if !self.stateful || false {
            self.h = Arr2d::zeros((n, h));
        }
        for _t in 0..t {}
        hs
    }
}
