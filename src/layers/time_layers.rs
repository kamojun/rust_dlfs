use crate::functions::*;
use crate::layers::*;
use crate::params::*;
use crate::types::*;
use itertools::izip;
use ndarray::{Array1, Axis, Dimension, RemoveAxis};

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
        let t = h_prev.dot(&*self.wh.p()) + x.dot(&*self.wx.p()) + &*self.b.p();
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
        let dh_prev = dt.dot(&self.wh.p().t());
        let dx = dt.dot(&self.wx.p().t());
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
    time_size: usize,
    channel_size: usize,
    hidden_size: usize,
    layers: Vec<RNN<'a>>,
}

impl<'a> TimeRNN<'a> {
    pub fn new(wx: &'a P1<Arr2d>, wh: &'a P1<Arr2d>, b: &'a P1<Arr1d>, time_size: usize) -> Self {
        let (channel_size, hidden_size) = wx.p().dim();
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
            time_size,
            channel_size,
            hidden_size,
            stateful: true,
        }
    }
    pub fn forward(&mut self, xs: Arr3d) -> Arr3d {
        let batch_size = xs.dim().0;
        let mut hs = Arr3d::zeros((batch_size, self.time_size, self.hidden_size));
        if !self.stateful || self.h.len() == 0 {
            self.h = Arr2d::zeros((batch_size, self.hidden_size));
        }
        for (t, layer) in self.layers.iter_mut().enumerate() {
            self.h = layer.forward(xs.index_axis(Axis(1), t).to_owned(), self.h.clone());
            hs.index_axis_mut(Axis(1), t).assign(&self.h);
        }
        hs
    }
    pub fn backward<D: Dimension>(&mut self, dhs: Array<f32, D>) -> Arr3d {
        let batch_size = dhs.len() / (self.time_size * self.hidden_size);
        let dhs = dhs
            .into_shape((batch_size, self.time_size, self.hidden_size))
            .unwrap();
        let mut dxs = Arr3d::zeros((batch_size, self.time_size, self.channel_size));
        /// 一番端っこのRNNではhはゼロ?
        let mut dh = Arr2d::zeros((batch_size, self.hidden_size));
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
        let channel_num = self.w.p().dim().1;
        let mut out: Arr3d = Array3::zeros((batch_size, time_size, channel_num));
        for (mut _o, _x) in out.outer_iter_mut().zip(idx.outer_iter()) {
            _o.assign(&pickup1(&self.w.p(), Axis(0), _x));
        }
        self.idx = idx;
        out
    }
    pub fn backward(&mut self, dout: Arr3d) {
        let mut dw = Array2::zeros(self.w.p().dim());
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
    x: Arr2d,
}
impl<'a> TimeAffine<'a> {
    pub fn new(w: &'a P1<Arr2d>, b: &'a P1<Arr1d>) -> Self {
        Self {
            w,
            b,
            x: Default::default(),
        }
    }
    pub fn forward(&mut self, x: Arr2d) -> Arr2d {
        self.x = x;
        // dot(&self.x, &self.w.p) + &self.b.p
        self.x.dot(&*self.w.p()) + &*self.b.p()
    }
    pub fn backward(&mut self, dout: Arr2d) -> Arr2d {
        // let (b, t, c_out) = dout.dim(); // batch, time, channel
        // let c_in = self.x.dim().2;
        // let dout = dout.into_shape((b * t, c_out)).unwrap();
        // let rx = self
        //     .x
        //     .clone()
        //     .into_shape((b * t, c_in))
        //     .expect("self.x and dout must have same dimention!");
        self.b.store(dout.sum_axis(Axis(0)));
        self.w.store(self.x.t().dot(&dout));
        dout.dot(&self.w.p().t())
    }
}

/// SoftMaxWithLossと全く同じになりそうなので、とりあえずはあとで
pub struct TimeSoftmaxWithLoss {
    pred: Arr2d,
    target: Array1<usize>,
}
impl TimeSoftmaxWithLoss {
    fn forward(&mut self, xs: Arr2d, ts: Array1<usize>) -> f32 {
        self.pred = softmax(xs);
        self.target = ts;
        // 本ではもう少しごちゃごちゃやっているmaskがなんとか
        cross_entropy_error_target(&self.pred, &self.target)
    }
    fn backward(&mut self) -> Arr2d {
        unimplemented!();
    }
}

#[derive(Default, Clone)]
struct CacheLSTM {
    x: Arr2d,
    h_prev: Arr2d,
    c_prev: Arr2d,
    i: Arr2d,
    f: Arr2d,
    g: Arr2d,
    o: Arr2d,
    c_next: Arr2d,
}
struct LSTM<'a> {
    /// (channel, 4*hidden)
    wx: &'a P1<Arr2d>,
    /// (hidden, 4*hidden)
    wh: &'a P1<Arr2d>,
    /// (4*hidden, )
    b: &'a P1<Arr1d>,
    cache: CacheLSTM,
}
trait Derivative {
    fn dsigmoid(&self) -> Self;
    fn dtanh(&self) -> Self;
}
impl<D: Dimension> Derivative for Array<f32, D> {
    /// self = sigmoid(x)のとき、dself/dx = self*(1-self)となる
    fn dsigmoid(&self) -> Self {
        self * &(1.0 - self)
    }
    /// self = tanh(x)のとき、dself/dx = 1-self**2となる
    fn dtanh(&self) -> Self {
        1.0 - self * self
    }
}

impl<'a> LSTM<'a> {
    /// x: (batch, channel), h_prev: (batch, hidden), c_prev: (batch, hidden)
    pub fn forward(&mut self, x: Arr2d, h_prev: Arr2d, c_prev: Arr2d) -> (Arr2d, Arr2d) {
        let (batch_size, hidden_size) = h_prev.dim();
        // (batch, 4*hidden)
        let a = x.dot(&*self.wx.p()) + h_prev.dot(&*self.wh.p()) + &*self.b.p();
        let mut chunks = a.axis_chunks_iter(Axis(1), hidden_size);
        let mut yielder = |f: fn(f32) -> f32| chunks.next().unwrap().mapv(f);
        // それぞれ (batch, hidden)
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let f = yielder(sigmoid); // forget
        let g = yielder(f32::tanh); // gain new info
        let i = yielder(sigmoid); // input
        let o = yielder(sigmoid); // output

        let c_next = &f * &c_prev + &g * &i; // c_prevを割合fで忘却し、gを割合iで追加する
        let h_next = &o * &c_next.mapv(f32::tanh); // c_nextをtanhして、割合oで出力する

        self.cache = CacheLSTM {
            x,
            h_prev,
            c_prev,
            i,
            f,
            g,
            o,
            c_next: c_next.clone(),
        };
        (h_next, c_next)
    }
    /// 返り値は(dx, dh_prev, dc_prev)
    pub fn backward(&mut self, dh_next: Arr2d, dc_next: Arr2d) -> (Arr2d, Arr2d, Arr2d) {
        let cache = self.cache.clone();
        let c_next_tanh = cache.c_next.mapv(f32::tanh); // 本来forwardで行った計算だが、cacheしてないので、復活させる
        let ds = dc_next + &dh_next * &cache.o * c_next_tanh.dtanh();
        let dc_prev = &ds * &cache.f;
        // dAを準備
        let (batch_size, hidden_size) = dh_next.dim();
        let mut dA = Arr2d::zeros((batch_size, hidden_size * 4));
        let mut chunk = dA.axis_chunks_iter_mut(Axis(1), hidden_size);
        let mut assign = |d| chunk.next().unwrap().assign(&d);
        // di, df, do, dgの順に格納していく
        assign(&ds * &cache.g * cache.i.dsigmoid());
        assign(&ds * &cache.c_prev * cache.f.dsigmoid());
        assign(&dh_next * &c_next_tanh * cache.o.dsigmoid());
        assign(&ds * &cache.i * cache.o.dtanh());
        // dAから、wh, wx, bの勾配を計算し格納
        self.wh.store(cache.h_prev.t().dot(&dA));
        self.wx.store(cache.x.t().dot(&dA));
        self.b.store(dA.sum_axis(Axis(0))); // <= batch方向に潰す

        let dx = dA.dot(&self.wx.p().t());
        let dh_prev = dA.dot(&self.wh.p().t());
        (dx, dh_prev, dc_prev)
    }
}

pub struct TimeLSTM<'a> {
    h: (Arr2d, Arr2d),
    c: (Arr2d, Arr2d),
    stateful: bool,
    time_size: usize,
    channel_size: usize,
    hidden_size: usize,
    layers: Vec<LSTM<'a>>,
}

impl<'a> TimeLSTM<'a> {
    pub fn new(wx: &'a P1<Arr2d>, wh: &'a P1<Arr2d>, b: &'a P1<Arr1d>, time_size: usize) -> Self {
        let (channel_size, mut hidden_size) = wx.p().dim();
        hidden_size /= 4; // wxは(channel, 4*hidden)
        let layers: Vec<_> = (0..time_size)
            .map(|_| LSTM {
                wx: wx,
                wh: wh,
                b: b,
                cache: Default::default(),
            })
            .collect();
        Self {
            layers,
            h: Default::default(),
            c: Default::default(),
            time_size,
            channel_size,
            hidden_size,
            stateful: false,
        }
    }
    pub fn forward(&mut self, xs: Arr3d) -> Arr3d {
        let batch_size = xs.dim().0;
        let mut hs = Arr3d::zeros((batch_size, self.time_size, self.hidden_size));
        if !self.stateful || self.h.0.len() == 0 {
            self.h.0 = Arr2d::zeros((batch_size, self.hidden_size));
            self.c.0 = self.h.0.clone();
        }
        for (layer, xst, mut hst) in izip!(
            self.layers.iter_mut(),
            xs.axis_iter(Axis(1)),
            hs.axis_iter_mut(Axis(1))
        ) {
            let (_h, _c) = layer.forward(xst.to_owned(), self.h.0.clone(), self.c.0.clone());
            hst.assign(&_h);
            self.h.0 = _h;
            self.c.0 = _c;
        }
        hs
    }
    pub fn backward<D: Dimension>(&mut self, dhs: Array<f32, D>) -> Arr3d {
        let batch_size = dhs.len() / (self.time_size * self.hidden_size);
        let dhs = dhs
            .into_shape((batch_size, self.time_size, self.hidden_size))
            .unwrap();
        let mut dxs = Arr3d::zeros((batch_size, self.time_size, self.channel_size));
        /// このdh, dcはLSTMの相互入出力に関する勾配
        /// hに関しては、上流から来るdhsもあるが、全く別物
        let mut dh = Arr2d::zeros((batch_size, self.hidden_size));
        let mut dc = dh.clone();
        for (layer, mut dxt, dht) in (izip!(
            self.layers.iter_mut(),
            dxs.axis_iter_mut(Axis(1)),
            dhs.axis_iter(Axis(1))
        ))
        .rev()
        {
            /// 上流からのdhtと、横からのdh(先頭ではゼロ)を加算する
            let (_dx, _dh, _dc) = layer.backward(dht.to_owned() + dh, dc);
            dh = _dh;
            dc = _dc;
            dxt.assign(&_dx);
        }
        dxs
    }
    pub fn reset_state(&mut self) {
        self.h = Default::default();
        self.c = Default::default();
    }
}
