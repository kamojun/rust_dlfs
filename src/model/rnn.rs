use crate::layers::loss_layer::*;
use crate::layers::time_layers::*;
use crate::layers::Dropout;
use crate::math::Norm;
use crate::optimizer::*;
use crate::params::*;
use crate::types::*;
use crate::util::randarr2d;
use ndarray::{Array, Array2, Axis, Ix2, Ix3, RemoveAxis};
use std::rc::Rc;

pub trait Rnnlm {
    fn forward(&mut self, x: Array2<usize>, t: Array2<usize>) -> f32;
    fn backward(&mut self);
    fn reset_state(&mut self) {
        unimplemented!();
    }
}
pub trait RnnlmParams {
    fn update_lr(&self, lr: f32) {
        for param in self.params() {
            param.update_lr(lr);
        }
    }
    fn update_clip_lr(&self, clip: f32, lr: f32) {
        for param in self.params() {
            param.update_clip_lr(clip, lr);
        }
    }
    fn update_clipgrads(&self, clip: f32, lr: f32) {
        let norm = self
            .params()
            .iter()
            .map(|x| x.grads_norm_squared())
            .sum::<f32>()
            .sqrt();
        let rate = (clip / (norm + 1e-6)).min(1.0) * lr;
        self.update_lr(rate);
    }
    fn params(&self) -> Vec<&Update>;
}
impl RnnlmParams for SimpleRnnlmParams {
    fn params(&self) -> Vec<&Update> {
        vec![
            &self.embed_w,
            &self.rnn_wx,
            &self.rnn_wh,
            &self.rnn_b,
            &self.affine_w,
            &self.affine_b,
        ]
    }
}

pub struct SimpleRnnlmParams {
    embed_w: P1<Arr2d>,
    rnn_wx: P1<Arr2d>,
    rnn_wh: P1<Arr2d>,
    rnn_b: P1<Arr1d>,
    affine_w: P1<Arr2d>,
    affine_b: P1<Arr1d>,
}
impl SimpleRnnlmParams {
    pub fn new(vocab_size: usize, wordvec_size: usize, hidden_size: usize) -> Self {
        let embed_w = P1::new(randarr2d(vocab_size, wordvec_size) / 100.0);
        let mat_init = |m, n| randarr2d(m, n) / (m as f32).sqrt();
        let rnn_wx = P1::new(mat_init(wordvec_size, hidden_size));
        let rnn_wh = P1::new(mat_init(hidden_size, hidden_size));
        let rnn_b = P1::new(Arr1d::zeros((hidden_size,)));
        let affine_w = P1::new(mat_init(hidden_size, vocab_size));
        let affine_b = P1::new(Arr1d::zeros((vocab_size,)));
        Self {
            embed_w,
            rnn_wx,
            rnn_wh,
            rnn_b,
            affine_w,
            affine_b,
        }
    }
    pub fn new_for_LSTM(vocab_size: usize, wordvec_size: usize, hidden_size: usize) -> Self {
        let embed_w = P1::new(randarr2d(vocab_size, wordvec_size) / 100.0);
        let mat_init = |m, n| randarr2d(m, n) / (m as f32).sqrt();
        let rnn_wx = P1::new(mat_init(wordvec_size, 4 * hidden_size));
        let rnn_wh = P1::new(mat_init(hidden_size, 4 * hidden_size));
        let rnn_b = P1::new(Arr1d::zeros((4 * hidden_size,)));
        let affine_w = P1::new(mat_init(hidden_size, vocab_size));
        let affine_b = P1::new(Arr1d::zeros((vocab_size,)));
        Self {
            embed_w,
            rnn_wx,
            rnn_wh,
            rnn_b,
            affine_w,
            affine_b,
        }
    }
}
pub struct SimpleRnnlm<'a> {
    vocab_size: usize,
    wordvec_size: usize,
    hidden_size: usize,
    embed: TimeEmbedding<'a>,
    rnn: TimeRNN<'a>,
    affine: TimeAffine<'a>,
    loss_layer: SoftMaxWithLoss,
}

/// 先頭の軸を落とす
fn remove_axis<T, D: RemoveAxis>(a: Array<T, D>) -> Array<T, D::Smaller> {
    let mut d = a.shape().to_vec();
    let f = d.remove(1);
    d[0] *= f;
    a.into_shape(d).unwrap().into_dimensionality().unwrap()
}
impl<'a> Rnnlm for SimpleRnnlm<'a> {
    fn forward(&mut self, x: Array2<usize>, t: Array2<usize>) -> f32 {
        let x = self.embed.forward(x);
        let x = remove_axis(self.rnn.forward(x));
        let x = self.affine.forward(x);
        let batch_time_size = t.len();
        let t = t.into_shape((batch_time_size,)).unwrap();
        self.loss_layer.forward2(x, t)
    }
    fn backward(&mut self) {
        let dout = self.loss_layer.backward();
        let dout = self.affine.backward(dout);
        let dout = self.rnn.backward(dout);
        self.embed.backward(dout);
    }
}

impl<'a> SimpleRnnlm<'a> {
    pub fn new(
        vocab_size: usize,
        wordvec_size: usize,
        hidden_size: usize,
        time_size: usize,
        params: &'a SimpleRnnlmParams,
    ) -> Self {
        let embed = TimeEmbedding::new(&params.embed_w);
        let rnn = TimeRNN::new(&params.rnn_wx, &params.rnn_wh, &params.rnn_b, time_size);
        let affine = TimeAffine::new(&params.affine_w, &params.affine_b);
        Self {
            vocab_size,
            wordvec_size,
            hidden_size,
            embed,
            rnn,
            affine,
            loss_layer: Default::default(),
        }
    }
}

pub struct SimpleRnnlmLSTM<'a> {
    vocab_size: usize,
    wordvec_size: usize,
    hidden_size: usize,
    embed: TimeEmbedding<'a>,
    rnn: TimeLSTM<'a>,
    affine: TimeAffine<'a>,
    loss_layer: SoftMaxWithLoss,
}

impl<'a> Rnnlm for SimpleRnnlmLSTM<'a> {
    fn forward(&mut self, x: Array2<usize>, t: Array2<usize>) -> f32 {
        let x = self.embed.forward(x);
        let x = remove_axis(self.rnn.forward(x));
        let x = self.affine.forward(x);
        let batch_time_size = t.len();
        let t = t.into_shape((batch_time_size,)).unwrap();
        self.loss_layer.forward2(x, t)
    }
    fn backward(&mut self) {
        let dout = self.loss_layer.backward();
        let dout = self.affine.backward(dout);
        let dout = self.rnn.conv_2d_3d(dout);
        let dout = self.rnn.backward(dout);
        self.embed.backward(dout);
    }
    fn reset_state(&mut self) {
        self.rnn.reset_state();
    }
}
impl<'a> SimpleRnnlmLSTM<'a> {
    pub fn new(
        vocab_size: usize,
        wordvec_size: usize,
        hidden_size: usize,
        time_size: usize,
        params: &'a SimpleRnnlmParams,
    ) -> Self {
        let embed = TimeEmbedding::new(&params.embed_w);
        let rnn = TimeLSTM::new(&params.rnn_wx, &params.rnn_wh, &params.rnn_b, time_size);
        let affine = TimeAffine::new(&params.affine_w, &params.affine_b);
        Self {
            vocab_size,
            wordvec_size,
            hidden_size,
            embed,
            rnn,
            affine,
            loss_layer: Default::default(),
        }
    }
}

pub struct RnnlmLSTM<'a> {
    vocab_size: usize,
    wordvec_size: usize,
    hidden_size: usize,
    embed: TimeEmbedding<'a>,
    dropouts: [Dropout<Ix3>; 3],
    rnn: [TimeLSTM<'a>; 2],
    affine: TimeAffine<'a>,
    loss_layer: SoftMaxWithLoss,
}
impl<'a> RnnlmParams for RnnlmLSTMParams {
    fn params(&self) -> Vec<&Update> {
        vec![
            self.embed_w.t(),
            &self.lstm_wx1,
            &self.lstm_wh1,
            &self.lstm_b1,
            &self.lstm_wx2,
            &self.lstm_wh2,
            &self.lstm_b2,
            &self.affine_b,
        ]
    }
}

pub struct RnnlmLSTMParams {
    embed_w: P2<Arr2d>,
    lstm_wx1: P1<Arr2d>,
    lstm_wh1: P1<Arr2d>,
    lstm_b1: P1<Arr1d>,
    lstm_wx2: P1<Arr2d>,
    lstm_wh2: P1<Arr2d>,
    lstm_b2: P1<Arr1d>,
    affine_b: P1<Arr1d>,
}
impl<'a> RnnlmLSTMParams {
    /// simplparamsでwordvec_sizeというのがあったが、これはhidden_sizeと共通にさせる
    /// これにより、embed_wとaffine_wを共有させる
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        let h = hidden_size;
        let embed_w = P2::new(randarr2d(vocab_size, h) / 100.0);
        let mat_init = |m, n| randarr2d(m, n) / (m as f32).sqrt();
        let lstm_wx1 = P1::new(mat_init(h, 4 * h));
        let lstm_wh1 = P1::new(mat_init(h, 4 * h));
        let lstm_b1 = P1::new(Arr1d::zeros((4 * h,)));
        let lstm_wx2 = P1::new(mat_init(h, 4 * h));
        let lstm_wh2 = P1::new(mat_init(h, 4 * h));
        let lstm_b2 = P1::new(Arr1d::zeros((4 * h,)));
        let affine_b = P1::new(Arr1d::zeros((vocab_size,)));
        Self {
            embed_w,
            lstm_wx1,
            lstm_wh1,
            lstm_b1,
            lstm_wx2,
            lstm_wh2,
            lstm_b2,
            affine_b,
        }
    }
}
impl<'a> RnnlmLSTM<'a> {
    pub fn new(
        vocab_size: usize,
        wordvec_size: usize,
        hidden_size: usize,
        time_size: usize,
        dropout_ratio: f32,
        params: &'a RnnlmLSTMParams,
    ) -> Self {
        let embed = TimeEmbedding::new(&params.embed_w);
        let lstm1 = TimeLSTM::new(
            &params.lstm_wx1,
            &params.lstm_wh1,
            &params.lstm_b1,
            time_size,
        );
        let lstm2 = TimeLSTM::new(
            &params.lstm_wx2,
            &params.lstm_wh2,
            &params.lstm_b2,
            time_size,
        );
        let affine = TimeAffine::new(params.embed_w.t(), &params.affine_b); // TODO embed_wを転置して、(hidden_size, vocab_size)にしなければならない。
        Self {
            vocab_size,
            wordvec_size,
            hidden_size,
            embed,
            dropouts: [
                Dropout::new(dropout_ratio),
                Dropout::new(dropout_ratio),
                Dropout::new(dropout_ratio),
            ],
            rnn: [lstm1, lstm2],
            affine,
            loss_layer: Default::default(),
        }
    }
}

impl<'a> Rnnlm for RnnlmLSTM<'a> {
    fn forward(&mut self, x: Array2<usize>, t: Array2<usize>) -> f32 {
        let mut x = self.embed.forward(x);
        for i in 0..2 {
            x = self.dropouts[i].train_forward(x);
            x = self.rnn[i].forward(x);
        }
        let x = remove_axis(self.dropouts[2].train_forward(x));
        let x = self.affine.forward(x);
        let batch_time_size = t.len();
        let t = t.into_shape((batch_time_size,)).unwrap();
        self.loss_layer.forward2(x, t)
    }
    fn backward(&mut self) {
        let dout2d = self.loss_layer.backward();
        let dout2d = self.affine.backward(dout2d);
        let mut dout3d = self.rnn[1].conv_2d_3d(dout2d);
        dout3d = self.dropouts[2].backward(dout3d);
        for i in (0..2).rev() {
            dout3d = self.rnn[i].backward(dout3d);
            dout3d = self.dropouts[i].backward(dout3d);
        }
        self.embed.backward(dout3d);
    }
}
