use crate::layers::loss_layer::*;
use crate::layers::time_layers::*;
use crate::math::Norm;
use crate::optimizer::*;
use crate::params::*;
use crate::types::*;
use crate::util::randarr2d;
use ndarray::{Array, Array2, Axis, RemoveAxis};

pub trait Rnnlm {
    fn forward(&mut self, x: Array2<usize>, t: Array2<usize>) -> f32;
    fn backward(&mut self);
    fn reset_state(&mut self) {
        unimplemented!();
    }
}
pub trait RnnlmParams {
    fn update(&self) {
        unimplemented!();
    }
    fn update_clip_lr(&self, clip: f32, lr: f32) {
        unimplemented!();
    }
    fn update_clipgrads(&self, clip: f32, lr: f32) {
        unimplemented!();
    }
}
impl RnnlmParams for SimpleRnnlmParams {
    fn update(&self) {
        self.embed_w.update();
        self.rnn_wx.update();
        self.rnn_wh.update();
        self.rnn_b.update();
        self.affine_w.update();
        self.affine_b.update();
    }
    fn update_clip_lr(&self, clip: f32, lr: f32) {
        self.embed_w.update_clip_lr(clip, lr);
        self.rnn_wx.update_clip_lr(clip, lr);
        self.rnn_wh.update_clip_lr(clip, lr);
        self.rnn_b.update_clip_lr(clip, lr);
        self.affine_w.update_clip_lr(clip, lr);
        self.affine_b.update_clip_lr(clip, lr);
    }
    fn update_clipgrads(&self, clip: f32, lr: f32) {
        let mut norm = 0.0;
        norm += self.embed_w.grads_sum().norm();
        norm += self.rnn_wx.grads_sum().norm();
        norm += self.rnn_wh.grads_sum().norm();
        norm += self.rnn_b.grads_sum().norm();
        norm += self.affine_w.grads_sum().norm();
        norm += self.affine_b.grads_sum().norm();
        norm = norm.sqrt();
        let rate = (clip / (norm + 1e-6)).min(1.0) * lr;
        self.embed_w.update_lr(rate);
        self.rnn_wx.update_lr(rate);
        self.rnn_wh.update_lr(rate);
        self.rnn_b.update_lr(rate);
        self.affine_w.update_lr(rate);
        self.affine_b.update_lr(rate);
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
        let dout = self.rnn.backward(dout);
        self.embed.backward(dout);
    }
    fn reset_state(&mut self) {}
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
