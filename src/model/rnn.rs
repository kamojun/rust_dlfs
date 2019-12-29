use crate::layers::loss_layer::*;
use crate::layers::time_layers::*;
use crate::optimizer::*;
use crate::params::*;
use crate::types::*;
use crate::util::randarr2d;
use ndarray::{Array2, Axis};

pub trait Rnnlm {
    fn forward(&mut self, x: Array2<usize>, t: Array2<usize>) -> f32;
    fn backward(&mut self);
}
pub trait RnnlmParams {
    fn update(&mut self) {
        unimplemented!();
    }
}
impl RnnlmParams for SimpleRnnlmParams {}

pub struct SimpleRnnlmParams {
    pub embed_w: P1<Arr2d>,
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
impl<'a> Rnnlm for SimpleRnnlm<'a> {
    fn forward(&mut self, x: Array2<usize>, t: Array2<usize>) -> f32 {
        let x = self.embed.forward(x);
        let x = self.rnn.forward(x);
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
