use crate::io::{Load, Save};
use crate::layers::loss_layer::*;
use crate::layers::time_layers::*;
use crate::layers::Dropout;
use crate::math::Norm;
use crate::optimizer::*;
use crate::params::*;
use crate::types::*;
use crate::util::{randarr2d, remove_axis};
use ndarray::{Array, Array2, Axis, Ix2, Ix3, RemoveAxis};
use std::rc::Rc;

pub trait Rnnlm {
    fn forward(&mut self, x: Array2<usize>, t: Array2<usize>) -> f32;
    fn eval_forward(&mut self, x: Array2<usize>, t: Array2<usize>) -> f32 {
        self.forward(x, t)
    }
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
    fn reset_grads(&self) {
        for param in self.params() {
            param.reset_grads();
        }
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
    pub embed_w: P1<Arr2d>,
    pub rnn_wx: P1<Arr2d>,
    pub rnn_wh: P1<Arr2d>,
    pub rnn_b: P1<Arr1d>,
    pub affine_w: P1<Arr2d>,
    pub affine_b: P1<Arr1d>,
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
    pub fn new_for_Decoder(vocab_size: usize, wordvec_size: usize, hidden_size: usize) -> Self {
        Self::new_for_LSTM(vocab_size, wordvec_size, hidden_size)
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
        let rnn = TimeLSTM::new(
            &params.rnn_wx,
            &params.rnn_wh,
            &params.rnn_b,
            time_size,
            true,
        );
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
    time_size: usize,
    embed: TimeEmbedding<'a>,
    dropouts: [Dropout<Ix3>; 3],
    rnn: [TimeLSTM<'a>; 2],
    affine: TimeAffine<'a>,
    loss_layer: SoftMaxWithLoss,
}
impl<'a> RnnlmParams for RnnlmLSTMParams {
    fn params(&self) -> Vec<&Update> {
        vec![
            self.affine_w.t(),
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
    // embed_w: P2<Arr2d>,
    lstm_wx1: P1<Arr2d>,
    lstm_wh1: P1<Arr2d>,
    lstm_b1: P1<Arr1d>,
    lstm_wx2: P1<Arr2d>,
    lstm_wh2: P1<Arr2d>,
    lstm_b2: P1<Arr1d>,
    affine_w: P2<Arr2d>,
    affine_b: P1<Arr1d>,
}
impl<'a> RnnlmLSTMParams {
    /// simplparamsでwordvec_sizeというのがあったが、これはhidden_sizeと共通にさせる
    /// これにより、embed_wとaffine_wを共有させる
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        let h = hidden_size;
        let embed_w = P1::new(randarr2d(vocab_size, h) / 100.0);
        // let mat_init = |m, n| randarr2d(m, n) / (m as f32).sqrt();
        let mat_h4h = || randarr2d(h, 4 * h) / (h as f32).sqrt();
        let lstm_wx1 = P1::new(mat_h4h());
        let lstm_wh1 = P1::new(mat_h4h());
        let lstm_b1 = P1::new(Arr1d::zeros((4 * h,)));
        let lstm_wx2 = P1::new(mat_h4h());
        let lstm_wh2 = P1::new(mat_h4h());
        let lstm_b2 = P1::new(Arr1d::zeros((4 * h,)));
        let affine_w = embed_w.t();
        let affine_b = P1::new(Arr1d::zeros((vocab_size,)));
        Self {
            // embed_w,
            lstm_wx1,
            lstm_wh1,
            lstm_b1,
            lstm_wx2,
            lstm_wh2,
            lstm_b2,
            affine_w, // (hidden_size, vocab_size)
            affine_b,
        }
    }
    pub fn summary(&self) {
        putsd!(self.affine_w.p().dim());
        putsd!(self.affine_w.t().p().dim());
        putsd!(self.lstm_b1.p().dim());
        putsd!(self.lstm_wx1.p().dim());
        putsd!(self.lstm_b2.p().dim());
        putsd!(self.lstm_wx2.p().dim());
    }
}
impl<'a> RnnlmLSTM<'a> {
    pub fn new(time_size: usize, dropout_ratio: f32, params: &'a RnnlmLSTMParams) -> Self {
        let embed = TimeEmbedding::new(params.affine_w.t());
        let vocab_size = params.affine_w.p().dim().1;
        let lstm1 = TimeLSTM::new(
            &params.lstm_wx1,
            &params.lstm_wh1,
            &params.lstm_b1,
            time_size,
            true,
        );
        let lstm2 = TimeLSTM::new(
            &params.lstm_wx2,
            &params.lstm_wh2,
            &params.lstm_b2,
            time_size,
            true,
        );
        let affine = TimeAffine::new(&params.affine_w, &params.affine_b);
        Self {
            vocab_size,
            time_size,
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
    pub fn predict(&mut self, x: Array2<usize>) -> Arr2d {
        let mut x = self.embed.forward(x);
        for i in 0..2 {
            // dropoutなし
            x = self.rnn[i].forward(x);
        }
        let x = remove_axis(x); // x: (timebatch, hidden)
        let x = self.affine.forward(x); // x: (timebatch, vocab) <- さっきのxと各全ての単語との内積を取る
        self.loss_layer.predict(x) // (timebatch, vocab) <- 各単語の確率
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
    fn eval_forward(&mut self, x: Array2<usize>, t: Array2<usize>) -> f32 {
        let mut x = self.embed.forward(x);
        for i in 0..2 {
            // x = self.dropouts[i].train_forward(x);
            x = self.rnn[i].forward(x);
        }
        let x = remove_axis(x);
        let x = self.affine.forward(x);
        let batch_time_size = t.len();
        let t = t.into_shape((batch_time_size,)).unwrap();
        self.loss_layer.forward2(x, t)
    }
    fn reset_state(&mut self) {
        for _r in self.rnn.iter_mut() {
            _r.reset_state();
        }
    }
}

pub trait SavableParams {
    fn param_names() -> (Vec<&'static str>, Vec<&'static str>) {
        (
            vec!["embed_w", "lstm_wx1", "lstm_wx2", "lstm_wx2", "lstm_wh2"],
            vec!["lstm_b1", "lstm_b2", "affine_b"],
        )
    }
    fn params_to_save(&self) -> Vec<(&Save, &str)>;
    fn load_new(params1: Vec<P1<Arr1d>>, params2: Vec<P1<Arr2d>>) -> Self;
}
impl SavableParams for RnnlmLSTMParams {
    fn param_names() -> (Vec<&'static str>, Vec<&'static str>) {
        (
            vec!["rnn_b", "lstm2_b", "affine_b"],
            vec!["embed_w", "lstm1_wx", "lstm1_wh", "lstm2_wx", "lstm2_wh"],
        )
    }
    fn params_to_save(&self) -> Vec<(&Save, &str)> {
        vec![
            (self.affine_w.t(), "embed_w"),
            (&self.lstm_wx1, "lstm1_wx"),
            (&self.lstm_wh1, "lstm1_wh"),
            (&self.lstm_b1, "lstm1_b"),
            (&self.lstm_wx2, "lstm2_wx"),
            (&self.lstm_wh2, "lstm2_wh"),
            (&self.lstm_b2, "lstm2_b"),
            (&self.affine_b, "affine_b"),
        ]
    }
    fn load_new(params1: Vec<P1<Arr1d>>, params2: Vec<P1<Arr2d>>) -> Self {
        // next.unwrap多すぎ
        let mut params1 = params1.into_iter();
        let mut params2 = params2.into_iter();
        let embed_w = params2.next().unwrap();
        Self {
            // embed_w,
            lstm_wx1: params2.next().unwrap(),
            lstm_wh1: params2.next().unwrap(),
            lstm_b1: params1.next().unwrap(),
            lstm_wx2: params2.next().unwrap(),
            lstm_wh2: params2.next().unwrap(),
            lstm_b2: params1.next().unwrap(),
            affine_w: embed_w.t(),
            affine_b: params1.next().unwrap(),
        }
    }
}
impl SavableParams for SimpleRnnlmParams {
    fn param_names() -> (Vec<&'static str>, Vec<&'static str>) {
        (
            vec!["rnn_b", "affine_b"],
            vec!["embed_w", "rnn_wx", "rnn_wh", "affine_w"],
        )
    }
    fn params_to_save(&self) -> Vec<(&Save, &str)> {
        vec![
            (&self.embed_w, "embed_w"),
            (&self.rnn_wx, "rnn_wx"),
            (&self.rnn_wh, "rnn_wh"),
            (&self.rnn_b, "rnn_b"),
            (&self.affine_w, "affine_w"),
            (&self.affine_b, "affine_b"),
        ]
    }
    fn load_new(params1: Vec<P1<Arr1d>>, params2: Vec<P1<Arr2d>>) -> Self {
        // next.unwrap多すぎ
        let mut params1 = params1.into_iter();
        let mut params2 = params2.into_iter();
        Self {
            embed_w: params2.next().unwrap(),
            rnn_wx: params2.next().unwrap(),
            rnn_wh: params2.next().unwrap(),
            rnn_b: params1.next().unwrap(),
            affine_w: params2.next().unwrap(),
            affine_b: params1.next().unwrap(),
        }
    }
}

pub trait RnnlmGen {
    fn generate(
        &mut self,
        start_ids: usize,
        skip_ids: Vec<usize>,
        sample_size: usize,
    ) -> Vec<usize>;
}
use std::collections::HashMap;
impl RnnlmGen for RnnlmLSTM<'_> {
    fn generate(
        &mut self,
        start_id: usize,
        skip_ids: Vec<usize>,
        sample_size: usize,
    ) -> Vec<usize> {
        assert_eq!(self.time_size, 1, "for rnnlmgen, self.time_size must be 1!");
        let mut word_ids = vec![start_id]; // ここに次の単語を追加していく
        let mut rng = thread_rng(); // random number generator
        for _ in 0..sample_size {
            let mut prob = self // 前回のサンプルを元に次の単語を予測
                .predict(Array2::from_elem((1, 1), word_ids[word_ids.len()-1]))
                .into_shape((self.vocab_size,))
                .unwrap();
            for i in &skip_ids {
                // まずい単語の確率をゼロにする
                prob[[*i]] = 0.0;
            }
            let dist = WeightedIndex::new(&prob).unwrap();
            let sample = dist.sample(&mut rng);
            // id_to_word.map(|dic| print!("{} ", dic[&sample]));
            word_ids.push(sample);
        }
        word_ids
    }
}

use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::thread_rng;
#[test]
fn hello_rand() {
    let choices = ['a', 'b', 'c'];
    let weights = [2, 1, 0];
    let dist = WeightedIndex::new(&weights).unwrap();
    let mut rng = thread_rng();
    for _ in 0..100 {
        // 50% chance to print 'a', 25% chance to print 'b', 25% chance to print 'c'
        println!("{}", choices[dist.sample(&mut rng)]);
    }
}
