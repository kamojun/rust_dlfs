use crate::io::*;
use crate::layers::loss_layer::*;
use crate::layers::time_layers::*;
use crate::model::rnn::{Rnnlm, RnnlmParams, SavableParams, SimpleRnnlmParams};
use crate::params::*;
use crate::types::*;
use crate::util::{randarr, remove_axis};
use ndarray::{s, Array2, Array3, Axis};

pub struct EncoderParams {
    embed_w: P1<Arr2d>,
    rnn_wx: P1<Arr2d>,
    rnn_wh: P1<Arr2d>,
    rnn_b: P1<Arr1d>,
}
impl EncoderParams {
    pub fn new(vocab_size: usize, wordvec_size: usize, hidden_size: usize) -> Self {
        let embed_w = P1::new(randarr(&[vocab_size, wordvec_size]) / 100.0);
        let mat_init = |m, n| randarr(&[m, n]) / (m as f32).sqrt();
        let rnn_wx = P1::new(mat_init(wordvec_size, 4 * hidden_size));
        let rnn_wh = P1::new(mat_init(hidden_size, 4 * hidden_size));
        let rnn_b = P1::new(Arr1d::zeros((4 * hidden_size,)));
        Self {
            embed_w,
            rnn_wx,
            rnn_wh,
            rnn_b,
        }
    }
}
impl RnnlmParams for EncoderParams {
    fn params(&self) -> Vec<&Update> {
        vec![&self.embed_w, &self.rnn_wx, &self.rnn_wh, &self.rnn_b]
    }
}
impl SavableParams for EncoderParams {
    fn param_names() -> (Vec<&'static str>, Vec<&'static str>) {
        (
            vec!["rnn_b", "affine_b"],
            vec!["embed_w", "rnn_wx", "rnn_wh", "affine_w"],
        )
    }
    fn params_to_save(&self) -> Vec<(&Save, &str)> {
        // ここら辺のコードの重複なくしたいな...
        vec![
            (&self.embed_w, "embed_w"),
            (&self.rnn_wx, "rnn_wx"),
            (&self.rnn_wh, "rnn_wh"),
            (&self.rnn_b, "rnn_b"),
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
        }
    }
}
pub struct Encoder<'a> {
    embed: TimeEmbedding<'a>,
    lstm: TimeLSTM<'a>,
    hs_dim: (usize, usize, usize),
}
impl<'a> Encoder<'a> {
    pub fn new(time_size: usize, params: &'a EncoderParams) -> Self {
        let embed = TimeEmbedding::new(&params.embed_w);
        let lstm = TimeLSTM::new(&params.rnn_wx, &params.rnn_wh, &params.rnn_b, time_size);
        Self {
            embed,
            lstm,
            hs_dim: Default::default(),
        }
    }
    fn forward(&mut self, idx: Array2<usize>) -> Arr2d {
        let xs = self.embed.forward(idx);
        let hs = self.lstm.forward(xs);
        self.hs_dim = hs.dim();
        // 本来上の層に3次元で渡すところだが、Decoderに対して、
        // lstmの端っこの1つのhだけ渡す
        hs.index_axis(Axis(1), hs.dim().1 - 1).to_owned()
    }
    fn backward(&mut self, dh: Arr2d) {
        // forward同様、端っこのdhだけ受け取る
        // それをzeroで初期化したdhsの端っこに埋め込む
        let mut dhs = Array3::<f32>::zeros(self.hs_dim);
        dhs.index_axis_mut(Axis(1), self.hs_dim.1 - 1).assign(&dh);
        let dout = self.lstm.backward(dhs);
        self.embed.backward(dout);
    }
}

pub struct Decoder<'a> {
    embed: TimeEmbedding<'a>,
    lstm: TimeLSTM<'a>,
    affine: TimeAffine<'a>,
}
impl<'a> Decoder<'a> {
    pub fn new(time_size: usize, params: &'a SimpleRnnlmParams) -> Self {
        let embed = TimeEmbedding::new(&params.embed_w);
        let lstm = TimeLSTM::new(&params.rnn_wx, &params.rnn_wh, &params.rnn_b, time_size);
        let affine = TimeAffine::new(&params.affine_w, &params.affine_b);
        Self {
            embed,
            lstm,
            affine,
        }
    }
    fn forawrd(&mut self, idx: Array2<usize>, h: Arr2d) -> Arr2d {
        // Encoderから端っこのhを受け取ってset_stateする
        self.lstm.set_state(h);
        let xs = self.embed.forward(idx);
        let hs = remove_axis(self.lstm.forward(xs));
        self.affine.forward(hs)
    }
    fn backward(&mut self, dscore: Arr2d) -> Arr2d {
        // Encoderに先頭のdhだけ渡す。
        let dout = self.affine.backward(dscore);
        let dout = self.lstm.conv_2d_3d(dout);
        let dout = self.lstm.backward(dout);
        let dh = dout.index_axis(Axis(1), 0).to_owned();
        let embed = self.embed.backward(dout);
        dh
    }
    fn generate(&mut self, h: Arr2d, start_id: usize, sample_size: usize) -> Vec<usize> {
        let batch_size = h.dim().0;
        const TIME_SIZE: usize = 1;
        let mut word_ids = vec![start_id]; // ここに次の単語を追加していく
        self.lstm.set_state(h);
        for _ in 0..sample_size {
            let x = Array2::from_elem((batch_size, TIME_SIZE), start_id);
            let out = remove_axis(self.embed.forward(x));
            let out = self.lstm.forward_piece(out);
            let score = self.affine.forward(out);
            // word_ids.push(argmax(score));
            word_ids.push(1);
        }
        word_ids
    }
}

pub struct Seq2Seq<'a> {
    encoder: Encoder<'a>,
    decoder: Decoder<'a>,
    loss_layer: SoftMaxWithLoss,
}

impl<'a> Seq2Seq<'a> {
    pub fn new(
        encoder_time_size: usize,
        decoder_time_size: usize,
        encoder_params: &'a EncoderParams,
        decoder_params: &'a SimpleRnnlmParams,
    ) -> Self {
        Self {
            encoder: Encoder::new(encoder_time_size, encoder_params),
            decoder: Decoder::new(decoder_time_size, decoder_params),
            loss_layer: Default::default(),
        }
    }
    fn generate(&mut self, idx: Array2<usize>, start_id: usize, sample_size: usize) -> Vec<usize> {
        let h = self.encoder.forward(idx);
        self.decoder.generate(h, start_id, sample_size)
    }
}

impl Rnnlm for Seq2Seq<'_> {
    /// xはencoderへ入力する
    /// tはdecoderにとってのコーパスである
    fn forward(&mut self, x: Array2<usize>, t: Array2<usize>) -> f32 {
        let decoder_data_length = (t.dim().1 - 1) as i32;
        let decoder_x = t.slice(s![.., ..decoder_data_length]).to_owned();
        let decoder_t = remove_axis(t.slice(s![.., -decoder_data_length..]).to_owned());
        let h = self.encoder.forward(x);
        let score = self.decoder.forawrd(decoder_x, h);
        self.loss_layer.forward2(score, decoder_t)
    }
    fn eval_forward(&mut self, x: Array2<usize>, t: Array2<usize>) -> f32 {
        self.forward(x, t)
    }
    fn backward(&mut self) {
        let dout = self.loss_layer.backward();
        let dh = self.decoder.backward(dout);
        let dout = self.encoder.backward(dh);
    }
    fn reset_state(&mut self) {
        self.decoder.lstm.reset_state();
        self.encoder.lstm.reset_state();
    }
}
