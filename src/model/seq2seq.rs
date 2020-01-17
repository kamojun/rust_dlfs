use crate::io::*;
use crate::layers::loss_layer::*;
use crate::layers::time_layers::*;
use crate::model::rnn::{Rnnlm, RnnlmGen, RnnlmParams, SavableParams, SimpleRnnlmParams};
use crate::params::*;
use crate::types::*;
use crate::util::{expand, randarr, remove_axis, split_arr};
use ndarray::{s, stack, Array2, Array3, Axis};

pub trait Encode {
    fn forward(&mut self, idx: Array2<usize>) -> Arr2d;
    fn backward(&mut self, dh: Arr2d);
    fn reset_state(&mut self);
}
pub trait Decode {
    fn forawrd(&mut self, idx: Array2<usize>, h: Arr2d) -> Arr2d;
    fn backward(&mut self, dscore: Arr2d) -> Arr2d;
    fn generate(&mut self, h: Arr2d, start_id: usize, sample_size: usize) -> Vec<usize>;
    fn reset_state(&mut self);
}
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
        // stateful=falseというのは、毎回のtime方向が独立しているということ
        let lstm = TimeLSTM::new(
            &params.rnn_wx,
            &params.rnn_wh,
            &params.rnn_b,
            time_size,
            false,
        );
        Self {
            embed,
            lstm,
            hs_dim: Default::default(),
        }
    }
}
impl Encode for Encoder<'_> {
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
    fn reset_state(&mut self) {
        self.lstm.reset_state();
    }
}

pub struct Decoder<'a> {
    embed: TimeEmbedding<'a>,
    lstm: TimeLSTM<'a>,
    affine: TimeAffine<'a>,
}
impl<'a> Decoder<'a> {
    fn new(time_size: usize, params: &'a SimpleRnnlmParams) -> Self {
        let embed = TimeEmbedding::new(&params.embed_w);
        // stateful=trueなのは、Encoderからhを受け取るから。
        let lstm = TimeLSTM::new(
            &params.rnn_wx,
            &params.rnn_wh,
            &params.rnn_b,
            time_size,
            true,
        );
        let affine = TimeAffine::new(&params.affine_w, &params.affine_b);
        Self {
            embed,
            lstm,
            affine,
        }
    }
}
impl Decode for Decoder<'_> {
    fn forawrd(&mut self, idx: Array2<usize>, h: Arr2d) -> Arr2d {
        // Encoderから端っこのhを受け取ってset_stateする
        // lstmのstateとしてはcもあるが、これは引き継がない
        self.lstm.set_state(Some(h), None);
        let xs = self.embed.forward(idx);
        let hs = remove_axis(self.lstm.forward(xs));
        self.affine.forward(hs)
    }
    fn backward(&mut self, dscore: Arr2d) -> Arr2d {
        // Encoderに先頭のdhだけ渡す。
        let dout = self.affine.backward(dscore);
        let dout = self.lstm.conv_2d_3d(dout);
        let dout = self.lstm.backward(dout);
        // let dh = dout.index_axis(Axis(1), 0).to_owned();
        self.embed.backward(dout);
        self.lstm.dh.clone()
    }
    fn generate(&mut self, h: Arr2d, start_id: usize, sample_size: usize) -> Vec<usize> {
        let batch_size = h.dim().0; // 基本的にはゼロを想定しているが、別にその必要もないな。(いや、そうなると出力を変える必要が出てきて面倒だ)
        const TIME_SIZE: usize = 1; // 時間方向には一つづづ進める必要がある。
        let mut word_ids = vec![start_id]; // ここに次の単語を追加していく
        let mut sample_id = start_id;
        self.lstm.set_state(Some(h), None);
        for _ in 0..sample_size {
            let x = Array2::from_elem((batch_size, TIME_SIZE), sample_id);
            let mut out = remove_axis(self.embed.forward(x));
            out = self.lstm.forward_piece(out);
            out = self.affine.forward(out); // (batch, word_num)
            let x: f32 = 0.0;
            let max = out.iter().fold(std::f32::NEG_INFINITY, |m, x| m.max(*x));
            sample_id = out
                .iter()
                .position(|x| *x == max)
                .expect("something is definetly wrong in generate");
            word_ids.push(sample_id);
        }
        self.lstm.reset_state(); // 1文作るたびに、reset
        word_ids
    }
    fn reset_state(&mut self) {
        self.lstm.reset_state();
    }
}

pub struct PeekyDecoder<'a> {
    embed: TimeEmbedding<'a>,
    lstm: TimeLSTM<'a>,
    affine: TimeAffine<'a>,
}
impl<'a> PeekyDecoder<'a> {
    fn new(time_size: usize, params: &'a SimpleRnnlmParams) -> Self {
        let embed = TimeEmbedding::new(&params.embed_w);
        // stateful=trueなのは、Encoderからhを受け取るから。
        let lstm = TimeLSTM::new(
            &params.rnn_wx,
            &params.rnn_wh,
            &params.rnn_b,
            time_size,
            true,
        );
        let affine = TimeAffine::new(&params.affine_w, &params.affine_b);
        Self {
            embed,
            lstm,
            affine,
        }
    }
}
impl Decode for PeekyDecoder<'_> {
    fn forawrd(&mut self, idx: Array2<usize>, h: Arr2d) -> Arr2d {
        // h: (batch, hidden)
        self.lstm.set_state(Some(h.clone()), None); // hの一つ目の入力
        let time_size = self.lstm.time_size;
        let hs = expand(h.clone(), Axis(1), time_size); // (batch, time, hidden)にふやす
        let mut xs = self.embed.forward(idx); // (batch, time, wordvec)
        xs = stack![Axis(2), hs.clone(), xs]; // xsにhsをくっつけて(batch, time, wordvec + hidden)
        xs = stack![Axis(2), hs, self.lstm.forward(xs)]; // lstmに通してからまたhsをつけて(batch, time, hidden * 2)
        self.affine.forward(remove_axis(xs))
    }
    fn backward(&mut self, dscore: Arr2d) -> Arr2d {
        // Encoderに先頭のdhだけ渡す。
        let dout = self.affine.backward(dscore); // (batchtime, hidden * 2)
        let hidden_size = dout.dim().1 / 2;
        let (dhs, dxs) = split_arr(dout, Axis(1), hidden_size); // hs, xsの順にforwardしたので
        let dxs = self.lstm.conv_2d_3d(dxs);
        let dhs = self.lstm.conv_2d_3d(dhs);
        let dout = self.lstm.backward(dxs);
        let (dhs1, dembed) = split_arr(dout, Axis(2), hidden_size); // forwardと同じ順番
        self.embed.backward(dembed);
        // dhの勾配を3箇所で合算
        let dh = self.lstm.dh.clone() + dhs.sum_axis(Axis(1)) + dhs1.sum_axis(Axis(1));
        dh
    }
    fn generate(&mut self, h: Arr2d, start_id: usize, sample_size: usize) -> Vec<usize> {
        let batch_size = h.dim().0; // 基本的にはゼロを想定しているが、別にその必要もないな。(いや、そうなると出力を変える必要が出てきて面倒だ)
        const TIME_SIZE: usize = 1; // 時間方向には一つづづ進める必要がある。
        let mut word_ids = vec![start_id]; // ここに次の単語を追加していく
        let mut sample_id = start_id;
        self.lstm.set_state(Some(h.clone()), None); // まずset_state
        for _ in 0..sample_size {
            let x = Array2::from_elem((batch_size, TIME_SIZE), sample_id);
            let mut out = remove_axis(self.embed.forward(x));
            out = stack![Axis(1), h.clone(), out]; // hをくっつける
            out = self.lstm.forward_piece(out);
            out = stack![Axis(1), h.clone(), out]; // hをくっつける
            out = self.affine.forward(out); // (batch, word_num)
            let max = out.iter().fold(std::f32::NEG_INFINITY, |m, x| m.max(*x));
            sample_id = out
                .iter()
                .position(|x| *x == max)
                .expect("something is definetly wrong in generate");
            word_ids.push(sample_id);
        }
        self.lstm.reset_state(); // 1文作るたびに、reset
        word_ids
    }
    fn reset_state(&mut self) {
        self.lstm.reset_state();
    }
}
pub struct Seq2Seq<E: Encode, D: Decode> {
    encoder: E,
    decoder: D,
    loss_layer: SoftMaxWithLoss,
}

impl<'a> Seq2Seq<Encoder<'a>, Decoder<'a>> {
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
}
impl<'a> Seq2Seq<Encoder<'a>, PeekyDecoder<'a>> {
    pub fn new(
        encoder_time_size: usize,
        decoder_time_size: usize,
        encoder_params: &'a EncoderParams,
        decoder_params: &'a SimpleRnnlmParams,
    ) -> Self {
        Self {
            encoder: Encoder::new(encoder_time_size, encoder_params),
            decoder: PeekyDecoder::new(decoder_time_size, decoder_params),
            loss_layer: Default::default(),
        }
    }
}

impl<E: Encode, D: Decode> Seq2Seq<E, D> {
    pub fn forward(&mut self, x: Array2<usize>, t: Array2<usize>) -> f32 {
        let decoder_data_length = (t.dim().1 - 1) as i32;
        let decoder_x = t.slice(s![.., ..decoder_data_length]).to_owned();
        let decoder_t = remove_axis(t.slice(s![.., -decoder_data_length..]).to_owned());
        let h = self.encoder.forward(x);
        let score = self.decoder.forawrd(decoder_x, h);
        self.loss_layer.forward2(score, decoder_t)
    }
    pub fn eval_forward(&mut self, x: Array2<usize>, t: Array2<usize>) -> f32 {
        self.forward(x, t)
    }
    pub fn backward(&mut self) {
        let dout = self.loss_layer.backward();
        let dh = self.decoder.backward(dout);
        let dout = self.encoder.backward(dh);
    }
    pub fn reset_state(&mut self) {
        self.decoder.reset_state();
        self.encoder.reset_state();
    }
    pub fn generate(
        &mut self,
        idx: Array2<usize>,
        start_id: usize,
        sample_size: usize,
    ) -> Vec<usize> {
        let h = self.encoder.forward(idx); // まずencoderに入力して、結果を得る
        let generated = self.decoder.generate(h, start_id, sample_size); // それを元に、出力側の開始記号から初めて出力文作成
        generated
    }
}
