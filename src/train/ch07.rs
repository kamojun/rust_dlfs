use crate::io::*;
use crate::model::rnn::*;
use crate::model::seq2seq::*;
use crate::optimizer::{NewAdam, NewSGD};
use crate::trainer::{RnnlmTrainer, Seq2SeqTrainer};
use crate::types::*;
use crate::util::*;
use ndarray::{array, Array2, Axis, Ix2};
use std::collections::HashMap;

fn gen_text() {
    const SAMPLE_SIZE: usize = 100;
    const TIME_SIZE: usize = 1;
    const MODEL_FILE_NAME: &str = "./data/BetterRnnlm";
    let corpus = read_csv::<usize>("./data/ptb/corpus.csv").expect("error in reading corpus");
    let id_to_word: HashMap<usize, String> = read_csv("./data/ptb/id.csv")
        .expect("error in reading id")
        .into_iter()
        .collect();
    let word_to_id: HashMap<String, usize> =
        id_to_word.iter().map(|(i, w)| (w.clone(), *i)).collect();
    let vocab_size = id_to_word.len(); // 単語数
    putsd!(vocab_size, corpus.len());
    let params = RnnlmLSTMParams::load_from_csv(MODEL_FILE_NAME).expect("error in loading params!");
    // let params = RnnlmLSTMParams::new(vocab_size, 100);
    let mut model = RnnlmLSTM::new(TIME_SIZE, 0.0, &params);
    puts!("model load done!");
    let skip_ids: Vec<_> = ["N", "<unk>", "$"]
        .into_iter()
        .map(|s| word_to_id[&s.to_string()])
        .collect();
    let mut text_id: Vec<_> = "the meaning of life is"
        .split(' ')
        .into_iter()
        .map(|s| word_to_id[&s.to_string()])
        .collect();
    let start_gen_from = text_id.pop().unwrap();
    for i in &text_id {
        model.predict(array![[*i]]);
    }
    let mut ids = model.generate(start_gen_from, skip_ids, SAMPLE_SIZE);
    text_id.append(&mut ids);
    let mut txt: Vec<_> = text_id
        .into_iter()
        .map(|i| id_to_word[&i].clone())
        .collect();
    let txt = replace_item(txt, String::from("<eos>"), String::from("\n")).join(" ");
    println!("{}", txt);
}

use itertools::concat;
/// 16+75  _91  
/// 52+607 _659
/// 75+22  _97
/// という形式の問題ファイルを受け取り、
/// 問題, 答え, 記号一覧を返す
use std::collections::HashSet;
fn load_additon_text(filename: &str) -> (Seq, Seq, Vec<char>, HashMap<char, usize>) {
    let raw = read_txt(filename).expect(&format!("couldn't load {}", filename));
    let mut charset = HashSet::new();
    for _r in raw.iter() {
        for c in _r.iter() {
            charset.insert(*c);
        }
    }
    putsd!(charset);
    let mut charvec: Vec<_> = charset.into_iter().collect();
    charvec.sort();
    let char_to_id: HashMap<_, _> = charvec.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    let xlen = raw[0].iter().position(|c| *c == '_').unwrap();
    let tlen = raw[0].len() - xlen;
    let x = Array2::from_shape_fn((raw.len(), xlen), |(i, j)| char_to_id[&raw[i][j]]);
    let t = Array2::from_shape_fn((raw.len(), tlen), |(i, j)| char_to_id[&raw[i][xlen + j]]);
    (x, t, charvec, char_to_id)
}
fn train_seq2seq() {
    const WORDVEC_SIZE: usize = 16;
    const HIDDEN_SIZE: usize = 128;
    const BATCH_SIZE: usize = 128;
    const MAX_EPOCH: usize = 25;
    const MAX_GRAD: f32 = 5.0;
    const LR: f32 = 0.001;
    const REVERSED: bool = true;
    let (mut x, t, chars, char_to_id) = load_additon_text("./data/addition.txt");
    let input_len = x.dim().1;
    if REVERSED {
        // 入力反転
        x = Array2::from_shape_fn((x.dim()), |(i, j)| x[[i, input_len - j - 1]]);
    }
    putsl!(x.index_axis(Axis(0), 0), t.index_axis(Axis(0), 0), chars);
    let ((x_train, t_train), (x_test, t_test)) = test_train_split(x, t, (9, 1));
    putsl!(x_train.dim(), t_train.dim(), x_test.dim(), t_test.dim());
    let out = |arr: &Seq, i| {
        arr.index_axis(Axis(0), i)
            .iter()
            .map(|i| chars[*i])
            .collect::<String>()
    };
    println!("{}{}", out(&x_train, 3), out(&t_train, 3));
    println!("{}{}", out(&x_test, 4), out(&t_test, 4));
    let vocab_size = chars.len();
    let encoder_time_size = input_len; // encoderの入力は計算式の左辺全体
    let decoder_time_size = t_train.dim().1 - 1; // decoderは右辺の入力から、一つずらしたものを出力するので、入力長は一つ短い
    let encoder_params = EncoderParams::new(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE);
    let decoder_params =
        SimpleRnnlmParams::new_for_PeekyDecoder(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE);
    // let decoder_params =
    // SimpleRnnlmParams::new_for_Decoder(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE);
    // let model = Seq2Seq::<Encoder, Decoder>::new(
    let model = Seq2Seq::<Encoder, PeekyDecoder, Ix2>::new(
        encoder_time_size,
        decoder_time_size,
        &encoder_params,
        &decoder_params,
    );
    let optimizer = NewAdam::new(LR, MAX_GRAD);
    let mut trainer = Seq2SeqTrainer::new(
        model,
        concat(vec![encoder_params.params(), decoder_params.params()]),
        optimizer,
    );
    trainer.fit(
        x_train,
        t_train,
        MAX_EPOCH,
        BATCH_SIZE,
        Some(20),
        Some((x_test, t_test, chars)),
        REVERSED,
    );
    // trainer.eval(&x_test, &t_test, &chars);
    trainer.print_ppl();
    trainer.print_acc();
}

#[test]
pub fn ch07_test() {
    train_seq2seq();
}
