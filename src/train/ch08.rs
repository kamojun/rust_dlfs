use crate::io::*;
use crate::model::rnn::*;
use crate::model::seq2seq::*;
use crate::optimizer::{NewAdam, NewSGD};
use crate::trainer::{RnnlmTrainer, Seq2SeqTrainer};
use crate::types::*;
use crate::util::*;
use ndarray::{array, s, Array2, Axis, Ix2};
use std::collections::HashMap;

use itertools::concat;
/// 例えば
/// 16+75  _91  
/// 52+607 _659
/// 75+22  _97
/// という形式の問題ファイルを受け取り、
/// 問題(_の手前まで), 答え(_以降), 記号一覧、記号->idを返す
use std::collections::HashSet;
fn load_additon_text(filename: &str) -> (Seq, Seq, Vec<char>, HashMap<char, usize>) {
    let raw = read_txt(filename).expect(&format!("couldn't load {}", filename));
    let mut charset = HashSet::new();
    for _r in raw.iter() {
        for c in _r.iter() {
            charset.insert(*c);
        }
    }
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
    const REVERSED: bool = false;
    let (mut x, t, chars, char_to_id) = load_additon_text("./data/date.txt");
    let input_len = x.dim().1;
    // if REVERSED {
    //     // 入力反転
    //     x = x.slice_move(s![..,..;-1]).to_owned();
    // }
    putsl!(x.index_axis(Axis(0), 0), t.index_axis(Axis(0), 0), chars);
    let ((x_train, t_train), (x_test, t_test)) = test_train_split(x, t, (9, 1));
    putsl!(x_train.dim(), t_train.dim(), x_test.dim(), t_test.dim());
    // Array1<char> -> String
    let out = |arr: &Seq, i| {
        arr.index_axis(Axis(0), i)
            .iter()
            .map(|i| chars[*i])
            .collect::<String>()
    };
    println!("{}{}", out(&x_train, 3), out(&t_train, 3));
    println!("{}{}", out(&x_test, 4), out(&t_test, 4));
    let vocab_size = chars.len();
    let encoder_time_size = input_len; // encoderの入力は問題文の全体
    let decoder_time_size = t_train.dim().1 - 1; // decoderは右辺の入力から、一つずらしたものを出力するので、入力長は一つ短い
    let encoder_params = EncoderParams::new(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE);
    let decoder_params =
        SimpleRnnlmParams::new_for_AttentionDecoder(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE);
    let model = Seq2Seq::<AttentionEncoder, AttentionDecoder>::new(
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
    trainer.print_ppl();
    trainer.print_acc();
}

#[test]
fn test_ch08() {
    train_seq2seq();
}
