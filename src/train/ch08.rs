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

fn train_seq2seq_with_dates() {
    const WORDVEC_SIZE: usize = 16;
    const HIDDEN_SIZE: usize = 128;
    const BATCH_SIZE: usize = 128;
    const MAX_EPOCH: usize = 5;
    const MAX_GRAD: f32 = 5.0;
    const LR: f32 = 0.001;
    const REVERSED: bool = false;
    let (mut x, t, chars, char_to_id) = load_underscore_separated_text("./data/date.txt");
    let input_len = x.dim().1;
    if REVERSED {
        // 入力反転
        x = x.slice_move(s![..,..;-1]).to_owned();
    }
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
    // let encoder_params = EncoderParams::new(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE);
    // let decoder_params =
    //     SimpleRnnlmParams::new_for_PeekyDecoder(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE);
    // let model = Seq2Seq::<Encoder, PeekyDecoder>::new(
    //     encoder_time_size,
    //     decoder_time_size,
    //     &encoder_params,
    //     &decoder_params,
    // );
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
    encoder_params
        .save_as_csv("trained/AttentionEncoder")
        .expect("error cannot save encoder params");
    decoder_params
        .save_as_csv("trained/AttentionDecoder")
        .expect("error cannot save decoder params");
}
fn show_attention() {
    let (x, t, chars, char_to_id) = load_underscore_separated_text("./data/date.txt");
    let encoder_time_size = x.dim().1; // encoderの入力は問題文の全体
    let decoder_time_size = t.dim().1 - 1; // decoderは右辺の入力から、一つずらしたものを出力するので、入力長は一つ短い
    let encoder_params = EncoderParams::load_from_csv("trained/AttentionEncoder")
        .expect("cannot load encoder params");
    let decoder_params = SimpleRnnlmParams::load_from_csv("trained/AttentionDecoder")
        .expect("cannot load decoder params");
    let mut model = Seq2Seq::<AttentionEncoder, AttentionDecoder>::new(
        encoder_time_size,
        decoder_time_size,
        &encoder_params,
        &decoder_params,
    );
    let input = "April 25, 1984               ";
    let inpvec: Vec<_> = String::from(input).chars().collect();
    let inparr = Array2::from_shape_fn((1, inpvec.len()), |(i, j)| {
        *char_to_id.get(&inpvec[j]).unwrap()
    });
    println!("attention!");
    let generated = model.generate(inparr, *char_to_id.get(&'_').unwrap(), decoder_time_size);
    println!(
        "{}->{}",
        input,
        generated.iter().map(|i| chars[*i]).collect::<String>()
    );
}

#[test]
fn test_ch08() {
    train_seq2seq_with_dates();
    // show_attention();
}
