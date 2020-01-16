use crate::io::*;
use crate::layers::loss_layer::*;
use crate::layers::*;
use crate::model::rnn::*;
use crate::optimizer::{NewSGD, SGD};
use crate::trainer::RnnlmTrainer;
use crate::trainer::Trainer;
use crate::util::*;
use std::collections::HashMap;

pub fn train() {
    const WORDVEC_SIZE: usize = 100;
    const HIDDEN_SIZE: usize = 100;
    const BATCH_SIZE: usize = 20;
    const TIME_SIZE: usize = 35;
    const LR: f32 = 20.0;
    const MAX_EPOCH: usize = 4;
    const MAX_GRAD: f32 = 0.25;

    // corpus ... 文章を、その単語のID列で表したもの
    // word_to_id ... 単語に対し、そのID(一応出現順に割り振る)を対応させた辞書
    // id_to_word ... 逆にしたもの
    let corpus = read_csv::<usize>("./data/ptb/corpus.csv").expect("error in reading corpus");
    let id_to_word: HashMap<usize, String> = read_csv("./data/ptb/id.csv")
        .expect("error in reading id")
        .into_iter()
        .collect();
    let vocab_size = id_to_word.len(); // 単語数
    putsd!(vocab_size, corpus.len());
    let params = SimpleRnnlmParams::new_for_LSTM(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE);
    let model = SimpleRnnlmLSTM::new(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE, TIME_SIZE, &params);
    let optimizer = NewSGD::new(LR, MAX_GRAD);

    let mut trainer = RnnlmTrainer::new(model, optimizer, params.params());
    trainer.fit(&corpus, MAX_EPOCH, BATCH_SIZE, TIME_SIZE, Some(20), None);
    trainer.print_ppl();

    trainer.model.reset_state();
    // テストデータで評価(TODO)
    let corpus_test =
        read_csv::<usize>("./data/ptb/test_corpus.csv").expect("error in reading corpus");
    let test_ppl = trainer.eval(&corpus_test, BATCH_SIZE, TIME_SIZE);
    putsd!(test_ppl);
}

pub fn train2(name: &str) {
    const WORDVEC_SIZE: usize = 650;
    const HIDDEN_SIZE: usize = 650;
    const BATCH_SIZE: usize = 20;
    const TIME_SIZE: usize = 35;
    const LR: f32 = 20.0;
    const MAX_EPOCH: usize = 4;
    const MAX_GRAD: f32 = 0.25;
    const DROPOUT: f32 = 0.5;
    // const WORDVEC_SIZE: usize = 100;
    // const HIDDEN_SIZE: usize = 100;
    // const BATCH_SIZE: usize = 20;
    // const TIME_SIZE: usize = 35;
    // const LR: f32 = 20.0;
    // const MAX_EPOCH: usize = 1;
    // const MAX_GRAD: f32 = 0.25;
    // const DROPOUT: f32 = 0.5;

    // corpus ... 文章を、その単語のID列で表したもの
    // word_to_id ... 単語に対し、そのID(一応出現順に割り振る)を対応させた辞書
    // id_to_word ... 逆にしたもの
    let corpus = read_csv::<usize>("./data/ptb/corpus.csv").expect("error in reading corpus");
    let corpus_test =
        read_csv::<usize>("./data/ptb/test_corpus.csv").expect("error in reading corpus");
    let corpus_val =
        read_csv::<usize>("./data/ptb/val_corpus.csv").expect("error in reading corpus");
    let vocab_size = corpus.iter().fold(usize::min_value(), |m, x| m.max(*x)) + 1; // 単語数
    putsd!(vocab_size, corpus.len());

    let params = RnnlmLSTMParams::new(vocab_size, WORDVEC_SIZE);
    let model = RnnlmLSTM::new(TIME_SIZE, DROPOUT, &params);
    let optimizer = NewSGD::new(LR, MAX_GRAD);

    let mut trainer = RnnlmTrainer::new(model, optimizer, params.params());
    trainer.fit(
        &corpus,
        MAX_EPOCH,
        BATCH_SIZE,
        TIME_SIZE,
        Some(20),
        Some(&corpus_val),
    );
    trainer.print_ppl();

    trainer.model.reset_state();
    let test_ppl = trainer.eval(&corpus_val, BATCH_SIZE, TIME_SIZE);
    putsd!(test_ppl);
    params.save_as_csv(&format!("./trained/rnnlm_{}", name));
}

#[test]
fn ch06_test() {
    train2("new");
    let params = RnnlmLSTMParams::load_from_csv("./trained/rnnlm_new");
}
