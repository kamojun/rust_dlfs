use crate::io::*;
use crate::layers::loss_layer::*;
use crate::layers::*;
use crate::model::rnn::*;
use crate::optimizer::{AdaGrad, Optimizer, SGD};
use crate::trainer::RnnlmTrainer;
use crate::trainer::Trainer;
use crate::util::*;
use std::collections::HashMap;

pub fn train() {
    const WORDVEC_SIZE: usize = 100;
    const HIDDEN_SIZE: usize = 100;
    const BATCH_SIZE: usize = 10;
    const TIME_SIZE: usize = 5;
    const LR: f32 = 0.1;
    const MAX_EPOCH: usize = 100;
    const CORPUS_SIZE: usize = 1000;

    // corpus ... 文章を、その単語のID列で表したもの
    // word_to_id ... 単語に対し、そのID(一応出現順に割り振る)を対応させた辞書
    // id_to_word ... 逆にしたもの
    let corpus = read_csv_small::<usize>("./data/ptb/corpus.csv", CORPUS_SIZE)
        .expect("error in reading corpus");
    let id_to_word: HashMap<usize, String> = read_csv("./data/ptb/id.csv")
        .expect("error in reading id")
        .into_iter()
        .collect();
    let vocab_size = corpus.iter().fold(usize::min_value(), |m, x| m.max(*x)) + 1; // 単語数
    let xs = corpus[..corpus.len() - 1].to_vec();
    let ts = corpus[1..].to_vec();
    let data_size = xs.len();
    putsd!(CORPUS_SIZE, vocab_size);

    let params = SimpleRnnlmParams::new(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE);
    let model = SimpleRnnlm::new(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE, TIME_SIZE, &params);
    let optimizer = AdaGrad::new(LR);
    let mut trainer = RnnlmTrainer::new(model, &params, optimizer);
    trainer.fit(xs, ts, MAX_EPOCH, BATCH_SIZE, TIME_SIZE, None);
    trainer.print_ppl();
}

#[test]
fn ch05_test() {
    train();
}
