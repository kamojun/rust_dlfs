use crate::io::*;
use crate::layers::loss_layer::*;
use crate::layers::*;
use crate::model::rnn::*;
use crate::optimizer::SGD;
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
    let xs = corpus[..corpus.len() - 1].to_vec();
    let ts = corpus[1..].to_vec();
    putsd!(vocab_size, corpus.len());

    let params = SimpleRnnlmParams::new_for_LSTM(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE);
    let model = SimpleRnnlmLSTM::new(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE, TIME_SIZE, &params);
    let optimizer = SGD { lr: LR };
    let mut trainer = RnnlmTrainer::new(model, &params, optimizer);
    trainer.fit(xs, ts, MAX_EPOCH, BATCH_SIZE, TIME_SIZE, Some(20));
    trainer.print_ppl();

    trainer.model.reset_state();
    // テストデータで評価(TODO)
    let corpus_test =
        read_csv::<usize>("./data/ptb/test_corpus.csv").expect("error in reading corpus");
    let test_ppl = trainer.eval(corpus_test, BATCH_SIZE, TIME_SIZE);
    putsd!(test_ppl);
}

#[test]
fn ch06_test() {
    train();
}
