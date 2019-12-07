use crate::io::*;
use crate::layers::*;
use crate::model::{Model2, CBOW};
use crate::optimizer::{AdaGrad, Optimizer, SGD};
use crate::trainer::{Trainer, Trainer2};
use crate::util::*;
use std::collections::HashMap;

pub fn train() {
    const WINDOW_SIZE: usize = 5;
    const HIDDEN_SIZE: usize = 100;
    const BATCH_SIZE: usize = 100;
    const MAX_EPOCH: usize = 10;

    let text = "You say goodbye and I say hello.";
    // corpus ... 文章を、その単語のID列で表したもの
    // word_to_id ... 単語に対し、そのID(一応出現順に割り振る)を対応させた辞書
    // id_to_word ... 逆にしたもの
    let corpus = read_csv::<usize>("./data/ptb/corpus.csv").expect("error in reading corpus");
    let id_to_word: HashMap<usize, String> = read_csv("./data/ptb/id.csv")
        .expect("error in reading id")
        .into_iter()
        .collect();
    // let word_to_id: HashMap<String, usize> = id_to_word.iter().map(|(i,w)| (w.clone(), *i)).collect();
    let vocab_size = id_to_word.len(); // 単語数
    putsl!(vocab_size);
    let (contexts, target) = create_contexts_target_arr(&corpus, WINDOW_SIZE);
    putsl!(contexts, target);
    let w_in = randarr2d(100, 200);
    let w_out = randarr2d(100, 200);
    let model = CBOW::new(&[w_in, w_out]);
    let mut optimizer = AdaGrad::new(1.0);
    // // let optimizer = SGD { lr: 1.0 };
    let mut trainer = Trainer2::new(model, optimizer);
    trainer.fit(contexts, target, MAX_EPOCH, BATCH_SIZE, None, None);
    // trainer.show_loss();
    // for (id, word_vec) in trainer.model.word_vecs().outer_iter().enumerate() {
    //     puts!(
    //         format!("{:<8}", id_to_word[&id]),
    //         word_vec.iter().cloned().collect::<Vec<f32>>()
    //     );
    // }
}

#[test]
fn ch04_test() {
    train();
}
