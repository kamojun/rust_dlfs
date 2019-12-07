use crate::layers::loss_layer::*;
use crate::layers::*;
use crate::model::SimpleCBOW;
use crate::optimizer::{AdaGrad, Optimizer, SGD};
use crate::trainer::Trainer;
use crate::util::*;

pub fn train() {
    const WINDOW_SIZE: usize = 1;
    const HIDDEN_SIZE: usize = 5;
    const BATCH_SIZE: usize = 3;
    const MAX_EPOCH: usize = 300;

    let text = "You say goodbye and I say hello.";
    // corpus ... 文章を、その単語のID列で表したもの
    // word_to_id ... 単語に対し、そのID(一応出現順に割り振る)を対応させた辞書
    // id_to_word ... 逆にしたもの
    let (corpus, word_to_id, id_to_word) = preprocess(text);
    let vocab_size = word_to_id.len(); // 単語数
    let (contexts, target) = create_contexts_target(&corpus, WINDOW_SIZE);
    let target = convert_one_hot_1(&target, vocab_size);
    putsl!(target);
    let contexts = convert_one_hot_2(&contexts, vocab_size);
    putsl!(contexts);
    let model = SimpleCBOW::<SoftMaxWithLoss>::new(vocab_size, HIDDEN_SIZE);
    let mut optimizer = AdaGrad::new(1.0);
    // let optimizer = SGD { lr: 1.0 };
    let mut trainer = Trainer::new(model, optimizer);
    trainer.fit(contexts, target, MAX_EPOCH, BATCH_SIZE, None, None);
    trainer.show_loss();
    for (id, word_vec) in trainer.model.word_vecs().outer_iter().enumerate() {
        puts!(
            format!("{:<8}", id_to_word[&id]),
            word_vec.iter().cloned().collect::<Vec<f32>>()
        );
    }
}

#[test]
fn ch03_test() {
    train();
}
