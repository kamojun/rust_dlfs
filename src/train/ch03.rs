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

    let text = "You say good bye and I say hello.";
    let (corpus, word_to_id, id_to_word) = preprocess(text);
    let vocab_size = word_to_id.len();
    let (contexts, target) = create_contexts_target(&corpus, WINDOW_SIZE);
    let target = convert_one_hot_1(&target, vocab_size);
    putsl!(target, contexts);
    let contexts = convert_one_hot_2(&contexts, vocab_size);
    let model = SimpleCBOW::<SoftMaxWithLoss>::new(vocab_size, HIDDEN_SIZE);
    let mut optimizer = AdaGrad::new(1.0);
    let mut trainer = Trainer::new(model, optimizer);
    trainer.fit(contexts, target, MAX_EPOCH, BATCH_SIZE, None, None);
    trainer.show_loss();
}

#[test]
fn ch03_test() {
    train();
}
