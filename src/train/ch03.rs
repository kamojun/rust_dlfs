use crate::layers::*;
use crate::util::*;

pub fn train() {
    let window_size = 1;
    let hidden_size = 5;
    let batch_size = 3;
    let max_epoch = 1000;

    let text = "You say good bye and I say hello.";
    let (corpus, word_to_id, id_to_word) = preprocess(text);
    let vocab_size = word_to_id.len();
    let (contexts, target) = create_contexts_target(&corpus, window_size);
    let target = convert_one_hot_1(&target, vocab_size);
    // let contexts = convert_one_hot_2(&corpus, vocab_size);
}
