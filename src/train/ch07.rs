use crate::io::*;
use crate::model::rnn::*;
use crate::util::replace_item;
use ndarray::array;
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

fn learn_addtion() {}

#[test]
pub fn ch07_test() {
    gen_text();
}
