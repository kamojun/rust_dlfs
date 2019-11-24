extern crate ndarray;
use ndarray::{Array, Array2, Array3, Axis};
use std::collections::HashMap;

pub fn preprocess(text: &str) -> (Vec<usize>, HashMap<String, usize>, HashMap<usize, String>) {
    let words = text
        .to_lowercase()
        .replace(".", " .")
        .split(' ')
        .map(|s| s.to_string())
        .collect::<Vec<String>>();
    let mut word_id = HashMap::new();
    let mut id_word = HashMap::new();
    let mut id = -1;
    for word in &words {
        word_id.entry(word.clone()).or_insert_with(|| {
            id += 1;
            id_word.insert(id as usize, word.clone());
            id as usize
        });
    }
    let corpus = words.iter().map(|w| word_id[w]).collect::<Vec<usize>>();
    (corpus, word_id, id_word)
}
/// コーパス(id化された単語列(文章))を受け取って、単語列とコンテキスト列を返す
pub fn create_contexts_target(
    corpus: &Vec<usize>,
    window_size: usize,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    let m = window_size * 2 + 1; // 各単語のコンテキストの長さ(その単語を含む)
    let n = match corpus.len().checked_sub(window_size * 2) {
        // コンテキストを持つ単語の長さ
        Some(x) => x,
        None => {
            panic!("window_size too large for the corpus!");
        }
    };
    let contexts_target = (0..n)
        .map(|i| (&corpus[i..(i + m)]).to_vec())
        .collect::<Vec<Vec<usize>>>();
    let target = contexts_target
        .iter()
        .map(|v| v[window_size])
        .collect::<Vec<usize>>();
    let contexts = contexts_target
        .iter()
        .map(|v| ([&v[0..window_size], &v[window_size + 1..m]].concat()))
        .collect::<Vec<Vec<usize>>>();
    (contexts, target)
}
pub fn convert_one_hot_1v(corpus: &Vec<usize>, vocab_size: usize) -> Vec<Vec<i32>> {
    corpus
        .iter()
        .map(|i| {
            let mut v = vec![0; vocab_size];
            v[*i] = 1;
            v
        })
        .collect::<Vec<Vec<i32>>>()
}
pub fn convert_one_hot_2v(corpus: &Vec<Vec<usize>>, vocab_size: usize) -> Vec<Vec<Vec<i32>>> {
    corpus
        .iter()
        .map(|v| convert_one_hot_1v(v, vocab_size))
        .collect::<Vec<Vec<Vec<i32>>>>()
}

pub fn convert_one_hot_1(corpus: &Vec<usize>, vocab_size: usize) -> Array2<i32> {
    let text_len = corpus.len();
    let mut arr = Array2::zeros((text_len, vocab_size));
    for (n, &id) in corpus.iter().enumerate() {
        arr[[n, id]] = 1;
    }
    arr
}
pub fn convert_one_hot_2(corpus: &Vec<Vec<usize>>, vocab_size: usize) -> Array3<i32> {
    let text_len = corpus.len();
    let context_len = match corpus.get(0) {
        Some(c) => c.len(),
        None => panic!("corpus len is zero! therefor context len unknown! by kamo"),
    };
    let mut arr = Array3::zeros((text_len, context_len, vocab_size));

    for (mut v, word) in arr.axis_iter_mut(Axis(0)).zip(corpus.iter()) {
        v.assign(&convert_one_hot_1(word, vocab_size));
    }
    arr
}

use rand::seq::SliceRandom;
use rand::thread_rng;
pub fn random_index(range: usize) -> Vec<usize> {
    let mut vec: Vec<usize> = (0..range).collect();
    vec.shuffle(&mut thread_rng());
    vec
}

// ↓うまくいかない...。
// 配列のジェネリック型として[T; N] (Nはusize)みたいなのができて然るべきだと思うのだが、できない。
// (T, T)みたいなtuple使おうすとると、v[i][j]的なアクセスができない。
// use std::ops::Index;
// pub fn vec_to_array<T, U>(v: Vec<U>, n: usize) -> Array2<T>
// where
//     U: Index<T> + Sized,
// {
//     Array2::from_shape_fn((v.len(), n), |(i, j)| v[i][j]);
// }
