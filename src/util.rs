extern crate ndarray;
use crate::params::Update;
use crate::types::*;
use ndarray::{
    Array, Array1, Array2, Array3, ArrayView1, Axis, Dimension, Ix2, Ix3, RemoveAxis, Slice,
};
use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;
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
pub fn create_contexts_target_arr(
    corpus: &Vec<usize>,
    window_size: usize,
) -> (Array2<usize>, Array1<usize>) {
    let m = window_size * 2 + 1; // 各単語のコンテキストの長さ(その単語を含む)
    let n = match corpus.len().checked_sub(window_size * 2) {
        // コンテキストを持つ単語の長さ
        Some(x) => x,
        None => {
            panic!("window_size too large for the corpus!");
        }
    };
    let contexts_target = Array2::from_shape_fn((n, m), |(i, j)| corpus[i + j]);
    let target = contexts_target.index_axis(Axis(1), window_size).to_owned();
    let context_window: Vec<usize> = (0..m).filter(|i| *i != window_size).collect();
    let contexts = pickup(&contexts_target, Axis(1), &context_window[..]);
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

pub fn convert_one_hot_1(corpus: &Vec<usize>, vocab_size: usize) -> Array2<f32> {
    let text_len = corpus.len();
    let mut arr = Array2::zeros((text_len, vocab_size));
    for (n, &id) in corpus.iter().enumerate() {
        arr[[n, id]] = 1.0;
    }
    arr
}
pub fn convert_one_hot_2(corpus: &Vec<Vec<usize>>, vocab_size: usize) -> Array3<f32> {
    let text_len = corpus.len();
    let context_len = match corpus.get(0) {
        Some(c) => c.len(),
        None => panic!("KAMO: corpus len is zero! therefor context len unknown!"),
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

pub fn randarr2d(m: usize, n: usize) -> Arr2d {
    Array::<f32, _>::random((m, n), StandardNormal)
}
pub fn randarr1d(m: usize) -> Arr1d {
    Array::<f32, _>::random((m,), StandardNormal)
}
pub fn randarr<D: Dimension>(dim: &[usize]) -> Array<f32, D> {
    // Array::<f32, _>::random(dim, StandardNormal)
    Array::<f32, _>::random(dim, StandardNormal)
        .into_dimensionality()
        .unwrap()
    // Default::default()
}
#[test]
fn test_randarr() {
    let a = randarr::<Ix2>(&[1, 2]);
}

extern crate num_traits;
use num_traits::Zero;
/// 元の行列より小さくなる(次元はそのまま)
pub fn pickup<T: Zero + Copy, D: RemoveAxis>(
    x: &Array<T, D>,
    axis: Axis,
    idx: &[usize],
) -> Array<T, D> {
    assert!(
        x.shape()[axis.0] >= idx.len(),
        "KAMO: sorry, haven't implemented yet"
    );
    // 同じ型で、一番外側の次元数だけidx.len()にしたいのだけど、現状では小さくすることしかできない。
    let mut a = x
        .slice_axis(axis, Slice::from(0..idx.len()))
        .mapv(|_| T::zero());
    // .to_owned();  // これならT::zero()は不要
    for (i, j) in idx.iter().enumerate() {
        let mut row = a.index_axis_mut(axis, i); // 移動先
        let base = &x.index_axis(axis, *j); // 移動元
        row.assign(base);
    }
    a
}

/// idxとしてArrayView1を受け取る。
/// pickup関数とほぼ同じ
pub fn pickup1<T: Zero + Copy, D: RemoveAxis>(
    x: &Array<T, D>,
    axis: Axis,
    idx: ArrayView1<usize>,
) -> Array<T, D> {
    assert!(
        x.shape()[axis.0] >= idx.len(),
        "KAMO: sorry, haven't implemented yet"
    );
    let mut a = x
        .slice_axis(axis, Slice::from(0..idx.len()))
        .mapv(|_| T::zero());
    for (i, j) in idx.iter().enumerate() {
        let mut row = a.index_axis_mut(axis, i); // 移動先
        let base = &x.index_axis(axis, *j); // 移動元
        row.assign(base);
    }
    a
}

// もう使ってないのに、overrideしてしまったからか、何度も文句を言われる
// pub trait Len {
//     fn len(&self) -> usize;
// }
// impl Len for ArrayView1<'_, usize> {
//     fn len(&self) -> usize {
//         self.len()
//     }
// }
// impl Len for &[usize] {
//     fn len(&self) -> usize {
//         self.len()
//     }
// }

// IntoIterも traitにしようとしたが...。
// pub trait LenIter<'a> {
//     type IntoIter: Iterator<Item = &'a usize>;
//     fn len(&self) -> usize;
//     fn iter(&'a self) -> Self::IntoIter;
// }
// impl<'a> LenIter<'a> for &[usize] {
//     type IntoIter = std::slice::Iter<'a, usize>;
//     fn len(&self) -> usize {
//         self.len()
//     }
//     fn iter(&'a self) -> Self::IntoIter {
//         self.iter()
//     }
// }
// impl<'a> LenIter<'a> for ArrayView1<'a, usize> {
//     type IntoIter = ndarray::iter::Iter<'a, usize, ndarray::Ix1>;
//     fn len(&self) -> usize {
//         self.len()
//     }
//     fn iter(&'a self) -> Self::IntoIter {
//         self.iter()
//     }
// }

/// Len traitを作ってジェネリックにしてみた
/// idxとして、&[T]は行けるのに、&[T; N](N=1,2,3...)はダメと言われる。
/// idx: &[T]にしてたら&[T; 3] (idx = &[1,2,3]とか)問題なく受け入れるのに、なんでだろう。
/// これ使うと結局stack-over-flowと言われた。なんでだろ。
// pub fn pickup0<'a, T: Zero + Copy, D: RemoveAxis>(
//     x: &Array<T, D>,
//     axis: Axis,
//     idx: impl IntoIterator<Item = &'a usize> + Len,
//     // idx: LenIter<'a>,
// ) -> Array<T, D> {
//     assert!(
//         x.shape()[axis.0] >= idx.len(),
//         "KAMO: sorry, haven't implemented yet"
//     );
//     // 同じ型で、一番外側の次元数だけidx.len()にしたいのだけど、現状では小さくすることしかできない。
//     let mut a = x
//         .slice_axis(axis, Slice::from(0..idx.len()))
//         .mapv(|_| T::zero());
//     // .to_owned();  // これならT::zero()は不要
//     for (i, j) in idx.into_iter().enumerate() {
//         let mut row = a.index_axis_mut(axis, i); // 移動先
//         let base = &x.index_axis(axis, *j); // 移動元
//         row.assign(base);
//     }
//     a
// }
/// 元の次元より大きくもできる
fn pickup2<T: Copy + Zero, D: RemoveAxis>(
    x: &Array<T, D>,
    axis: Axis,
    idx: &[usize],
) -> Array<T, D> {
    // assert!(true);   // idxのrangeとか色々調べた方が良さそうだな。
    let mut s = x.shape().to_vec();
    s[axis.0] = idx.len();
    // 本当は、array作ってからD型にconvertするより、s: shapeの時点でD型にしときたいが、やり方わからず。
    // なお、T: Zeroについては、aが初期化さえできれば良いので、zero出なくて、何か初期値があれば良い、多分
    let mut a = Array::zeros(s).into_dimensionality::<D>().expect("no way!");
    for (mut row, i) in a.axis_iter_mut(axis).zip(idx.iter()) {
        row.assign(&x.index_axis(axis, *i));
    }
    a
}

pub fn pickup_old<T: Copy, D: Dimension>(x: &Array<T, D>, idx: &[usize]) -> Array<T, D> {
    let x = x.view();
    let (data_len, input_dim) = match x.shape() {
        &[a, _, b] => (a, b),
        &[a, b] => (a, b),
        _ => panic!("KAMO: dimension of x must be 2 or 3 in model.fit!"),
    };
    let dim = x.slice_axis(Axis(0), Slice::from(0..idx.len())).dim();
    match x.ndim() {
        2 => x
            .into_dimensionality::<Ix2>()
            .map(|newx| {
                Array::from_shape_fn((idx.len(), input_dim), |(i, j)| newx[[idx[i], j]])
                    .into_shape(dim)
                    .expect("no way!")
            })
            .expect("no way!"),
        3 => {
            let channel_num = x.shape()[1];
            let newx = x.into_dimensionality::<Ix3>().expect("no way!");
            Array::from_shape_fn((idx.len(), channel_num, input_dim), |(i, j, k)| {
                newx[[idx[i], j, k]]
            })
            .into_shape(dim)
            .expect("no way!")
        }
        _ => panic!("dim must be 2 or 3, for now!"),
    }
}

pub fn test_pickup() {
    let arr = Array::from_shape_fn((3, 4), |(i, j)| i * j);
    putsl!(arr);
    // putsl!(pickup0(&arr, Axis(0), &[1, 1, 1]));   // => エラー!!なんじゃそりゃ
    let idx: &[_] = &[1, 2, 3];
    // putsl!(pickup0(&arr, Axis(0), idx)); // これは行ける。型指定したので
    // putsl!(pickup0(&arr, Axis(0), &[1, 1, 1][..])); // 明示的にsliceにしたのでok
    putsl!(pickup(&arr, Axis(0), &[1, 1, 1, 2, 2])); // そもそも引数が&[usize]なので、型推論される
}

pub fn replace_item<T: Eq + Clone>(mut v: Vec<T>, prev: T, new: T) -> Vec<T> {
    for i in v.iter_mut() {
        if *i == prev {
            *i = new.clone()
        }
    }
    v
}

/// 先頭の軸を落とす
pub fn remove_axis<T, D: RemoveAxis>(a: Array<T, D>) -> Array<T, D::Smaller> {
    let mut d = a.shape().to_vec();
    let f = d.remove(1);
    d[0] *= f;
    a.into_shape(d).unwrap().into_dimensionality().unwrap()
}
pub fn insert_axis<T, D: Dimension>(a: Array<T, D>, axis: Axis) -> Array<T, D::Larger> {
    let mut d = a.shape().to_vec();
    d.insert(axis.0, 1);
    // a.into_shape(d)
    // Default::default()
    unimplemented!();
}

pub fn test_train_split<T: Zero + Copy, D: RemoveAxis>(
    x: Array<T, D>,
    t: Array<T, D>,
    ratio: (usize, usize),
) -> ((Array<T, D>, Array<T, D>), (Array<T, D>, Array<T, D>)) {
    let data_len = x.shape()[0];
    assert_eq!(data_len, t.shape()[0], "x and t must have same length!");
    let idx = random_index(data_len);
    let split_here = data_len * ratio.0 / (ratio.0 + ratio.1);
    let x_train = pickup(&x, Axis(0), &idx[..split_here]);
    let t_train = pickup(&t, Axis(0), &idx[..split_here]);
    let x_test = pickup(&x, Axis(0), &idx[split_here..]);
    let t_test = pickup(&t, Axis(0), &idx[split_here..]);
    ((x_train, t_train), (x_test, t_test))
}

pub fn rev_string(s: String) -> String {
    s.chars().rev().collect()
}

pub fn expand<T: Copy + Default, D: Dimension>(
    arr: Array<T, D>,
    axis: Axis,
    num: usize,
) -> Array<T, D::Larger>
where
    D::Larger: RemoveAxis,
{
    let mut s = arr.shape().to_vec();
    let a = axis.0;
    s.insert(a, num);
    let mut arr2 = Array::from_elem(s, T::default())
        .into_dimensionality::<D::Larger>()
        .unwrap();
    for mut sub in arr2.axis_iter_mut(axis) {
        sub.assign(&arr);
    }
    arr2
}

pub fn split_arr<T: Copy + Default, D: Dimension>(
    arr: Array<T, D>,
    axis: Axis,
    left_size: usize,
) -> (Array<T, D>, Array<T, D>) {
    let arr1 = arr.slice_axis(axis, Slice::from(..left_size)).to_owned();
    let arr2 = arr.slice_axis(axis, Slice::from(left_size..)).to_owned();
    (arr1, arr2)
}
