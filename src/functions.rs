use super::types::{Arr1d, Arr2d};
use ndarray::{Array, Array1, Axis, RemoveAxis};
extern crate ndarray_stats;
// use ndarray_stats::SummaryStatisticsExt;

pub fn reverse_one_hot(one_hot_target: &Arr2d) -> Array1<usize> {
    one_hot_target.map_axis(Axis(1), |v| {
        v.iter()
            .position(|x| *x == 1.0)
            .expect("found empty target!")
    })
}

pub fn cross_entropy_error(pred: &Arr2d, one_hot_target: &Arr2d) -> Arr1d {
    // cross_entropy_error_target(pred, reverse_one_hot(one_hot_target)
    -(pred.mapv(|x| (x + 1e-7).ln()) * one_hot_target).sum_axis(Axis(1))
}

pub fn softmax(input: Arr2d) -> Arr2d {
    // input - input.max(Axis(1))
    let input = input.mapv(|x| x.exp());
    let sum = input.sum_axis(Axis(1));
    input / sum.insert_axis(Axis(1))
}
pub fn softmaxd<D: RemoveAxis>(input: Array<f32, D>) -> Array<f32, D> {
    // input - input.max(Axis(1))
    let ndim = input.ndim();
    let input = input.mapv(|x| x.exp());
    let sum = input.sum_axis(Axis(ndim - 1));
    input / sum.insert_axis(Axis(ndim - 1))
}

pub fn cross_entropy_error_target(pred: &Arr2d, target: &Array1<usize>) -> f32 {
    let mut entropy = Array::zeros(target.len());
    for (i, e) in entropy.iter_mut().enumerate() {
        *e = -(pred[[i, target[i]]] + 1e-7).ln()
    }
    entropy
        .mean()
        .expect("error computing cross_entropy_error_target")
}

/// 2つのベクトルの角度のコサインを返す
/// 1~-1の類似度を表す
pub fn cos_similarity(x: Arr1d, y: Arr1d) -> f32 {
    let nx = x.mapv(|a| a.powi(2)).sum().sqrt() + 1e-7;
    let ny = y.mapv(|a| a.powi(2)).sum().sqrt() + 1e-7;
    x.dot(&y) / (nx * ny)
}

/// 単純に内積を取る
pub fn cos(u: &Arr1d, v: &Arr1d) -> f32 {
    u.dot(v)
}

pub fn normalize<D: RemoveAxis>(m: Array<f32, D>) -> Array<f32, D> {
    let d = m.shape().len();
    let nm = m
        .mapv(|a| a * a)
        .sum_axis(Axis(d - 1))
        .mapv(f32::sqrt)
        .insert_axis(Axis(d - 1))
        + 1e-7;
    m / nm
}
