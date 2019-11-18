use super::types::{arr1d, arr2d};
use ndarray::{Array, Array1, Axis, Zip};

pub fn reverse_one_hot(one_hot_target: &arr2d) -> Array1<usize> {
    one_hot_target.map_axis(Axis(1), |v| {
        v.iter()
            .position(|x| *x == 1.0)
            .expect("found empty target!")
    })
}

pub fn cross_entropy_error(pred: &arr2d, one_hot_target: &arr2d) -> arr1d {
    // cross_entropy_error_target(pred, reverse_one_hot(one_hot_target)
    -(pred.mapv(|x| (x + 1e-7).ln()) * one_hot_target).sum_axis(Axis(1))
}

pub fn softmax(input: arr2d) -> arr2d {
    // input - input.max(Axis(1))
    let sum = input.sum_axis(Axis(1));
    input.mapv(|x| x.exp()) / sum
}

pub fn cross_entropy_error_target(pred: &arr2d, target: &Array1<usize>) -> arr1d {
    let mut entropy = Array::zeros(target.len());
    for (i, e) in entropy.iter_mut().enumerate() {
        *e = -(pred[[i, target[i]]] + 1e-7).ln()
    }
    entropy
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    #[test]
    fn cross_entropy_error_test() {
        let pred: arr2d = arr2(&[[1.0, 2.0, 3.0]]);
        let target: Array1<usize> = arr1(&[1]);
        let one_hot_target: arr2d = arr2(&[[0.0, 1.0, 0.0]]);
        println!("onehot: {}", cross_entropy_error(&pred, &one_hot_target));
        println!("reverse: {}", reverse_one_hot(&one_hot_target));
        println!("index: {}", cross_entropy_error_target(&pred, &target));
    }
}
