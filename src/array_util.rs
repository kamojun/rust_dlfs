use ndarray::{
    Array, Array1, Array2, Array3, ArrayView1, Axis, Dimension, Ix2, Ix3, RemoveAxis, Slice,
};

pub trait AxisUtil {
    type T;
    type D: Dimension;
    fn remove_axis(self, axis: Axis) -> Array<Self::T, Self::D>;
}

impl<T, D: RemoveAxis> AxisUtil for Array<T, D> {
    type T = T;
    type D = D::Smaller;
    fn remove_axis(self, axis: Axis) -> Array<T, D::Smaller> {
        let mut d = self.shape().to_vec();
        let f = d.remove(axis.0);
        d[axis.0] *= f;
        self.into_shape(d).unwrap().into_dimensionality().unwrap()
    }
}

#[test]
fn axisutil_test() {
    let arr = Array::from_elem((2, 3, 4), 0);
    // putsd!(arr.clone().remove_axis(Axis(0)));
    // putsd!(arr.clone().remove_axis(Axis(1)));
    // putsd!(arr.clone().remove_axis(Axis(2)));
}

use std::convert::TryInto;
pub trait ReshapeUtil<T> {
    fn reshape2<D: Dimension>(self, shape: &[i32]) -> Array<T, D>;
}
impl<T, D: Dimension> ReshapeUtil<T> for Array<T, D> {
    fn reshape2<D2: Dimension>(self, shape: &[i32]) -> Array<T, D2> {
        let shape = match shape.iter().position(|i| *i == -1) {
            Some(x) => {
                let mut v = shape.to_vec();
                v.remove(x);
                let mut v = v
                    .iter()
                    .map(|_v| (*_v).try_into())
                    .collect::<Result<Vec<usize>, _>>()
                    .expect("you can contain only one -1!");
                let otherdim = v.iter().fold(1, |a, b| a * b);
                v.insert(x, self.len() / otherdim);
                v
            }
            None => shape
                .iter()
                .map(|_v| (*_v).try_into())
                .collect::<Result<Vec<usize>, _>>()
                .expect("you can contain only one -1!"),
        };
        self.into_shape(shape)
            .unwrap()
            .into_dimensionality()
            .unwrap()
    }
}
#[test]
fn reshape_test() {
    let arr = Array::from_shape_fn((2, 3, 4), |(i, j, k)| 12 * i + 4 * j + k);
    putsl!(arr, arr.reshape2::<Ix3>(&[4, 3, 2]));
}
