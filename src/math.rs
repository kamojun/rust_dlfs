use ndarray::{Array, Dimension};

pub trait Derivative {
    fn dsigmoid(&self) -> Self;
    fn dtanh(&self) -> Self;
}
impl<D: Dimension> Derivative for Array<f32, D> {
    /// self = sigmoid(x)のとき、dself/dx = self*(1-self)となる
    fn dsigmoid(&self) -> Self {
        self * &(1.0 - self)
    }
    /// self = tanh(x)のとき、dself/dx = 1-self**2となる
    fn dtanh(&self) -> Self {
        1.0 - self * self
    }
}
pub trait Norm {
    fn norm2(&self) -> f32;
}
impl<D: Dimension> Norm for Array<f32, D> {
    fn norm2(&self) -> f32 {
        self.map(|x| x * x).sum()
    }
}
