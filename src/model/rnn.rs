use crate::types::*;
use ndarray::{Array2, Axis};
pub trait Rnnlm {
    fn forward(&mut self, x: Array2<usize>, t: Array2<usize>) -> f32;
    fn backward(&mut self);
}

pub struct SimpleRnnlm {
    vocab_size: usize,
    wordvec_size: usize,
    hidden_size: usize,
}
impl Rnnlm for SimpleRnnlm {
    fn forward(&mut self, x: Array2<usize>, t: Array2<usize>) -> f32 {
        unimplemented!();
    }
    fn backward(&mut self) {
        unimplemented!();
    }
}
