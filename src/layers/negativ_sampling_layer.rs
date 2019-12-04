extern crate ndarray;
use super::SigmodWithLoss;
use crate::functions::*;
use crate::types::{Arr1d, Arr2d};
use crate::util::*;
use ndarray::{Array, Array1, Array2, Array3, Axis};

pub struct EmbeddingDot {}
impl EmbeddingDot {
    fn forward(&mut self, input: Arr2d, h: Array1<usize>) -> Arr2d {
        unimplemented!();
    }
}

pub struct NegativeSamplingLoss {
    sample_size: usize,
    // sampler,
    loss_layers: [SigmodWithLoss; 6],
    embed_loss_layers: [EmbeddingDot; 6],
}
impl NegativeSamplingLoss {
    fn forward(&mut self, input: Arr2d, target: Array1<usize>) -> f32 {
        let batch_size = target.shape()[0];
        let negative_sample = Array2::<usize>::ones((1, 1));
        let ci = input.clone();
        let loss = self.positive_forward(ci, target);

        // let negative_label = Array1::zeros((batch_size,));
        for i in 0..self.sample_size {
            let negative_target = negative_sample.index_axis(Axis(1), i).to_owned();
            // let score = self.embed_loss_layers[i + 1].forward(input, negative_target);
        }

        0.0
    }
    fn positive_forward(&mut self, input: Arr2d, target: Array1<usize>) -> f32 {
        let batch_size = target.shape()[0];
        let score = self.embed_loss_layers[0].forward(input, target);
        let correct_label = Array1::<usize>::ones((batch_size,));
        // self.loss_layers[0].forward2(score, correct_label)
        0.0
    }
}
