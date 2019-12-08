use crate::io::*;
use crate::model::*;
use crate::optimizer::{AdaGrad, Optimizer, SGD};
use crate::types::*;
use crate::util::*;
extern crate ndarray;
use ndarray::{s, Array, Array1, Axis, Dim, Dimension, Ix2, RemoveAxis, Slice};

/// データ型をf32から、単語idのusizeにしようとしたら、どうしても
/// Modelを作り替える必要があったので
pub struct Trainer2<M: Model2, T: Optimizer> {
    pub model: M,
    optimizer: T,
    loss_list: Vec<f32>,
}

impl<M: Model2, T: Optimizer> Trainer2<M, T> {
    pub fn new(model: M, optimizer: T) -> Self {
        Self {
            model,
            optimizer,
            loss_list: Vec::new(),
        }
    }
    pub fn fit<D: RemoveAxis>(
        &mut self,
        x: Array<usize, D>, // 単語idの入力
        t: Array1<usize>,   // 分類学習
        max_epoch: usize,
        batch_size: usize,
        // max_grad: Option<f32>,
        eval_interval: Option<usize>,
    ) {
        let (data_len, input_dim) = match x.shape() {
            &[a, _, b] => (a, b),
            &[a, b] => (a, b),
            _ => panic!("KAMO: dimension of x must be 2 or 3 in model.fit!"),
        };
        let max_iters = data_len / batch_size;
        self.loss_list = Vec::<f32>::new();
        let eval_interval = eval_interval.unwrap_or(max_iters);
        0;
        for epoch in 1..=max_epoch {
            let idx = random_index(data_len);
            //　一定回数イテレーションするたびに平均の損失を記録する
            let mut total_loss: f32 = 0.0;
            let mut loss_count: i32 = 0;
            for iters in 1..=max_iters {
                let batch_idx = &idx[(iters - 1) * batch_size..iters * batch_size];
                let batch_data = pickup(&x, Axis(0), batch_idx);
                let batch_target = pickup(&t, Axis(0), batch_idx);
                let loss = self.model.forward(batch_data, batch_target);
                self.model.backward();
                let grads = self.model.grads();
                self.optimizer.update(self.model.params(), grads);
                total_loss += loss; // 1バッチごとに損失を加算していく
                loss_count += 1; // バッチ回数を記録
                if iters % eval_interval == 0 {
                    let avg_loss = total_loss / loss_count as f32;
                    println!(
                        "|epoch {}| iter {}/{} | loss {}",
                        epoch, iters, max_iters, avg_loss
                    );
                    self.loss_list.push(avg_loss);
                    total_loss = 0.0;
                    loss_count = 0;
                }
            }
        }
    }
    pub fn show_loss(&self) {
        println!("{:?}", self.loss_list)
    }
}

pub struct Trainer<M: Model, T: Optimizer> {
    pub model: M,
    optimizer: T,
    loss_list: Vec<f32>,
}

impl<M: Model, T: Optimizer> Trainer<M, T> {
    pub fn new(model: M, optimizer: T) -> Self {
        Self {
            model,
            optimizer,
            loss_list: Vec::new(),
        }
    }
    pub fn fit<D: RemoveAxis>(
        &mut self,
        x: Array<f32, D>,
        t: Arr2d,
        max_epoch: usize,
        batch_size: usize,
        max_grad: Option<f32>,
        eval_interval: Option<usize>,
    ) {
        let (data_len, input_dim) = match x.shape() {
            &[a, _, b] => (a, b),
            &[a, b] => (a, b),
            _ => panic!("KAMO: dimension of x must be 2 or 3 in model.fit!"),
        };
        let target_size = t.shape()[1];
        let max_iters = data_len / batch_size;
        self.loss_list = Vec::<f32>::new();
        let eval_interval = eval_interval.unwrap_or(max_iters);

        for epoch in 1..=max_epoch {
            let idx = random_index(data_len);
            //　一定回数イテレーションするたびに平均の損失を記録する
            let mut total_loss: f32 = 0.0;
            let mut loss_count: i32 = 0;
            for iters in 1..=max_iters {
                let batch_idx = &idx[(iters - 1) * batch_size..iters * batch_size];
                // x[batch_idx, :]的なことをしたいのだが...。簡潔にかけないのか?
                // let batch_data =
                //     Array::from_shape_fn((batch_size, input_dim), |(i, j)| x[[batch_idx[i], j]]);
                let batch_data = pickup(&x, Axis(0), batch_idx);
                // let batch_target =
                //     Array::from_shape_fn((batch_size, target_size), |(i, j)| t[[batch_idx[i], j]]);
                let batch_target = pickup(&t, Axis(0), batch_idx);
                let loss = self.model.forwardx(batch_data, batch_target);
                self.model.backward(batch_size);
                let grads1d = self.model.grads1d();
                let grads2d = self.model.grads2d();
                // self.optimizer
                //     .update1d(self.model.params1d(), self.model.grads1d());
                self.optimizer.update1d(self.model.params1d(), grads1d); // こう書かないと文句言われるんだが、無駄じゃないか?
                self.optimizer.update2d(self.model.params2d(), grads2d);
                total_loss += loss; // 1バッチごとに損失を加算していく
                loss_count += 1; // バッチ回数を記録
                if iters % eval_interval == 0 {
                    let avg_loss = total_loss / loss_count as f32;
                    println!(
                        "|epoch {}| iter {}/{} | loss {}",
                        epoch, iters, max_iters, avg_loss
                    );
                    self.loss_list.push(avg_loss);
                    total_loss = 0.0;
                    loss_count = 0;
                }
            }
        }
    }
    pub fn show_loss(&self) {
        println!("{:?}", self.loss_list)
    }
}
