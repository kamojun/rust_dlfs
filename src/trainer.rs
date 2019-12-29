use crate::io::*;
use crate::model::rnn::*;
use crate::model::*;
use crate::optimizer::{AdaGrad, Optimizer, SGD};
use crate::types::*;
use crate::util::*;
extern crate ndarray;
use ndarray::{s, Array, Array1, Array2, Axis, Dim, Dimension, Ix2, RemoveAxis, Slice};

pub struct RnnlmTrainer<'a, R: Rnnlm, P: RnnlmParams, O: Optimizer> {
    model: R,
    params: &'a P,
    optimizer: O,
    time_idx: usize,
    ppl_list: Vec<f32>,
}
impl<'a, R: Rnnlm, P: RnnlmParams, O: Optimizer> RnnlmTrainer<'a, R, P, O> {
    pub fn new(model: R, params: &'a P, optimizer: O) -> Self {
        Self {
            model,
            params,
            optimizer,
            time_idx: 0,
            ppl_list: Vec::new(),
        }
    }
    pub fn get_batch(&self, x: Arr2d, t: Arr2d, batch_size: usize, time_size: usize) {
        let data_size = x.shape()[0];
        let sample_num = data_size / time_size;
    }
    pub fn fit(
        &mut self,
        xs: Vec<usize>,
        ts: Vec<usize>,
        max_epoch: usize,
        batch_size: usize,
        time_size: usize,
        eval_interval: Option<usize>,
    ) {
        let data_size = xs.len();
        //  (batch_size, time_size)型のデータを学習に用いる
        let mut eval_loss = 0.0;

        let time_shift = data_size / batch_size;
        let max_iters = time_shift / time_size;
        let time_shift = max_iters * time_size;
        let eval_interval = eval_interval.unwrap_or(max_iters);

        let xsa = Array2::from_shape_fn((batch_size, time_shift), |(i, j)| xs[i * time_shift + j]);
        let tsa = Array2::from_shape_fn((batch_size, time_shift), |(i, j)| ts[i * time_shift + j]);

        let start_time = std::time::Instant::now();
        // 単純に同じデータで学習を繰り返す。
        // ランダム性はない
        for epoch in 1..=max_epoch {
            let x_batches = xsa.axis_chunks_iter(Axis(1), time_size);
            let t_batches = tsa.axis_chunks_iter(Axis(1), time_size);
            for (iter, (batch_x, batch_t)) in x_batches.zip(t_batches).enumerate() {
                eval_loss += self.model.forward(batch_x.to_owned(), batch_t.to_owned());
                self.model.backward();
                self.params.update();
                if iter % eval_interval == 0 {
                    let ppl = (eval_loss / eval_interval as f32).exp();
                    let elapsed_time = std::time::Instant::now() - start_time;
                    println!(
                        "|epoch {}| iter {}/{} | time {}[s] | perplexity {}",
                        epoch,
                        iter,
                        max_iters,
                        elapsed_time.as_secs(),
                        ppl
                    );
                    self.ppl_list.push(ppl);
                    // lossについてはeval_intervalごとの評価を行う。
                    eval_loss = 0.0;
                }
            }
        }
    }
}

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
                self.optimizer.update2d(self.model.params(), grads);
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
        println!(
            "{:?}",
            // self.loss_list.iter().step_by(100).collect::<Vec<_>>()
            self.loss_list
        )
    }

    pub fn save_params(&mut self) {
        for p in self.model.params() {}
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
