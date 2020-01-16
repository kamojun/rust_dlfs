use crate::io::*;
use crate::model::rnn::*;
use crate::model::seq2seq::*;
use crate::model::*;
use crate::optimizer::{AdaGrad, NewAdam, NewSGD, Optimizer, SGD};
use crate::params::Update;
use crate::types::*;
use crate::util::*;
use itertools::izip;
// extern crate ndarray;
// use ndarray::iter::AxisChunksIter;
use ndarray::{s, Array, Array1, Array2, Axis, Dim, Dimension, Ix2, RemoveAxis, Slice};

pub struct Seq2SeqTrainer<'a> {
    pub model: Seq2Seq<'a>,
    optimizer: NewAdam,
    params: Vec<&'a Update>,
    ppl_list: Vec<f32>,
    acc_list: Vec<f32>,
    max_iters: usize,
}
impl<'a> Seq2SeqTrainer<'a> {
    pub fn new(model: Seq2Seq<'a>, params: Vec<&'a Update>, optimizer: NewAdam) -> Self {
        Self {
            model,
            params,
            optimizer,
            ppl_list: Vec::new(),
            acc_list: Vec::new(),
            max_iters: 0,
        }
    }
    pub fn print_ppl(&self) {
        putsd!(self.ppl_list);
    }
    pub fn print_acc(&self) {
        putsd!(self.acc_list);
    }
    fn print_progress(
        &mut self,
        epoch: usize,
        iter: usize,
        start_time: std::time::Instant,
        ppl: f32,
    ) {
        let elapsed_time = std::time::Instant::now() - start_time;
        println!(
            "|epoch {}| iter {}/{} | time {}[s] | perplexity {}",
            epoch,
            iter + 1,
            self.max_iters,
            elapsed_time.as_secs(),
            ppl
        );
        self.ppl_list.push(ppl);
    }
    pub fn fit(
        &mut self,
        x_train: Array2<usize>,
        t_train: Array2<usize>,
        max_epoch: usize,
        batch_size: usize,
        eval_interval: Option<usize>,
        eval_problem: Option<(Seq, Seq, Vec<char>)>,
        reversed: bool,
    ) {
        let data_len = x_train.dim().0;
        self.max_iters = data_len / batch_size;
        let start_time = std::time::Instant::now();
        let eval_interval = eval_interval.unwrap_or(self.max_iters);
        let (x_test, t_test, chars) = eval_problem.unwrap_or_default();
        let do_eval = x_test.len() > 0;
        for epoch in 1..=max_epoch {
            let epoch_idx = random_index(data_len);
            let epoch_data = pickup(&x_train, Axis(0), &epoch_idx);
            let epoch_target = pickup(&t_train, Axis(0), &epoch_idx);
            let mut eval_loss = 0.0;
            for (iter, (batch_x, batch_t)) in (izip![
                epoch_data.axis_chunks_iter(Axis(0), batch_size),
                epoch_target.axis_chunks_iter(Axis(0), batch_size)
            ])
            .enumerate()
            {
                eval_loss += self.model.forward(batch_x.to_owned(), batch_t.to_owned());
                self.model.backward();
                self.optimizer.clipgrads(&self.params);
                self.optimizer.update(&self.params);
                if (iter + 1) % eval_interval == 0 {
                    let ppl = (eval_loss / eval_interval as f32).exp();
                    self.print_progress(epoch, iter, start_time, ppl);
                    eval_loss = 0.0;
                }
            }
            if do_eval {
                self.eval(&x_test, &t_test, &chars, reversed);
            }
        }
    }
    pub fn eval(&mut self, x_test: &Seq, t_test: &Seq, chars: &Vec<char>, reversed: bool) {
        let mut correct_count = 0.0;
        let start_id = t_test[[0, 0]];
        let sample_size = t_test.dim().1 - 1; // t_testの2つ目以降を予測する
        for (i, (_x, _t)) in x_test
            .axis_chunks_iter(Axis(0), 1)
            .zip(t_test.axis_chunks_iter(Axis(0), 1))
            .enumerate()
        {
            let guess = self.model.generate(_x.to_owned(), start_id, sample_size);
            self.params.iter().inspect(|p| p.reset_grads());
            let is_correct = _t.iter().zip(guess.iter()).all(|(a, g)| a == g); // answerとguessを比較
            correct_count += if is_correct { 1.0 } else { 0.0 };
            if i < 10 {
                let mut problem: String = _x.iter().map(|i| chars[*i]).collect();
                let guess: String = guess.iter().map(|i| chars[*i]).collect();
                let ans: String = _t.iter().map(|i| chars[*i]).collect();
                if reversed {
                    problem = rev_string(problem)
                }
                println!("{}{}", problem, ans);
                println!("{}{}", problem, guess);
                println!("{}", if guess == ans { "collect!" } else { "wrong!" });
            }
        }
        let acc = correct_count / x_test.dim().0 as f32;
        putsd!(acc);
        self.acc_list.push(acc);
    }
}

pub struct RnnlmTrainer<'a, R: Rnnlm> {
    pub model: R,
    optimizer: NewSGD,
    params: Vec<&'a Update>,
    ppl_list: Vec<f32>,
    acc_list: Vec<f32>,
    max_iters: usize,
}
impl<'a, R: Rnnlm> RnnlmTrainer<'a, R> {
    pub fn new(model: R, optimizer: NewSGD, params: Vec<&'a Update>) -> Self {
        Self {
            model,
            params,
            optimizer,
            ppl_list: Vec::new(),
            acc_list: Vec::new(),
            max_iters: 0,
        }
    }
    pub fn print_ppl(&self) {
        putsd!(self.ppl_list);
    }
    fn print_progress(
        &mut self,
        epoch: usize,
        iter: usize,
        start_time: std::time::Instant,
        ppl: f32,
    ) {
        let elapsed_time = std::time::Instant::now() - start_time;
        println!(
            "|epoch {}| iter {}/{} | time {}[s] | perplexity {}",
            epoch,
            iter + 1,
            self.max_iters,
            elapsed_time.as_secs(),
            ppl
        );
        self.ppl_list.push(ppl);
    }
    fn get_baches(
        corpus: &'a Vec<usize>,
        batch_size: usize,
        time_size: usize,
        time_position: &mut usize,
    ) -> (Array2<usize>, Array2<usize>) {
        let data_size = corpus.len() - 1;
        let batch_time_offset = data_size / batch_size; // 各列でどれだけずらすか
        let time_shift = (batch_time_offset / time_size) * time_size; // time_sizeで割り切れるようにする
        /// ↓各列の先頭
        let time_offsets: Vec<_> = (0..batch_size)
            .map(|i| *time_position + i * batch_time_offset)
            .collect();
        // 各列で、
        let position = |i, j| (time_offsets[i] + j) % data_size;
        let xsa = Array2::from_shape_fn((batch_size, time_shift), |(i, j)| corpus[position(i, j)]);
        let tsa = Array2::from_shape_fn((batch_size, time_shift), |(i, j)| {
            corpus[position(i, j) + 1]
        });
        *time_position += time_shift; // 次の1列目はこの位置に来る
        (xsa, tsa)
    }
    pub fn fit(
        &mut self,
        corpus: &Vec<usize>,
        max_epoch: usize,
        batch_size: usize,
        time_size: usize,
        eval_interval: Option<usize>,
        corpus_val: Option<&Vec<usize>>,
    ) {
        let data_size = corpus.len() - 1;
        //  (batch_size, time_size)型のデータを学習に用いる
        self.max_iters = (data_size / batch_size) / time_size;
        let eval_interval = eval_interval.unwrap_or(self.max_iters);
        let mut time_position = 0;
        let start_time = std::time::Instant::now();
        let mut best_ppl = std::f32::INFINITY;
        for epoch in 1..=max_epoch {
            let mut eval_loss = 0.0;
            let (xsa, tsa) = Self::get_baches(corpus, batch_size, time_size, &mut time_position);
            let x_batches = xsa.axis_chunks_iter(Axis(1), time_size);
            let t_batches = tsa.axis_chunks_iter(Axis(1), time_size);
            for (iter, (batch_x, batch_t)) in x_batches.zip(t_batches).enumerate() {
                eval_loss += self.model.forward(batch_x.to_owned(), batch_t.to_owned());
                self.model.backward();
                self.optimizer.update_clip_lr(&self.params);
                // self.optimizer.update();
                if (iter + 1) % eval_interval == 0 {
                    let ppl = (eval_loss / eval_interval as f32).exp();
                    self.print_progress(epoch, iter, start_time, ppl);
                    eval_loss = 0.0;
                }
            }
            match corpus_val {
                None => {}
                Some(_corpus_val) => {
                    self.model.reset_state(); // train_corpusでの記憶をなくす
                    let ppl = self.eval(_corpus_val, batch_size, time_size);
                    self.model.reset_state(); // evalでの記憶をなくす, TODO: 本当はここでtrain中の記憶を復活させるべきな気もする。
                    if best_ppl > ppl {
                        best_ppl = ppl;
                    } else {
                        self.optimizer.lr /= 4.0;
                    }
                }
            }
        }
    }
    pub fn eval(&mut self, corpus_eval: &Vec<usize>, batch_size: usize, time_size: usize) -> f32 {
        let data_size = corpus_eval.len() - 1;
        //  (batch_size, time_size)型のデータを学習に用いる
        let mut eval_loss = 0.0;
        let max_iters = data_size / (batch_size * time_size);
        let time_shift = max_iters * time_size;
        let mut time_position = 0;
        let (xsa, tsa) = Self::get_baches(corpus_eval, batch_size, time_size, &mut time_position);
        let x_batches = xsa.axis_chunks_iter(Axis(1), time_size);
        let t_batches = tsa.axis_chunks_iter(Axis(1), time_size);
        for (iter, (batch_x, batch_t)) in x_batches.zip(t_batches).enumerate() {
            eval_loss += self
                .model
                .eval_forward(batch_x.to_owned(), batch_t.to_owned());
            if (iter + 1) % 10 == 0 {
                println!("|iter {}/{} |", iter + 1, max_iters);
            }
        }
        (eval_loss / max_iters as f32).exp() // ppl
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
