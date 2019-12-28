use crate::io::*;
use crate::layers::loss_layer::SoftMaxWithLoss;
use crate::model::*;
use crate::optimizer::{AdaGrad, Adam, Optimizer, SGD};
use crate::trainer::*;
use crate::types::*;
use crate::util::*;
extern crate ndarray;
use ndarray::{Array, Axis, Slice};

pub fn train2() {
    const PRINT_ITER_NUM: usize = 10;
    const HIDDEN_SIZE: usize = 10;
    const BATCH_SIZE: usize = 30;
    const MAX_EPOCH: usize = 300;

    let data = csv_to_array::<f32>("./data/spiral/x.csv").expect("cannot read data csv");
    let target = csv_to_array::<f32>("./data/spiral/t.csv").expect("cannot read target csv");
    let data_len = data.shape()[0];
    assert_eq!(data_len, target.shape()[0]);
    println!("{:?}", target);
    println!("{:?}, {:?}", data.dim(), target.dim());

    let mut model = TwoLayerNet::<SoftMaxWithLoss>::new(2, HIDDEN_SIZE, 3);
    // let mut optimizer = SGD { lr: 1.0 };
    // let mut optimizer = AdaGrad::new(1.0);
    let mut optimizer = Adam::new(0.001, 0.9, 0.999);
    let mut trainer = Trainer::new(model, optimizer);
    trainer.fit(
        data,
        target,
        MAX_EPOCH,
        BATCH_SIZE,
        None,
        Some(PRINT_ITER_NUM),
    );
    trainer.show_loss();
}

pub fn train() {
    const INPUT_DIM: usize = 2;
    const TARGET_SIZE: usize = 3;
    const PRINT_ITER_NUM: usize = 10;
    let hidden_size = 10;
    let batch_size = 30;
    let max_epoch = 300;

    let data = read_csv::<[f32; INPUT_DIM]>("./data/spiral/x.csv").expect("cannot read data csv");
    let data = Array::from_shape_fn((data.len(), INPUT_DIM), |(i, j)| data[i][j]);
    let target =
        read_csv::<[f32; TARGET_SIZE]>("./data/spiral/t.csv").expect("cannot read target csv");
    let target = Array::from_shape_fn((target.len(), TARGET_SIZE), |(i, j)| target[i][j]);
    let data_len = data.shape()[0];
    assert_eq!(data_len, target.shape()[0]);

    let mut model = TwoLayerNet::<SoftMaxWithLoss>::new(2, hidden_size, 3);
    // let mut optimizer = SGD { lr: 1.0 };
    let mut optimizer = AdaGrad::new(1.0);

    let max_iters = data_len / batch_size;
    let mut loss_list = Vec::<f32>::new();

    for epoch in 1..=max_epoch {
        let idx = random_index(data_len);
        //　一定回数イテレーションするたびに平均の損失を記録する
        let mut total_loss: f32 = 0.0;
        let mut loss_count: i32 = 0;
        for iters in 1..=max_iters {
            let batch_idx = &idx[(iters - 1) * batch_size..iters * batch_size];
            // こういう事したいのだが...。
            // data.slice_axis(Axis(0), batch_idx);
            let batch_data =
                Array::from_shape_fn((batch_size, INPUT_DIM), |(i, j)| data[[batch_idx[i], j]]);
            let batch_target = Array::from_shape_fn((batch_size, TARGET_SIZE), |(i, j)| {
                target[[batch_idx[i], j]]
            });
            let loss = model.forward(batch_data, &batch_target);
            model.backward(batch_size);
            // moedl.params1dArr1del.grads1dは片方が&mut self, もう片方が&selfでArr1d
            // 実際はselfの中の別々のもArr2dスしているので、問題はないはずだが、Arr2d
            // コンパイラとしてはそれらのselfの同一フィールドにアクセスしている可能性がある
            // つまり結局同一のフィール度に参照と可変参照の両方でアクセスする可能性があるので
            // エラーとしているのか...。
            // optimizer.update1d(model.params1d(), model.grads1d());
            //
            // let grads1d: Vec<Arr1d> = model.grads1d().into_iter().map(Arr1d::clone).collect();
            // let grads2d: Vec<Arr2d> = model.grads2d().into_iter().map(Arr2d::clone).collect();
            // optimizer.update1d(model.params1d(), grads1d);
            // // optimizer.update2d(model.params2d(), grads2d);
            // optimizer.update1d(model.params1d(), model.grads1d());
            // optimizer.update2d(model.params2d(), model.grads2d());
            total_loss += loss; // 1バッチごとに損失を加算していく
            loss_count += 1; // バッチ回数を記録
            if iters % PRINT_ITER_NUM == 0 {
                let avg_loss = total_loss / loss_count as f32;
                println!(
                    "|epoch {}| iter {}/{} | loss {}",
                    epoch, iters, max_iters, avg_loss
                );
                loss_list.push(avg_loss);
                total_loss = 0.0;
                loss_count = 0;
            }
        }
    }
    println!("hello?");
    putsl!(data);
    putsl!(target);
    putsl!(loss_list);
}

#[test]
fn ch01_train() {
    train2();
}
