use crate::layers::*;
use crate::model::*;
use crate::types::*;
use crate::util::*;

pub fn train(input_size: i32, hidden_size: i32) {
    let hidden_size = 10;
    let batch_size = 30;
    let max_epoch = 300;

    // let data, target = load_data();
    let data: arr2d;
    let target: arr1d;

    let model = TwoLayerNet::new(2, hidden_size, 3);
    // let optimizer

    // let data_size = data.len();
    // let max_iters = data_size / batch_size;
    // let total_loss: f32 = 0.0;
    // let loss_count: f32 = 0.0;
    // let loss_list = Vec::<f32>::new();

    for epoch in (1..max_epoch) {}
}
