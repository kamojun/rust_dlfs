use crate::layers::loss_layer::*;
use crate::layers::time_layers::*;

pub struct Encoder<'a> {
    vocab_size: usize,
    wordvec_size: usize,
    hidden_size: usize,
    embed: TimeEmbedding<'a>,
    rnn: TimeRNN<'a>,
}

pub struct Decoder<'a> {
    vocab_size: usize,
    wordvec_size: usize,
    hidden_size: usize,
    embed: TimeEmbedding<'a>,
    rnn: TimeRNN<'a>,
    affine: TimeAffine<'a>,
    loss_layer: SoftMaxWithLoss,
}
