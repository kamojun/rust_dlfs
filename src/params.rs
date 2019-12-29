use crate::types::*;
use std::cell::RefCell;

pub trait Params {
    fn update(&mut self);
}

pub trait Param<T: Default> {
    fn new(p: T) -> Self;
    fn store(&self, g: T);
}

#[derive(Default)]
pub struct P1<T: Default> {
    /// データ本体
    pub p: T,
    /// 学習によって得られた勾配
    grads: RefCell<Vec<T>>,
    /// optimizerが使う情報
    cache: RefCell<Vec<T>>,
}

impl<T: Default> Param<T> for P1<T> {
    fn new(p: T) -> Self {
        Self {
            p,
            ..Default::default()
        }
    }
    fn store(&self, g: T) {
        self.grads.borrow_mut().push(g);
    }
}
