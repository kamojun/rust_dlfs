use crate::types::*;
use std::cell::RefCell;

pub trait Params {
    fn update(&mut self);
}

pub trait Param {
    fn update(&mut self);
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

impl<T: Default> P1<T> {
    pub fn new(p: T) -> Self {
        Self {
            p,
            ..Default::default()
        }
    }
    pub fn store(&self, g: T) {
        self.grads.borrow_mut().push(g);
    }
}

impl Param for P1<Arr2d> {
    fn update(&mut self) {
        self.p += &self.grads.borrow()[0];
    }
}
