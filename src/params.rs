use crate::types::*;
use std::cell::{Ref, RefCell};

pub trait Param<T: Default> {
    fn new(p: T) -> Self;
    fn store(&self, g: T);
    fn p(&self) -> Ref<T>;
}

#[derive(Default)]
pub struct P1<T: Default> {
    /// データ本体
    _p: RefCell<T>,
    /// 学習によって得られた勾配
    grads: RefCell<Vec<T>>,
    /// optimizerが使う情報
    cache: RefCell<Vec<T>>,
}

impl<T: Default> Param<T> for P1<T> {
    fn new(p: T) -> Self {
        Self {
            _p: RefCell::new(p),
            ..Default::default()
        }
    }
    fn store(&self, g: T) {
        self.grads.borrow_mut().push(g);
    }
    fn p(&self) -> Ref<T> {
        self._p.borrow()
    }
}
impl P1<Arr1d> {
    pub fn update(&self) {
        let g = self.grads.borrow()[0].clone();
        *self._p.borrow_mut() += &g;
    }
}
impl P1<Arr2d> {
    pub fn update(&self) {
        let g = self.grads.borrow()[0].clone();
        *self._p.borrow_mut() += &g;
    }
}
