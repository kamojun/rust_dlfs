use crate::types::*;
use ndarray::{Array, Dimension};
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
    // /// optimizerが使う情報
    // cache: RefCell<Vec<T>>,
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

impl<D: Dimension> P1<Array<f32, D>> {
    pub fn grads_sum(&self) -> Array<f32, D> {
        let sum = Array::zeros(self.grads.borrow()[0].dim());
        let g = self.grads.borrow().iter().fold(sum, |sum, x| sum + x);
        *self.grads.borrow_mut() = Vec::new();
        g
    }
    pub fn update(&self) {
        self.update_lr(0.1);
    }
    pub fn update_lr(&self, lr: f32) {
        let g = self.grads_sum() * lr;
        *self._p.borrow_mut() -= &g;
    }
    pub fn update_clip_lr(&self, clip: f32, lr: f32) {
        let mut g = self.grads_sum();
        let norm = g.map(|x| x * x).sum();
        g *= (clip / (norm + 1e-6)).min(1.0) * lr;
        *self._p.borrow_mut() -= &g;
    }
}
