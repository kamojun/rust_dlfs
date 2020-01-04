use crate::math::*;
use crate::types::*;
use ndarray::{Array, Dimension};
use std::cell::{Ref, RefCell};
use std::rc::Rc;

pub trait Param<T> {
    // fn new(p: T) -> Self;
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
impl<T: Default> P1<T> {
    pub fn new(p: T) -> Self {
        Self {
            _p: RefCell::new(p),
            ..Default::default()
        }
    }
}

impl<T: Default> Param<T> for P1<T> {
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
        self.grads.borrow().iter().fold(sum, |sum, x| sum + x)
    }
}
pub trait Update {
    fn grads_norm_squared(&self) -> f32;
    fn update_lr(&self, lr: f32);
    fn update_clip_lr(&self, clip: f32, lr: f32);
}
impl<D: Dimension> Update for P1<Array<f32, D>> {
    fn grads_norm_squared(&self) -> f32 {
        /// 勾配の和のnormの二乗を返す。
        self.grads_sum().norm2()
    }
    fn update_lr(&self, lr: f32) {
        let g = self.grads_sum() * lr;
        *self._p.borrow_mut() -= &g;
        *self.grads.borrow_mut() = Vec::new();
    }
    fn update_clip_lr(&self, clip: f32, lr: f32) {
        let mut g = self.grads_sum();
        let norm = g.map(|x| x * x).sum();
        g *= (clip / (norm + 1e-6)).min(1.0) * lr;
        *self._p.borrow_mut() -= &g;
        *self.grads.borrow_mut() = Vec::new();
    }
}

/// これにP1を包むと転置した状態でアクセスできる
pub struct P2<T: Default> {
    /// データ本体
    _p: RefCell<T>,
    /// 関連付けるP1
    p1: P1<T>,
}

impl<T: Default> P2<T> {
    pub fn new(p1: T) -> Self {
        Self {
            _p: Default::default(),
            p1: P1::new(p1),
        }
    }
    pub fn t(&self) -> &P1<T> {
        &self.p1
    }
}
impl<'a, A: Default + Copy, D: Dimension> Param<Array<A, D>> for P2<Array<A, D>> {
    fn store(&self, g: Array<A, D>) {
        self.p1.grads.borrow_mut().push(g.t().to_owned());
    }
    fn p(&self) -> Ref<Array<A, D>> {
        *self._p.borrow_mut() = self.p1.p().t().to_owned();
        self._p.borrow()
    }
}
impl<'a, A: Default + Copy, D: Dimension> Param<Array<A, D>> for Rc<P1<Array<A, D>>> {
    fn store(&self, g: Array<A, D>) {
        self.as_ref().store(g);
    }
    fn p(&self) -> Ref<Array<A, D>> {
        self.as_ref().p()
    }
}
