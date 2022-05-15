use crate::{
    graph::{Gamma, InputParam, OutputPort, Rvsdg, Theta},
    ir::Const,
    utils::HashMap,
    values::{Cell, Ptr},
};
use std::{cell::RefCell, mem::take, thread};

thread_local! {
    // FIXME: https://github.com/rust-lang/rust-clippy/issues/8493
    #[allow(clippy::declare_interior_mutable_const)]
    static VALUE_BUFFERS: RefCell<Vec<HashMap<OutputPort, Const>>>
        = const { RefCell::new(Vec::new()) };
}

#[derive(Debug)]
pub struct ConstantStore {
    values: HashMap<OutputPort, Const>,
    tape_len: u16,
}

impl ConstantStore {
    pub fn new(tape_len: u16) -> Self {
        let values = VALUE_BUFFERS
            .with_borrow_mut(|buffers| buffers.pop())
            .unwrap_or_default();
        debug_assert!(values.is_empty());

        Self { values, tape_len }
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }

    pub fn add<C>(&mut self, source: OutputPort, constant: C)
    where
        C: Into<Const>,
    {
        let constant = constant.into();
        let replaced = self.values.insert(source, constant);
        debug_assert!(replaced == None || replaced == Some(constant));
    }

    pub fn remove(&mut self, source: OutputPort) {
        self.values.remove(&source);
    }

    pub fn get(&self, source: OutputPort) -> Option<Const> {
        self.values.get(&source).copied()
    }

    pub fn ptr(&self, source: OutputPort) -> Option<Ptr> {
        self.values
            .get(&source)
            .map(|value| value.into_ptr(self.tape_len))
    }

    pub fn ptr_is_zero(&self, source: OutputPort) -> bool {
        self.ptr(source).map_or(false, Ptr::is_zero)
    }

    pub fn cell(&self, source: OutputPort) -> Option<Cell> {
        self.values.get(&source).copied().map(Const::into_cell)
    }

    pub fn bool(&self, source: OutputPort) -> Option<bool> {
        self.values.get(&source).and_then(Const::as_bool)
    }

    pub fn theta_invariant_inputs_into(
        &self,
        theta: &Theta,
        graph: &Rvsdg,
        destination: &mut Self,
    ) {
        for (input, param) in theta.invariant_input_pairs() {
            if let Some(constant) = self.get(graph.input_source(input)) {
                destination.add(param.output(), constant);
            }
        }
    }

    #[track_caller]
    pub fn gamma_inputs_into(
        &mut self,
        gamma: &Gamma,
        graph: &Rvsdg,
        true_branch: &mut Self,
        false_brach: &mut Self,
    ) {
        for (&input, &[true_param, false_param]) in gamma.inputs().iter().zip(gamma.input_params())
        {
            let source = graph.input_source(input);

            if let Some(constant) = self.get(source) {
                let true_param = gamma.true_branch().to_node::<InputParam>(true_param);
                true_branch.add(true_param.output(), constant);

                let false_param = gamma.false_branch().to_node::<InputParam>(false_param);
                false_brach.add(false_param.output(), constant);
            }
        }
    }

    /// Get the constant store's tape len
    pub fn tape_len(&self) -> u16 {
        self.tape_len
    }

    pub fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
    }
}

impl Clone for ConstantStore {
    fn clone(&self) -> Self {
        let values =
            if let Some(mut values) = VALUE_BUFFERS.with_borrow_mut(|buffers| buffers.pop()) {
                values.clone_from(&self.values);
                values
            } else {
                self.values.clone()
            };

        Self {
            values,
            tape_len: self.tape_len,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.values.clone_from(&source.values);
        self.tape_len = source.tape_len;
    }
}

impl Drop for ConstantStore {
    fn drop(&mut self) {
        if !thread::panicking() && self.values.capacity() != 0 {
            let mut values = take(&mut self.values);
            values.clear();

            VALUE_BUFFERS.with_borrow_mut(|buffers| buffers.push(values));
        }
    }
}
