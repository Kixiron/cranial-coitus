use crate::{
    graph::{Gamma, InputParam, OutputPort, Rvsdg, Theta},
    ir::Const,
    values::{Cell, Ptr},
};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct ConstantStore {
    values: BTreeMap<OutputPort, Const>,
    tape_len: u16,
}

impl ConstantStore {
    pub fn new(tape_len: u16) -> Self {
        Self {
            values: BTreeMap::new(),
            tape_len,
        }
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
}
