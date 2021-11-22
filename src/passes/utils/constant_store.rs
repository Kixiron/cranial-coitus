use crate::{
    graph::{Gamma, InputParam, OutputPort, Rvsdg, Theta},
    ir::Const,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct ConstantStore {
    values: BTreeMap<OutputPort, Const>,
}

impl ConstantStore {
    pub fn new() -> Self {
        Self {
            values: BTreeMap::new(),
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

    pub fn get(&self, source: OutputPort) -> Option<Const> {
        self.values.get(&source).copied()
    }

    pub fn i32(&self, source: OutputPort) -> Option<i32> {
        self.values.get(&source).and_then(Const::convert_to_i32)
    }

    pub fn u8(&self, source: OutputPort) -> Option<u8> {
        self.values.get(&source).and_then(Const::convert_to_u8)
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
