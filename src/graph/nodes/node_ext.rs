//! The [`NodeExt`] trait for behavior shared between nodes

use crate::graph::{EdgeDescriptor, EdgeKind, InputPort, NodeId, OutputPort};
use tinyvec::TinyVec;

pub type InputPorts = TinyVec<[InputPort; 4]>;
pub type InputPortKinds = TinyVec<[(InputPort, EdgeKind); 4]>;

pub type OutputPorts = TinyVec<[OutputPort; 4]>;
pub type OutputPortKinds = TinyVec<[(OutputPort, EdgeKind); 4]>;

// TODO:
// - .effect_inputs()
// - .value_inputs()
// - .effect_outputs()
// - .value_outputs()
// - .has_effect_inputs()
// - .has_value_inputs()
// - .has_effect_outputs()
// - .has_value_outputs()
// - .update_output()
// - .update_outputs()
// - .num_input_ports()
// - .num_input_effect_ports()
// - .num_input_value_ports()
// - .num_output_ports()
// - .num_output_effect_ports()
// - .num_output_value_ports()
// - .has_input(&self, InputPort) -> bool
// - .has_output(&self, OutputPort) -> bool
pub trait NodeExt {
    fn node(&self) -> NodeId;

    fn input_desc(&self) -> EdgeDescriptor;

    fn all_input_ports(&self) -> InputPorts;

    fn all_input_port_kinds(&self) -> InputPortKinds;

    fn update_inputs<F>(&mut self, mut update: F) -> bool
    where
        F: FnMut(InputPort, EdgeKind) -> Option<InputPort>,
    {
        let mut changed = false;
        for (input, kind) in self.all_input_port_kinds() {
            if let Some(new_input) = update(input, kind) {
                self.update_input(input, new_input);
                changed = true;
            }
        }

        changed
    }

    // TODO: Should this return a bool?
    fn update_input(&mut self, from: InputPort, to: InputPort);

    fn output_desc(&self) -> EdgeDescriptor;

    fn all_output_ports(&self) -> OutputPorts;

    fn all_output_port_kinds(&self) -> OutputPortKinds;

    fn update_outputs<F>(&mut self, mut update: F) -> bool
    where
        F: FnMut(OutputPort, EdgeKind) -> Option<OutputPort>,
    {
        let mut changed = false;
        for (output, kind) in self.all_output_port_kinds() {
            if let Some(new_input) = update(output, kind) {
                self.update_output(output, new_input);
                changed = true;
            }
        }

        changed
    }

    // TODO: Should this return a bool?
    fn update_output(&mut self, from: OutputPort, to: OutputPort);
}
