use crate::graph::{EdgeDescriptor, InputPort, NodeId, OutputPort};
use tinyvec::TinyVec;

pub trait NodeExt {
    fn node(&self) -> NodeId;

    fn input_desc(&self) -> EdgeDescriptor;

    fn all_input_ports(&self) -> TinyVec<[InputPort; 4]>;

    fn update_inputs<F>(&mut self, mut update: F) -> bool
    where
        F: FnMut(InputPort) -> Option<InputPort>,
    {
        let mut changed = false;
        for input in self.all_input_ports() {
            if let Some(new_input) = update(input) {
                self.update_input(input, new_input);
                changed = true;
            }
        }

        changed
    }

    fn update_input(&mut self, from: InputPort, to: InputPort);

    fn output_desc(&self) -> EdgeDescriptor;

    fn all_output_ports(&self) -> TinyVec<[OutputPort; 4]>;
}
