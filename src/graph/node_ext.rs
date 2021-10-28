use crate::graph::{EdgeDescriptor, InputPort, NodeId, OutputPort};
use tinyvec::TinyVec;

pub trait NodeExt {
    fn node(&self) -> NodeId;

    fn input_desc(&self) -> EdgeDescriptor;

    fn all_input_ports(&self) -> TinyVec<[InputPort; 4]>;

    fn output_desc(&self) -> EdgeDescriptor;

    fn all_output_ports(&self) -> TinyVec<[OutputPort; 4]>;
}
