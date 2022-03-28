use crate::graph::{
    nodes::node_ext::{InputPortKinds, InputPorts, OutputPortKinds, OutputPorts},
    EdgeCount, EdgeDescriptor, EdgeKind, InputPort, NodeExt, NodeId, OutputPort,
};
use tinyvec::tiny_vec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scan {
    /// The id of the scan node
    node: NodeId,

    /// The direction the scan goes in, forwards or backwards
    direction: ScanDirection,

    /// The pointer to start scanning at
    ptr: InputPort,

    /// The distance the scan steps by
    step: InputPort,

    /// The byte values being scanned for
    needle: InputPort,

    /// A pointer to a cell containing one of the requested needles,
    /// effectively `ptr ± offset` where `offset` is the offset to a cell
    /// with one of the requested needles and `ptr` is `self.ptr`.
    ///
    /// The relative offset can be either positive or negative (that is,
    /// the output pointer can be greater or less than the input pointer)
    /// no matter the scan direction because of the wrapping semantics of
    /// the program tape.
    ///
    /// Loops infinitely if no cells with the proper needle value can be found
    output_ptr: OutputPort,

    /// Input effect stream
    input_effect: InputPort,

    /// Output effect stream
    output_effect: OutputPort,
}

impl Scan {
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::graph) const fn new(
        node: NodeId,
        direction: ScanDirection,
        ptr: InputPort,
        step: InputPort,
        needle: InputPort,
        output_ptr: OutputPort,
        input_effect: InputPort,
        output_effect: OutputPort,
    ) -> Self {
        Self {
            node,
            direction,
            ptr,
            step,
            needle,
            output_ptr,
            input_effect,
            output_effect,
        }
    }

    /// Get the scan's direction
    pub const fn direction(&self) -> ScanDirection {
        self.direction
    }

    /// Get the scan's pointer
    pub const fn ptr(&self) -> InputPort {
        self.ptr
    }

    /// Get the scan's step
    pub const fn step(&self) -> InputPort {
        self.step
    }

    /// Get the scan's output pointer
    pub const fn output_ptr(&self) -> OutputPort {
        self.output_ptr
    }

    /// Get the scan's needle
    pub const fn needle(&self) -> InputPort {
        self.needle
    }

    /// Get the scan's input effect
    pub const fn input_effect(&self) -> InputPort {
        self.input_effect
    }

    /// Get the scan's output effect
    pub const fn output_effect(&self) -> OutputPort {
        self.output_effect
    }
}

impl NodeExt for Scan {
    fn node(&self) -> NodeId {
        self.node
    }

    fn input_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::new(EdgeCount::one(), EdgeCount::exact(3))
    }

    fn all_input_ports(&self) -> InputPorts {
        tiny_vec![self.ptr, self.step, self.needle, self.input_effect]
    }

    fn all_input_port_kinds(&self) -> InputPortKinds {
        tiny_vec![[_; 4] =>
            (self.ptr, EdgeKind::Value),
            (self.step, EdgeKind::Value),
            (self.needle, EdgeKind::Value),
            (self.input_effect, EdgeKind::Effect),
        ]
    }

    fn update_input(&mut self, from: InputPort, to: InputPort) {
        if from == self.ptr {
            self.ptr = to;
        }

        if from == self.step {
            self.step = to;
        }

        if from == self.needle {
            self.needle = to;
        }

        if from == self.input_effect {
            self.input_effect = to;
        }
    }

    fn output_desc(&self) -> EdgeDescriptor {
        EdgeDescriptor::new(EdgeCount::one(), EdgeCount::one())
    }

    fn all_output_ports(&self) -> OutputPorts {
        tiny_vec![self.output_ptr, self.output_effect]
    }

    fn all_output_port_kinds(&self) -> OutputPortKinds {
        tiny_vec![[_; 4] =>
            (self.output_ptr, EdgeKind::Value),
            (self.output_effect, EdgeKind::Effect),
        ]
    }

    fn update_output(&mut self, from: OutputPort, to: OutputPort) {
        if self.output_ptr == from {
            self.output_ptr = to;
        }

        if self.output_effect == from {
            self.output_effect = to;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScanDirection {
    Forward,
    Backward,
}

impl ScanDirection {
    /// Returns `true` if the scan direction is [`Forward`].
    ///
    /// [`Forward`]: ScanDirection::Forward
    #[must_use]
    pub const fn is_forward(&self) -> bool {
        matches!(self, Self::Forward)
    }

    /// Returns `true` if the scan direction is [`Backward`].
    ///
    /// [`Backward`]: ScanDirection::Backward
    #[must_use]
    pub const fn is_backward(&self) -> bool {
        matches!(self, Self::Backward)
    }
}
