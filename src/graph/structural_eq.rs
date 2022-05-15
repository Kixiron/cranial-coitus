use crate::graph::{NodeExt, Rvsdg};
use std::{
    cmp::min,
    collections::{BTreeSet, VecDeque},
};

impl Rvsdg {
    pub fn structural_eq(&self, other: &Self) -> bool {
        if self.start_nodes.len() != other.start_nodes.len()
            || self.end_nodes.len() != other.end_nodes.len()
        {
            return false;
        } else if self.start_nodes.len() != 1 || self.end_nodes.len() != 1 {
            panic!(
                "encountered graphs with {} start and {} end nodes",
                self.start_nodes.len(),
                self.end_nodes.len(),
            )
        }

        let (mut queue, mut equal) = (
            VecDeque::with_capacity(min(self.nodes.len(), other.nodes.len()) / 2),
            BTreeSet::new(),
        );
        queue.push_back((self.end_nodes[0], other.end_nodes[0]));

        while let Some((lhs, rhs)) = queue.pop_back() {
            equal.insert((lhs, rhs));
            let (lhs, rhs) = (&self.nodes[&lhs], &other.nodes[&rhs]);

            if lhs.kind() != rhs.kind()
                || matches!(lhs.as_int_value().zip(rhs.as_int_value()), Some((lhs, rhs)) if lhs != rhs)
                || matches!(lhs.as_byte_value().zip(rhs.as_byte_value()), Some((lhs, rhs)) if lhs != rhs)
                || matches!(lhs.as_bool_value().zip(rhs.as_bool_value()), Some((lhs, rhs)) if lhs != rhs)
            {
                return false;
            } else if let Some((lhs, rhs)) = lhs.as_theta().zip(rhs.as_theta()) {
                // TODO: Theta subgraphs
            } else if let Some((lhs, rhs)) = lhs.as_gamma().zip(rhs.as_gamma()) {
                // TODO: Gamma subgraphs
            }

            // Input edges
            for ((lhs, lhs_kind), (rhs, rhs_kind)) in lhs
                .all_input_port_kinds()
                .into_iter()
                .zip(rhs.all_input_port_kinds())
            {
                if lhs_kind != rhs_kind {
                    return false;
                }

                let pair = (self.input_source_id(lhs), other.input_source_id(rhs));
                if !equal.contains(&pair) {
                    queue.push_back(pair);
                }
            }

            // Output edges
            for ((lhs, lhs_kind), (rhs, rhs_kind)) in lhs
                .all_output_port_kinds()
                .into_iter()
                .zip(rhs.all_output_port_kinds())
            {
                if lhs_kind != rhs_kind {
                    return false;
                }

                let pair = (
                    self.output_dest_id(lhs).unwrap(),
                    other.output_dest_id(rhs).unwrap(),
                );
                if !equal.contains(&pair) {
                    queue.push_back(pair);
                }
            }
        }

        true
    }
}
