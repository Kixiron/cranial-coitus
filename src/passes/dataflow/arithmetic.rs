use crate::{
    graph::{Add, Rvsdg, Sub},
    passes::dataflow::{
        domain::{ByteSet, Domain, IntSet},
        Dataflow,
    },
    values::Ptr,
};

impl Dataflow {
    pub(super) fn compute_add(&mut self, graph: &mut Rvsdg, add: Add) -> Option<()> {
        let (lhs, rhs) = (graph.input_source(add.lhs()), graph.input_source(add.rhs()));
        // TODO: We can use default values when a domain can't be found
        let (lhs_domain, rhs_domain) = (self.domain(lhs)?, self.domain(rhs)?);

        let sum = match (lhs_domain, rhs_domain) {
            // Operations over bytes are subject to cell wrapping rules
            (Domain::Byte(lhs), Domain::Byte(rhs)) => {
                let mut output = ByteSet::empty();

                // We have to do a cartesian product of both sets in order to create the possible outputs
                if self.settings.cell_operations_wrap {
                    for lhs in lhs.iter() {
                        for rhs in rhs.iter() {
                            output.insert(lhs.wrapping_add(rhs));
                        }
                    }

                // If cell operations don't wrap we can eliminate any values that would wrap
                } else {
                    for lhs in lhs.iter() {
                        for rhs in rhs.iter() {
                            // If an overflow occurs when cell operations cannot wrap, ignore the sum
                            if let Some(sum) = lhs.checked_add(rhs) {
                                output.insert(sum);
                            }
                        }
                    }
                }

                Domain::Byte(output)
            }

            // Byte & integer additions always produce an integer
            (Domain::Int(lhs), Domain::Byte(rhs)) | (Domain::Byte(rhs), Domain::Int(lhs)) => {
                let mut output = IntSet::empty(self.tape_len());

                // We have to do a cartesian product of both sets in order to create the possible outputs
                if self.settings.tape_operations_wrap {
                    for lhs in lhs.iter() {
                        for rhs in rhs.iter() {
                            output.add(lhs.wrapping_add_u8(rhs));
                        }
                    }

                // If tape operations don't wrap we can eliminate any values that would wrap
                } else {
                    for lhs in lhs.iter() {
                        for rhs in rhs.iter() {
                            // If an overflow occurs when tape operations cannot wrap, ignore the sum
                            if let Some(sum) = lhs.checked_add_u8(rhs) {
                                output.add(sum);
                            }
                        }
                    }
                }

                Domain::Int(output)
            }

            (Domain::Int(lhs), Domain::Int(rhs)) => {
                let mut output = IntSet::empty(self.tape_len());

                // We have to do a cartesian product of both sets in order to create the possible outputs
                if self.settings.tape_operations_wrap {
                    for lhs in lhs.iter() {
                        for rhs in rhs.iter() {
                            output.add(lhs.wrapping_add(rhs));
                        }
                    }

                // If tape operations don't wrap we can eliminate any values that would wrap
                } else {
                    for lhs in lhs.iter() {
                        for rhs in rhs.iter() {
                            // If an overflow occurs when tape operations cannot wrap, ignore the sum
                            if let Some(sum) = lhs.checked_add(rhs) {
                                output.add(sum);
                            }
                        }
                    }
                }

                Domain::Int(output)
            }

            // We don't perform addition on booleans
            (Domain::Bool(_), _) | (_, Domain::Bool(_)) => return None,
        };

        // If there's only one value produced by the addition,
        // replace it with a constant
        if self.can_mutate {
            match &sum {
                Domain::Byte(sum) => {
                    if let Some(sum) = sum.as_singleton() {
                        let node = graph.byte(sum);
                        self.add_domain(node.value(), sum);
                        graph.rewire_dependents(add.value(), node.value());
                        self.changes.inc::<"const-add">();
                    }
                }
                Domain::Int(sum) => {
                    if let Some(sum) = sum.as_singleton() {
                        let node = graph.int(sum);
                        self.add_domain(node.value(), sum);
                        graph.rewire_dependents(add.value(), node.value());
                        self.changes.inc::<"const-add">();
                    }
                }
                Domain::Bool(_) => unreachable!(),
            }
        }

        self.add_domain(add.value(), sum);
        None
    }

    pub(super) fn compute_sub(&mut self, graph: &mut Rvsdg, sub: Sub) -> Option<()> {
        let (lhs, rhs) = (graph.input_source(sub.lhs()), graph.input_source(sub.rhs()));
        // TODO: We can use default values when a domain can't be found
        let (lhs_domain, rhs_domain) = (self.domain(lhs)?, self.domain(rhs)?);

        let difference = match (lhs_domain, rhs_domain) {
            // Operations over bytes are subject to cell wrapping rules
            (Domain::Byte(lhs), Domain::Byte(rhs)) => {
                let mut output = ByteSet::empty();

                // We have to do a cartesian product of both sets in order to create the possible outputs
                if self.settings.cell_operations_wrap {
                    for lhs in lhs.iter() {
                        for rhs in rhs.iter() {
                            output.insert(lhs.wrapping_sub(rhs));
                        }
                    }

                // If cell operations don't wrap we can eliminate any values that would wrap
                } else {
                    for lhs in lhs.iter() {
                        for rhs in rhs.iter() {
                            // If an overflow occurs when cell operations cannot wrap, ignore the difference
                            if let Some(difference) = lhs.checked_sub(rhs) {
                                output.insert(difference);
                            }
                        }
                    }
                }

                Domain::Byte(output)
            }

            // Byte & integer subtractions always produce an integer
            (Domain::Int(lhs), Domain::Byte(rhs)) => {
                let mut output = IntSet::empty(self.tape_len());

                // We have to do a cartesian product of both sets in order to create the possible outputs
                if self.settings.tape_operations_wrap {
                    for lhs in lhs.iter() {
                        for rhs in rhs.iter() {
                            output.add(lhs.wrapping_sub_u8(rhs));
                        }
                    }

                // If tape operations don't wrap we can eliminate any values that would wrap
                } else {
                    for lhs in lhs.iter() {
                        for rhs in rhs.iter() {
                            // If an overflow occurs when tape operations cannot wrap, ignore the difference
                            if let Some(difference) = lhs.checked_sub_u8(rhs) {
                                output.add(difference);
                            }
                        }
                    }
                }

                Domain::Int(output)
            }
            (Domain::Byte(lhs), Domain::Int(rhs)) => {
                let mut output = IntSet::empty(self.tape_len());

                // We have to do a cartesian product of both sets in order to create the possible outputs
                if self.settings.tape_operations_wrap {
                    for lhs in lhs.iter() {
                        for rhs in rhs.iter() {
                            output.add(Ptr::wrapping_sub_ptr_u8(lhs, rhs));
                        }
                    }

                // If tape operations don't wrap we can eliminate any values that would wrap
                } else {
                    for lhs in lhs.iter() {
                        for rhs in rhs.iter() {
                            // If an overflow occurs when tape operations cannot wrap, ignore the difference
                            if let Some(difference) = Ptr::checked_sub_ptr_u8(lhs, rhs) {
                                output.add(difference);
                            }
                        }
                    }
                }

                Domain::Int(output)
            }

            (Domain::Int(lhs), Domain::Int(rhs)) => {
                let mut output = IntSet::empty(self.tape_len());

                // We have to do a cartesian product of both sets in order to create the possible outputs
                if self.settings.tape_operations_wrap {
                    for lhs in lhs.iter() {
                        for rhs in rhs.iter() {
                            output.add(lhs.wrapping_sub(rhs));
                        }
                    }

                // If tape operations don't wrap we can eliminate any values that would wrap
                } else {
                    for lhs in lhs.iter() {
                        for rhs in rhs.iter() {
                            // If an overflow occurs when tape operations cannot wrap, ignore the difference
                            if let Some(difference) = lhs.checked_sub(rhs) {
                                output.add(difference);
                            }
                        }
                    }
                }

                Domain::Int(output)
            }

            // We don't perform subtraction on booleans
            (Domain::Bool(_), _) | (_, Domain::Bool(_)) => return None,
        };

        // If there's only one value produced by the subtraction,
        // replace it with a constant
        if self.can_mutate {
            match &difference {
                Domain::Byte(difference) => {
                    if let Some(difference) = difference.as_singleton() {
                        let node = graph.byte(difference);
                        self.add_domain(node.value(), difference);
                        graph.rewire_dependents(sub.value(), node.value());
                        self.changes.inc::<"const-sub">();
                    }
                }
                Domain::Int(difference) => {
                    if let Some(difference) = difference.as_singleton() {
                        let node = graph.int(difference);
                        self.add_domain(node.value(), difference);
                        graph.rewire_dependents(sub.value(), node.value());
                        self.changes.inc::<"const-sub">();
                    }
                }
                Domain::Bool(_) => unreachable!(),
            }
        }

        self.add_domain(sub.value(), difference);
        None
    }
}
