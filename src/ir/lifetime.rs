use crate::ir::{
    Add, Assign, AssignTag, Call, Cmp, Expr, Gamma, Instruction, LifetimeEnd, Load, Mul, Neg, Not,
    Store, Sub, Theta, Value, VarId,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn analyze_block<F>(instructions: &mut Vec<Instruction>, mut skip_lifetime: F)
where
    F: FnMut(VarId) -> bool,
{
    let mut last_usage = BTreeMap::new();
    for (idx, inst) in instructions.iter_mut().enumerate() {
        analyze_instruction(&mut last_usage, idx, inst);
    }

    let mut final_uses = BTreeMap::new();
    for (var, index) in last_usage {
        final_uses
            .entry(index)
            .or_insert_with(|| Vec::with_capacity(1))
            .push(var);
    }

    let (mut idx, mut offset) = (0, 0);
    while idx < instructions.len() {
        if let Some(vars) = final_uses.get(&(idx - offset)) {
            for &var in vars {
                if !skip_lifetime(var) {
                    instructions.insert(idx + 1, Instruction::LifetimeEnd(LifetimeEnd::new(var)));

                    offset += 1;
                    idx += 1;
                }
            }
        }

        idx += 1;
    }
}

fn analyze_instruction(
    last_usage: &mut BTreeMap<VarId, usize>,
    idx: usize,
    inst: &mut Instruction,
) {
    match inst {
        Instruction::Call(Call { args, .. }) => {
            for arg in args {
                if let Value::Var(var) = *arg {
                    last_usage.insert(var, idx);
                }
            }
        }

        &mut Instruction::Assign(Assign {
            var,
            ref value,
            tag,
            ..
        }) => {
            last_usage.insert(var, idx);

            if !tag.is_input_param() {
                match *value {
                    Expr::Cmp(Cmp { lhs, rhs, .. }) => {
                        if let Value::Var(var) = lhs {
                            last_usage.insert(var, idx);
                        }

                        if let Value::Var(var) = rhs {
                            last_usage.insert(var, idx);
                        }
                    }

                    Expr::Add(Add { lhs, rhs }) => {
                        if let Value::Var(var) = lhs {
                            last_usage.insert(var, idx);
                        }

                        if let Value::Var(var) = rhs {
                            last_usage.insert(var, idx);
                        }
                    }

                    Expr::Sub(Sub { lhs, rhs }) => {
                        if let Value::Var(var) = lhs {
                            last_usage.insert(var, idx);
                        }

                        if let Value::Var(var) = rhs {
                            last_usage.insert(var, idx);
                        }
                    }

                    Expr::Mul(Mul { lhs, rhs }) => {
                        if let Value::Var(var) = lhs {
                            last_usage.insert(var, idx);
                        }

                        if let Value::Var(var) = rhs {
                            last_usage.insert(var, idx);
                        }
                    }

                    Expr::Not(Not { value }) => {
                        if let Value::Var(var) = value {
                            last_usage.insert(var, idx);
                        }
                    }

                    Expr::Neg(Neg { value }) => {
                        if let Value::Var(var) = value {
                            last_usage.insert(var, idx);
                        }
                    }

                    Expr::Load(Load { ptr, .. }) => {
                        if let Value::Var(var) = ptr {
                            last_usage.insert(var, idx);
                        }
                    }

                    Expr::Call(Call { ref args, .. }) => {
                        for &arg in args {
                            if let Value::Var(var) = arg {
                                last_usage.insert(var, idx);
                            }
                        }
                    }

                    Expr::Value(value) => {
                        if let Value::Var(var) = value {
                            last_usage.insert(var, idx);
                        }
                    }
                }
            }
        }

        Instruction::Theta(Theta {
            body,
            inputs,
            outputs,
            cond,
            ..
        }) => {
            // Collect all the keys we need to preserve
            let mut saved_keys = BTreeSet::new();
            if let Some(Value::Var(cond)) = *cond {
                saved_keys.insert(cond);
            }
            saved_keys.extend(outputs.values().filter_map(Value::as_var));

            for var in inputs
                .values()
                .filter_map(Value::as_var)
                .chain(outputs.keys().copied())
            {
                last_usage.insert(var, idx);
                saved_keys.insert(var);
            }

            analyze_block(body, |var| saved_keys.contains(&var));

            last_usage.extend(body.iter().filter_map(|inst| {
                if let &Instruction::Assign(Assign {
                    tag: AssignTag::InputParam(_),
                    // TODO: Do we need to analyze sub-expressions?
                    value: Expr::Value(Value::Var(var)),
                    ..
                }) = inst
                {
                    Some((var, idx))
                } else {
                    None
                }
            }));
        }

        Instruction::Gamma(Gamma {
            cond,
            true_branch,
            true_outputs,
            false_branch,
            false_outputs,
            ..
        }) => {
            if let Value::Var(var) = *cond {
                last_usage.insert(var, idx);
            }

            last_usage.extend(
                true_outputs
                    .keys()
                    .chain(false_outputs.keys())
                    .copied()
                    .map(|var| (var, idx)),
            );

            let input_vars = true_branch
                .iter()
                .chain(false_branch.iter())
                .filter_map(|inst| {
                    if let Instruction::Assign(Assign {
                        tag: AssignTag::InputParam(_),
                        // TODO: Do we need to analyze sub-expressions?
                        value: Expr::Value(Value::Var(var)),
                        ..
                    }) = inst
                    {
                        Some(*var)
                    } else {
                        None
                    }
                });
            last_usage.extend(input_vars.clone().map(|var| (var, idx)));

            let mut saved_keys = BTreeSet::new();
            saved_keys.extend(true_outputs.keys().chain(false_outputs.keys()).copied());
            saved_keys.extend(input_vars);

            analyze_block(true_branch, |var| saved_keys.contains(&var));
            analyze_block(false_branch, |var| saved_keys.contains(&var));
        }

        &mut Instruction::Store(Store { ptr, value, .. }) => {
            if let Value::Var(var) = ptr {
                last_usage.insert(var, idx);
            }

            if let Value::Var(var) = value {
                last_usage.insert(var, idx);
            }
        }

        Instruction::LifetimeEnd(_) => unreachable!(),
    }
}
