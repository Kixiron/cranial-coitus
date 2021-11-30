use crate::ir::{
    Add, Assign, AssignTag, Call, Eq, Expr, Gamma, Instruction, LifetimeEnd, Load, Mul, Neg, Not,
    Store, Theta, Value, VarId,
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

        Instruction::Assign(Assign { var, value, .. }) => {
            last_usage.insert(*var, idx);

            match value {
                Expr::Eq(Eq { lhs, rhs }) => {
                    if let Value::Var(var) = *lhs {
                        last_usage.insert(var, idx);
                    }

                    if let Value::Var(var) = *rhs {
                        last_usage.insert(var, idx);
                    }
                }

                Expr::Add(Add { lhs, rhs }) => {
                    if let Value::Var(var) = *lhs {
                        last_usage.insert(var, idx);
                    }

                    if let Value::Var(var) = *rhs {
                        last_usage.insert(var, idx);
                    }
                }

                Expr::Mul(Mul { lhs, rhs }) => {
                    if let Value::Var(var) = *lhs {
                        last_usage.insert(var, idx);
                    }

                    if let Value::Var(var) = *rhs {
                        last_usage.insert(var, idx);
                    }
                }

                &mut Expr::Not(Not { value }) => {
                    if let Value::Var(var) = value {
                        last_usage.insert(var, idx);
                    }
                }

                &mut Expr::Neg(Neg { value }) => {
                    if let Value::Var(var) = value {
                        last_usage.insert(var, idx);
                    }
                }

                &mut Expr::Load(Load { ptr, .. }) => {
                    if let Value::Var(var) = ptr {
                        last_usage.insert(var, idx);
                    }
                }

                Expr::Call(Call { args, .. }) => {
                    for &mut arg in args {
                        if let Value::Var(var) = arg {
                            last_usage.insert(var, idx);
                        }
                    }
                }

                &mut Expr::Value(value) => {
                    if let Value::Var(var) = value {
                        last_usage.insert(var, idx);
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
            for &var in inputs.keys().chain(outputs.keys()) {
                last_usage.insert(var, idx);
            }

            analyze_block(body, |var| {
                outputs.contains_key(&var)
                    || inputs.contains_key(&var)
                    || outputs.values().any(|value| *value == Value::Var(var))
                    || inputs.values().any(|value| *value == Value::Var(var))
                    || Some(Value::Var(var)) == *cond
            });
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

            for &var in true_outputs.keys().chain(false_outputs.keys()) {
                last_usage.insert(var, idx);
            }

            let mut inputs = BTreeSet::new();
            for inst in true_branch.iter().chain(false_branch.iter()) {
                if let Instruction::Assign(Assign {
                    var,
                    tag: AssignTag::InputParam(_),
                    ..
                }) = *inst
                {
                    last_usage.insert(var, idx);
                    inputs.insert(var);
                }
            }

            analyze_block(true_branch, |var| {
                true_outputs.contains_key(&var)
                    || inputs.contains(&var)
                    || true_outputs.values().any(|value| *value == Value::Var(var))
            });

            analyze_block(false_branch, |var| {
                false_outputs.contains_key(&var)
                    || inputs.contains(&var)
                    || false_outputs
                        .values()
                        .any(|value| *value == Value::Var(var))
            });
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
