use crate::{
    ir::{
        AssignTag, Block as CirBlock, Expr as CirExpr, Instruction as CirInstruction, Pretty,
        PrettyConfig, VarId, Variance,
    },
    jit::{
        basic_block::{
            Add, Assign, BlockId, Blocks, Branch, Eq, Input, Instruction, Load, Mul, Neg, Not,
            Output, Phi, RValue, Store, Sub, Terminator, Type, ValId, Value,
        },
        block_builder::BlockBuilder,
        RETURN_SUCCESS,
    },
    utils::AssertNone,
};
use std::collections::BTreeMap;

pub fn translate(block: &CirBlock) -> Blocks {
    let (mut builder, mut phis) = (BlockBuilder::new(), BTreeMap::new());
    for inst in block {
        translate_inst(&mut builder, &mut phis, inst, None);
    }

    // Set the final block to return zero for success
    builder
        .current()
        .set_terminator(Terminator::Return(Value::U8(RETURN_SUCCESS)));
    builder.finalize();

    // Run a few really basic optimization passes to clean up from codegen
    remove_noop_jumps(&mut builder);

    // TODO: Intelligent block ordering
    let blocks = Blocks::new(builder.into_blocks());
    tracing::debug!(
        "produced ssa ir: {}",
        blocks.pretty_print(PrettyConfig::minimal()),
    );

    blocks
}

fn remove_noop_jumps(builder: &mut BlockBuilder) {
    let blocks: Vec<_> = builder.blocks.keys().copied().collect();
    for block in blocks {
        let mut should_collapse = None;
        // if let Some(block) = builder.blocks.get(&block) {
        //     if block.instructions().is_empty() {
        //         if let Some(target) = block.terminator().as_jump() {
        //             should_collapse = Some((block.id(), target, builder.entry() == block.id()));
        //         }
        //     }
        // }

        if let Some((source, target, is_entry)) = should_collapse {
            // If the block is the entry point, set the entry point to the block it jumps to
            if is_entry {
                builder.set_entry(target);
            }

            // Reroute all blocks that jump to this block to instead jump to the new target
            for block in builder.blocks.values_mut() {
                // FIXME: This is broken, `target` isn't the proper predecessor block
                // Correct any phi nodes originating from the removed block
                for inst in block.instructions_mut() {
                    if let Some(phi) = inst
                        .as_mut_assign()
                        .and_then(|assign| assign.rval_mut().as_mut_phi())
                    {
                        if phi.lhs_src() == source {
                            *phi.lhs_src_mut() = target;
                        }
                        if phi.rhs_src() == source {
                            *phi.rhs_src_mut() = target;
                        }
                    }
                }

                if let Some(jump_target) = block.terminator_mut().as_mut_jump() {
                    if *jump_target == source {
                        *jump_target = target;
                    }
                } else if let Some(branch) = block.terminator_mut().as_mut_branch() {
                    if branch.true_jump() == source {
                        branch.set_true_jump(target);
                    }

                    if branch.false_jump() == source {
                        branch.set_false_jump(target);
                    }
                }
            }

            // Remove the redundant block
            builder.blocks.remove(&source).debug_unwrap();
        }
    }
}

fn translate_inst(
    builder: &mut BlockBuilder,
    phis: &mut BTreeMap<VarId, Vec<(BlockId, ValId, bool)>>,
    inst: &CirInstruction,
    previous_block: Option<BlockId>,
) {
    match inst {
        CirInstruction::Call(call) => {
            if call.function != "output" {
                panic!(
                    "got standalone call `{}()` when `output()` was expected",
                    call.function,
                );
            } else if call.args.len() != 1 {
                panic!(
                    "got {} args to `output` call when one was expected",
                    call.args.len(),
                );
            }

            let value = builder.get(call.args[0]);
            builder.push(Output::new(value));
        }

        CirInstruction::Assign(assign) => match &assign.value {
            CirExpr::Eq(eq) => {
                let val = builder.create_val();

                let lhs = builder.get(eq.lhs);
                let rhs = builder.get(eq.rhs);
                // debug_assert_eq!(lhs.ty(), rhs.ty());

                builder.assign(assign.var, (val, Type::Bool));
                builder.push(Assign::new(val, Eq::new(lhs, rhs)));
            }

            CirExpr::Add(add) => {
                let val = builder.create_val();
                let lhs = builder.get(add.lhs);
                let rhs = builder.get(add.rhs);
                // debug_assert_eq!(lhs.ty(), rhs.ty());

                builder.assign(assign.var, (val, lhs.ty()));
                builder.push(Assign::new(val, Add::new(lhs, rhs)));
            }

            CirExpr::Sub(sub) => {
                let val = builder.create_val();
                let lhs = builder.get(sub.lhs);
                let rhs = builder.get(sub.rhs);
                // debug_assert_eq!(lhs.ty(), rhs.ty());

                builder.assign(assign.var, (val, lhs.ty()));
                builder.push(Assign::new(val, Sub::new(lhs, rhs)));
            }

            CirExpr::Mul(mul) => {
                let val = builder.create_val();
                let lhs = builder.get(mul.lhs);
                let rhs = builder.get(mul.rhs);
                // debug_assert_eq!(lhs.ty(), rhs.ty());

                builder.assign(assign.var, (val, lhs.ty()));
                builder.push(Assign::new(val, Mul::new(lhs, rhs)));
            }

            // TODO: Not vs. BitNot
            CirExpr::Not(not) => {
                let val = builder.create_val();
                let value = builder.get(not.value);

                builder.assign(assign.var, (val, value.ty()));
                builder.push(Assign::new(val, Not::new(value)));
            }

            CirExpr::Neg(neg) => {
                let val = builder.create_val();
                let value = builder.get(neg.value);

                builder.assign(assign.var, (val, value.ty()));
                builder.push(Assign::new(val, Neg::new(value)));
            }

            CirExpr::Load(load) => {
                let val = builder.create_val();
                let ptr = builder.get(load.ptr);

                builder.assign(assign.var, (val, Type::U8));
                builder.push(Assign::new(val, Load::new(ptr)));
            }

            CirExpr::Call(call) => {
                if call.function != "input" {
                    panic!(
                        "got assign call `{}()` when `input()` was expected",
                        call.function,
                    );
                } else if !call.args.is_empty() {
                    panic!(
                        "got {} args to `input` call when zero were expected",
                        call.args.len(),
                    );
                }

                let val = builder.create_val();
                // Input always returns a byte from this perspective, the
                // internal implementation returns a u16 but that's all
                // just an internal detail
                builder.assign(assign.var, (val, Type::U8));
                builder.push(Assign::new(val, Input::new()));
            }

            &CirExpr::Value(value) => {
                if let AssignTag::InputParam(Variance::Variant { feedback_from }) = assign.tag {
                    let val = builder.create_val();
                    let lhs = builder.get(value);

                    builder.assign(assign.var, (val, lhs.ty()));

                    let previous_block = previous_block.unwrap();
                    builder.push(Assign::new(
                        val,
                        Phi::new(lhs, previous_block, lhs, previous_block),
                    ));

                    phis.entry(feedback_from)
                        .or_insert_with(|| Vec::with_capacity(1))
                        .push((builder.current_block_id(), val, false));
                } else {
                    let value = builder.get(value);
                    let source_block = builder.current_block_id();

                    // Resolve the rhs of phi nodes
                    if assign.tag.is_output_param() {
                        if let Some(phis) = phis.get(&assign.var) {
                            for &(block, val, resolve_lhs) in phis {
                                let block = if builder.current_block_id() == block {
                                    builder.current()
                                } else {
                                    builder.blocks.get_mut(&block).unwrap()
                                };

                                for inst in block.instructions_mut() {
                                    if let Instruction::Assign(assign) = inst {
                                        if assign.value() == val {
                                            if let RValue::Phi(phi) = assign.rval_mut() {
                                                if resolve_lhs {
                                                    *phi.lhs_mut() = value;
                                                    *phi.lhs_src_mut() = source_block;
                                                } else {
                                                    *phi.rhs_mut() = value;
                                                    *phi.rhs_src_mut() = source_block;
                                                }

                                                // assert_eq!(phi.lhs().ty(), phi.rhs().ty());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    builder.assign(assign.var, value);
                }
            }
        },

        CirInstruction::Theta(theta) => {
            let current_block = builder.current_block_id();
            let [body, tail] = builder.allocate_blocks();
            let body_id = body.id();

            builder.current().set_terminator(Terminator::Jump(body_id));

            // TODO: Phi nodes for params
            builder.create_block(body);
            for inst in &theta.body {
                translate_inst(builder, phis, inst, Some(current_block));
            }

            // If `cond` is true, jump to the head of the theta body. Otherwise
            // continue to the next block of instructions
            let cond = builder.get(theta.cond.unwrap());
            builder
                .current()
                .set_terminator(Terminator::Branch(Branch::new(cond, body_id, tail.id())));

            // Create & enter a new block for the following instructions
            builder.create_block(tail);
        }

        CirInstruction::Gamma(gamma) => {
            let current_block = builder.current_block_id();
            let [true_block, false_block, tail] = builder.allocate_blocks();

            // Branch to either of the branches based off of the gamma condition
            // TODO: There's optimizations to be made with empty branches
            let cond = builder.get(gamma.cond);
            builder
                .current()
                .set_terminator(Terminator::Branch(Branch::new(
                    cond,
                    true_block.id(),
                    false_block.id(),
                )));

            builder.create_block(true_block);
            for inst in &gamma.true_branch {
                if !inst.is_output_param() {
                    translate_inst(builder, phis, inst, Some(current_block));
                }
            }
            let true_branch_id = builder.current_block_id();

            // After executing the branch, jump to the instructions after the gamma
            builder
                .current()
                .set_terminator(Terminator::Jump(tail.id()));

            builder.create_block(false_block);
            for inst in &gamma.false_branch {
                if !inst.is_output_param() {
                    translate_inst(builder, phis, inst, Some(current_block));
                }
            }
            let false_branch_id = builder.current_block_id();

            // After executing the branch, jump to the instructions after the gamma
            builder
                .current()
                .set_terminator(Terminator::Jump(tail.id()));

            // Create & enter a new block for the following instructions
            builder.create_block(tail);

            for (var_id, true_val) in gamma.true_outputs.iter() {
                let val = builder.create_val();
                let true_val = builder.get(*true_val);
                let false_val = builder.get(*gamma.false_outputs.get(var_id).unwrap());
                debug_assert_eq!(true_val.ty(), false_val.ty());

                builder.assign(*var_id, (val, true_val.ty()));
                builder.push(Assign::new(
                    val,
                    Phi::new(true_val, true_branch_id, false_val, false_branch_id),
                ));
            }
        }

        CirInstruction::Store(store) => {
            let ptr = builder.get(store.ptr);
            let value = builder.get(store.value);

            builder.push(Store::new(ptr, value));
        }

        // Ignore lifetimes
        // TODO: Delete these from the CIR entirely?
        CirInstruction::LifetimeEnd(_) => {}
    }
}
