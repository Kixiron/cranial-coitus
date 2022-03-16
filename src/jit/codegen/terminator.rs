use crate::{
    jit::{
        basic_block::{BasicBlock, BlockId, Terminator, ValId, Value},
        codegen::Codegen,
    },
    utils::HashMap,
};
use cranelift::{
    frontend::FunctionBuilder,
    prelude::{
        types::{B1, I16, I32, I8},
        InstBuilder, TrapCode, Type, Value as ClifValue,
    },
};

impl<'a> Codegen<'a> {
    pub(super) fn terminator(&mut self, current_block: &BasicBlock, terminator: &Terminator) {
        match terminator {
            // FIXME: Block arguments
            &Terminator::Jump(target) => {
                let params = collect_block_params(
                    current_block.id(),
                    target,
                    &mut self.builder,
                    &self.values,
                    self.ssa,
                    &mut self.param_buffer,
                );
                self.builder.ins().jump(self.blocks[&target], params);
            }

            // FIXME: Block arguments
            Terminator::Branch(branch) => {
                let cond = match branch.condition() {
                    Value::U8(byte) => self.builder.ins().iconst(I8, byte),
                    Value::U16(int) => self.builder.ins().iconst(I16, int.0 as i64),
                    Value::TapePtr(uint) => self.builder.ins().iconst(I32, uint),
                    Value::Bool(bool) => self.builder.ins().bconst(B1, bool),
                    Value::Val(value, _) => self.values[&value].0,
                };

                let true_params = collect_block_params(
                    current_block.id(),
                    branch.true_jump(),
                    &mut self.builder,
                    &self.values,
                    self.ssa,
                    &mut self.param_buffer,
                );

                // Jump to the true branch if the condition is true (non-zero)
                self.builder
                    .ins()
                    .brnz(cond, self.blocks[&branch.true_jump()], true_params);

                let false_params = collect_block_params(
                    current_block.id(),
                    branch.false_jump(),
                    &mut self.builder,
                    &self.values,
                    self.ssa,
                    &mut self.param_buffer,
                );

                // Otherwise jump to the false branch
                self.builder
                    .ins()
                    .jump(self.blocks[&branch.false_jump()], false_params);
            }

            &Terminator::Return(value) => {
                // FIXME: return should only accept u8 values
                let value = match value {
                    Value::U8(byte) => self.builder.ins().iconst(I8, byte),
                    Value::U16(int) => self.builder.ins().iconst(I8, int.0 as i64),
                    Value::TapePtr(uint) => self.builder.ins().iconst(I8, uint),
                    Value::Bool(bool) => self.builder.ins().iconst(I8, bool as i64),
                    Value::Val(value, _) => {
                        let (value, ty) = self.values[&value];
                        if ty != I8 {
                            self.builder.ins().ireduce(I8, value)
                        } else {
                            value
                        }
                    }
                };

                self.builder.ins().return_(&[value]);
            }

            // Unreachable code can, by definition, never be reached
            Terminator::Unreachable => {
                self.builder.ins().trap(TrapCode::UnreachableCodeReached);
            }

            // We shouldn't have errors by this point in compilation
            Terminator::Error => unreachable!(),
        }
    }
}

// TODO: Refactor this
fn collect_block_params<'a>(
    current: BlockId,
    target: BlockId,
    builder: &mut FunctionBuilder,
    values: &HashMap<ValId, (ClifValue, Type)>,
    ssa: &[BasicBlock],
    param_buffer: &'a mut Vec<ClifValue>,
) -> &'a [ClifValue] {
    param_buffer.clear();
    param_buffer.extend(
        ssa.iter()
            .find(|block| block.id() == target)
            .unwrap()
            .instructions()
            .iter()
            .filter_map(|inst| {
                inst.as_assign()
                    .and_then(|assign| assign.rval().as_phi())
                    .and_then(|phi| {
                        if phi.lhs_src() == current {
                            Some(phi.lhs())
                        } else if phi.rhs_src() == current {
                            Some(phi.rhs())
                        } else {
                            None
                        }
                    })
                    .and_then(|value| match value {
                        Value::U8(byte) => Some(builder.ins().iconst(I8, byte)),
                        Value::U16(int) => Some(builder.ins().iconst(I8, int.0 as i64)),
                        Value::TapePtr(uint) => Some(builder.ins().iconst(I32, uint)),
                        Value::Bool(bool) => Some(builder.ins().bconst(B1, bool)),
                        // FIXME: Sometimes the value doesn't exist???
                        Value::Val(value, _) => values.get(&value).map(|&(value, _)| value),
                    })
            }),
    );

    &*param_buffer
}
