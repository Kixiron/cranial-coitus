use std::ops::Not;

use cranelift::prelude::{
    types::{B1, I16, I32, I64, I8},
    InstBuilder, IntCC, MemFlags, StackSlotData, StackSlotKind,
};

use crate::{
    ir::{CmpKind, Pretty, PrettyConfig},
    jit::{
        basic_block::{Instruction, RValue, Type, Value},
        codegen::{assert_type, Codegen},
        ffi, State,
    },
    utils::AssertNone,
    values::{Cell, Ptr},
};

impl<'a> Codegen<'a> {
    pub(super) fn instruction(&mut self, inst: &Instruction) {
        match inst {
            Instruction::Store(store) => {
                // Get the value to store
                let value = match store.value() {
                    Value::U8(byte) => self.builder.ins().iconst(I8, byte),
                    Value::U16(int) => self.builder.ins().iconst(I8, Cell::from(int)),
                    Value::TapePtr(uint) => self.builder.ins().iconst(I8, uint.into_cell()),
                    Value::Val(value, _) => {
                        let (value, ty) = self.values[&value];
                        if ty == I8 {
                            value
                        } else {
                            // TODO: Proper wrapping
                            self.builder.ins().ireduce(I8, value)
                        }
                    }
                    Value::Bool(_) => unreachable!(),
                };

                if !store.ptr().ty().is_ptr() {
                    tracing::error!(
                        "stored to non-pointer: {}",
                        inst.pretty_print(PrettyConfig::minimal()),
                    );
                }

                let tape_start = self.tape_start();
                match store.ptr() {
                    // We can optimize stores with const-known offsets to use
                    // constant offsets instead of dynamic ones
                    Value::U8(byte) => {
                        self.builder.ins().store(
                            MemFlags::trusted(),
                            value,
                            tape_start,
                            byte.into_ptr(self.tape_len),
                        );
                    }
                    Value::U16(int) => {
                        self.builder.ins().store(
                            MemFlags::trusted(),
                            value,
                            tape_start,
                            Ptr::new(int.0, self.tape_len),
                        );
                    }
                    Value::TapePtr(uint) => {
                        self.builder
                            .ins()
                            .store(MemFlags::trusted(), value, tape_start, uint);
                    }

                    // FIXME: Bounds checking & wrapping on pointers
                    Value::Val(offset, _ty) => {
                        let offset = {
                            let (value, ty) = self.values[&offset];
                            if ty != I64 {
                                self.builder.ins().uextend(I64, value)
                            } else {
                                value
                            }
                        };
                        let pointer = self.builder.ins().iadd(tape_start, offset);

                        // // Tape pointer bounds checking
                        // let inbounds = self.builder.ins().icmp(
                        //     // `tape_end` points to the *last* element of the tape
                        //     IntCC::UnsignedLessThanOrEqual,
                        //     pointer,
                        //     tape_end,
                        // );
                        //
                        // // Note: Doesn't wrap properly :/
                        // let checked_pointer = builder.ins(inbounds, pointer, tape_end).select();

                        self.builder.ins().store(MemFlags::new(), value, pointer, 0);
                    }

                    Value::Bool(_) => unreachable!(),
                }
            }

            Instruction::Assign(assign) => {
                let value = match assign.rval() {
                    RValue::Cmp(cmp) => {
                        let lhs = match cmp.lhs() {
                            Value::U8(byte) => self.builder.ins().iconst(I8, byte),
                            Value::U16(int) => self.builder.ins().iconst(I16, int.0 as i64),
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
                        let rhs = match cmp.rhs() {
                            Value::U8(byte) => self.builder.ins().iconst(I8, byte),
                            Value::U16(int) => self.builder.ins().iconst(I16, int.0 as i64),
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

                        // Figure out what kind of comparison we're doing
                        let cmp_kind = match cmp.kind() {
                            CmpKind::Eq => IntCC::Equal,
                            CmpKind::Neq => IntCC::NotEqual,
                            CmpKind::Less => IntCC::UnsignedLessThan,
                            CmpKind::Greater => IntCC::UnsignedGreaterThan,
                            CmpKind::LessEq => IntCC::UnsignedLessThanOrEqual,
                            CmpKind::GreaterEq => IntCC::UnsignedGreaterThanOrEqual,
                        };

                        // TODO: Optimize to `.icmp_imm()` when either side is an immediate value
                        (self.builder.ins().icmp(cmp_kind, lhs, rhs), B1)
                    }

                    RValue::Phi(phi) => {
                        // Figure out the phi value's type
                        let ty = match phi.lhs() {
                            Value::U8(_) => I8,
                            Value::U16(_) => I16,
                            Value::TapePtr(_) => I32,
                            Value::Bool(_) => B1,
                            Value::Val(value, _ty) => self.values[&value].1,
                        };

                        // Create the phi value
                        let phi = self
                            .builder
                            .append_block_param(self.builder.current_block().unwrap(), ty);

                        (phi, ty)
                    }

                    // TODO: How much does neg actually make sense? Is it even needed or generated?
                    RValue::Neg(neg) => {
                        let (value, ty) = match neg.value() {
                            Value::U8(byte) => (self.builder.ins().iconst(I8, byte), I8),
                            Value::U16(int) => (self.builder.ins().iconst(I16, int.0 as i64), I16),
                            Value::TapePtr(uint) => (self.builder.ins().iconst(I32, uint), I32),
                            Value::Bool(_) => unreachable!("cannot negate a boolean"),
                            Value::Val(value, _ty) => self.values[&value],
                        };

                        (self.builder.ins().ineg(value), ty)
                    }

                    // TODO: What cranelift does is pretty clever tbh, we should do this
                    //       for CIR and our bb form since all you need is type info to
                    //       do this properly so we can just have a single "not" node
                    RValue::Not(not) => match not.value() {
                        Value::U8(byte) => (self.builder.ins().iconst(I8, !byte), I8),
                        Value::U16(int) => {
                            (self.builder.ins().iconst(I16, int.not().0 as i64), I16)
                        }
                        Value::TapePtr(uint) => (self.builder.ins().iconst(I32, !uint), I32),
                        Value::Bool(bool) => (self.builder.ins().bconst(B1, !bool), B1),
                        Value::Val(value, _ty) => {
                            let (value, ty) = self.values[&value];
                            (self.builder.ins().bnot(value), ty)
                        }
                    },
                    RValue::BitNot(bit_not) => match bit_not.value() {
                        Value::U8(byte) => (self.builder.ins().iconst(I8, !byte), I8),
                        Value::U16(int) => {
                            (self.builder.ins().iconst(I16, int.not().0 as i64), I16)
                        }
                        Value::TapePtr(uint) => (self.builder.ins().iconst(I32, !uint), I32),
                        Value::Bool(bool) => (self.builder.ins().bconst(B1, !bool), B1),
                        Value::Val(value, _ty) => {
                            let (value, ty) = self.values[&value];
                            (self.builder.ins().bnot(value), ty)
                        }
                    },

                    // FIXME: This is hell
                    RValue::Add(add) => {
                        let (lhs, rhs, ty) = match add.lhs() {
                            Value::U8(byte) => {
                                let lhs = self.builder.ins().iconst(I8, byte);
                                let rhs = match add.rhs() {
                                    Value::U8(byte) => self.builder.ins().iconst(I8, byte),
                                    Value::U16(int) => {
                                        self.builder.ins().iconst(I8, Cell::from(int))
                                    }
                                    Value::TapePtr(uint) => {
                                        self.builder.ins().iconst(I8, uint.into_cell())
                                    }
                                    Value::Bool(_) => unreachable!(),
                                    Value::Val(value, _) => {
                                        let (value, ty) = self.values[&value];
                                        if ty == I8 {
                                            value
                                        // TODO: Proper wrapping?
                                        } else if ty == I16 || ty == I32 || ty == I64 {
                                            self.builder.ins().ireduce(I8, value)
                                        } else {
                                            panic!("{}", ty)
                                        }
                                    }
                                };

                                (lhs, rhs, I8)
                            }
                            Value::U16(int) => {
                                let lhs = self.builder.ins().iconst(I16, int.0 as i64);
                                let rhs = match add.rhs() {
                                    Value::U8(byte) => self.builder.ins().iconst(I16, byte),
                                    Value::U16(int) => self.builder.ins().iconst(I16, int.0 as i64),
                                    Value::TapePtr(uint) => self.builder.ins().iconst(I16, uint),
                                    Value::Bool(_) => unreachable!(),
                                    Value::Val(value, _) => {
                                        let (value, ty) = self.values[&value];
                                        if ty == I16 {
                                            value
                                        } else if ty == I8 {
                                            self.builder.ins().uextend(I16, value)
                                        // TODO: Proper wrapping?
                                        } else if ty == I32 || ty == I64 {
                                            self.builder.ins().ireduce(I16, value)
                                        } else {
                                            panic!("{}", ty)
                                        }
                                    }
                                };

                                (lhs, rhs, I16)
                            }
                            Value::TapePtr(uint) => {
                                // FIXME: Should use I16 for pointer
                                let lhs = self.builder.ins().iconst(I32, uint.value() as i64);
                                let rhs = match add.rhs() {
                                    Value::U8(byte) => self.builder.ins().iconst(I32, byte),
                                    Value::U16(int) => self
                                        .builder
                                        .ins()
                                        .iconst(I32, Ptr::new(int.0, self.tape_len)),
                                    Value::TapePtr(uint) => self.builder.ins().iconst(I32, uint),
                                    Value::Bool(_) => unreachable!(),
                                    Value::Val(value, _) => {
                                        let (value, ty) = self.values[&value];
                                        if ty == I32 {
                                            value
                                        // TODO: Proper wrapping for I16?
                                        } else if ty == I8 || ty == I16 {
                                            self.builder.ins().uextend(I32, value)
                                        // TODO: Proper wrapping?
                                        } else if ty == I64 {
                                            self.builder.ins().ireduce(I32, value)
                                        } else {
                                            panic!("{}", ty)
                                        }
                                    }
                                };

                                (lhs, rhs, I32)
                            }
                            Value::Val(value, ty) => {
                                let expected_ty = match ty {
                                    Type::U8 => I8,
                                    Type::U16 => I16,
                                    Type::Ptr => I32,
                                    Type::Bool => unreachable!(),
                                };
                                let (value, clif_ty) = self.values[&value];

                                let lhs = match ty {
                                    Type::U8 if clif_ty == I8 => value,
                                    Type::U8 => self.builder.ins().ireduce(I8, value),

                                    Type::U16 if clif_ty == I16 => value,
                                    Type::U16 if clif_ty == I8 => {
                                        self.builder.ins().uextend(I16, value)
                                    }
                                    Type::U16 => self.builder.ins().ireduce(I16, value),

                                    Type::Ptr if clif_ty == I32 => value,
                                    Type::Ptr if clif_ty == I8 || clif_ty == I16 => {
                                        self.builder.ins().uextend(I32, value)
                                    }
                                    Type::Ptr => self.builder.ins().ireduce(I32, value),

                                    Type::Bool => unreachable!(),
                                };

                                let rhs = match add.rhs() {
                                    Value::U8(byte) => self.builder.ins().iconst(expected_ty, byte),
                                    Value::U16(int) => self.builder.ins().iconst(
                                        expected_ty,
                                        if ty == Type::Ptr {
                                            Ptr::new(int.0, self.tape_len).value() as i64
                                        } else {
                                            int.0 as i64
                                        },
                                    ),
                                    Value::TapePtr(uint) => {
                                        self.builder.ins().iconst(expected_ty, uint)
                                    }
                                    Value::Val(value, _) => {
                                        let (value, ty) = self.values[&value];
                                        // TODO: Proper wrapping?
                                        if ty == expected_ty {
                                            value
                                        } else if ty == I8 {
                                            self.builder.ins().uextend(expected_ty, value)
                                        } else if ty == I16 && expected_ty == I8 {
                                            self.builder.ins().ireduce(expected_ty, value)
                                        } else if ty == I16 || (ty == I32 && expected_ty == I64) {
                                            self.builder.ins().uextend(expected_ty, value)
                                        } else if ty == I32 || ty == I64 {
                                            self.builder.ins().ireduce(expected_ty, value)
                                        } else {
                                            panic!("{}", ty)
                                        }
                                    }
                                    Value::Bool(_) => unreachable!(),
                                };

                                (lhs, rhs, expected_ty)
                            }
                            Value::Bool(_) => unreachable!(),
                        };

                        // TODO: Optimize to `.iadd_imm()` when either side is an immediate value
                        (self.builder.ins().iadd(lhs, rhs), ty)
                    }

                    RValue::Sub(sub) => {
                        let lhs = match sub.lhs() {
                            Value::U8(byte) => self.builder.ins().iconst(I64, byte),
                            Value::U16(int) => self.builder.ins().iconst(I64, int.0 as i64),
                            Value::TapePtr(uint) => self.builder.ins().iconst(I32, uint),
                            Value::Bool(_) => unreachable!(),
                            Value::Val(value, _) => {
                                let (value, ty) = self.values[&value];
                                if ty == I64 {
                                    value
                                } else if ty == I8 || ty == I32 {
                                    self.builder.ins().uextend(I64, value)
                                } else {
                                    panic!("{}", ty)
                                }
                            }
                        };
                        let rhs = match sub.rhs() {
                            Value::U8(byte) => self.builder.ins().iconst(I64, byte),
                            Value::U16(int) => self.builder.ins().iconst(I64, int.0 as i64),
                            Value::TapePtr(uint) => self.builder.ins().iconst(I64, uint),
                            Value::Bool(_) => unreachable!(),
                            Value::Val(value, _ty) => {
                                let (value, ty) = self.values[&value];
                                if ty == I64 {
                                    value
                                } else if ty == I8 || ty == I32 {
                                    self.builder.ins().uextend(I64, value)
                                } else {
                                    panic!("{}", ty)
                                }
                            }
                        };

                        // TODO: Optimize to `.isub_imm()` when either side is an immediate value
                        let sub = self.builder.ins().isub(lhs, rhs);
                        (self.builder.ins().ireduce(I32, sub), I32)
                    }

                    RValue::Mul(mul) => {
                        let lhs = match mul.lhs() {
                            Value::U8(byte) => self.builder.ins().iconst(I64, byte),
                            Value::U16(int) => self.builder.ins().iconst(I64, int.0 as i64),
                            Value::TapePtr(uint) => self.builder.ins().iconst(I64, uint),
                            Value::Bool(_) => unreachable!(),
                            Value::Val(value, _ty) => {
                                let (value, ty) = self.values[&value];
                                if ty == I64 {
                                    value
                                } else if ty == I8 || ty == I32 {
                                    self.builder.ins().uextend(I64, value)
                                } else {
                                    panic!("{}", ty)
                                }
                            }
                        };
                        let rhs = match mul.rhs() {
                            Value::U8(byte) => self.builder.ins().iconst(I64, byte),
                            Value::U16(int) => self.builder.ins().iconst(I64, int.0 as i64),
                            Value::TapePtr(uint) => self.builder.ins().iconst(I64, uint),
                            Value::Bool(_) => unreachable!(),
                            Value::Val(value, _ty) => {
                                let (value, ty) = self.values[&value];
                                if ty == I64 {
                                    value
                                } else if ty == I8 || ty == I32 {
                                    self.builder.ins().uextend(I64, value)
                                } else {
                                    panic!("{}", ty)
                                }
                            }
                        };

                        // TODO: Optimize to `.imul_imm()` when either side is an immediate value
                        (self.builder.ins().imul(lhs, rhs), I64)
                    }

                    // TODO: Bounds checking & wrapping on pointers
                    RValue::Load(load) => {
                        // Get the value to store
                        let offset = match load.ptr() {
                            Value::U8(byte) => self.builder.ins().iconst(I64, byte),
                            Value::U16(int) => self
                                .builder
                                .ins()
                                .iconst(I64, Ptr::new(int.0, self.tape_len)),
                            Value::TapePtr(uint) => self.builder.ins().iconst(I64, uint),
                            Value::Val(offset, _ty) => {
                                let (value, ty) = self.values[&offset];
                                if ty == I64 {
                                    value
                                } else if ty == I8 || ty == I32 {
                                    self.builder.ins().uextend(I64, value)
                                } else {
                                    panic!("{}", ty)
                                }
                            }
                            Value::Bool(_) => unreachable!(),
                        };

                        let tape_start = self.tape_start();
                        let pointer = self.builder.ins().iadd(tape_start, offset);

                        // TODO: Optimize this to use a constant offset instead of add when possible
                        (self.builder.ins().load(I8, MemFlags::new(), pointer, 0), I8)
                    }

                    RValue::Input(_) => {
                        assert_type::<unsafe extern "fastcall" fn(*mut State, *mut u8) -> bool>(
                            ffi::input,
                        );

                        // Create a block to house the stuff that happens after the function
                        // call and associated error check
                        let call_prelude = self.builder.create_block();

                        // Allocate a stack slot for the input value
                        let input_slot = self
                            .builder
                            .create_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 1));
                        let input_slot_addr =
                            self.builder.ins().stack_addr(self.ptr_type, input_slot, 0);

                        // Call the input function
                        let (input_func, state_ptr) = (self.input_function(), self.state_ptr());
                        let input_call = self
                            .builder
                            .ins()
                            .call(input_func, &[state_ptr, input_slot_addr]);
                        let input_failed = self.builder.inst_results(input_call)[0];

                        // If the call to input failed (and therefore returned true),
                        // branch to the io error handler
                        let error_handler = self.io_error_handler();
                        self.builder.ins().brnz(input_failed, error_handler, &[]);

                        // Otherwise if the call didn't fail, so we use the return value from the
                        // call to the input function by discarding the high 8 bits that used
                        // to contain the error status of the function call
                        self.builder.ins().jump(call_prelude, &[]);
                        self.builder.switch_to_block(call_prelude);

                        // Load the input value written to the stack
                        let input_value = self.builder.ins().stack_load(I8, input_slot, 0);

                        (input_value, I8)
                    }
                };

                self.values
                    .insert(assign.value(), value)
                    .debug_unwrap_none();
            }

            Instruction::Output(output) => {
                assert_type::<unsafe extern "fastcall" fn(*mut State, *const u8, usize) -> bool>(
                    ffi::output,
                );

                // Create a block to house the stuff that happens after the function
                // call and associated error check
                let call_prelude = self.builder.create_block();

                // Create a stack slot to hold the arguments
                let bytes_slot = self.builder.create_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    output.values().len() as u32,
                ));

                // Store the arguments into the stack slot
                for (offset, &value) in output.values().iter().enumerate() {
                    let value = match value {
                        Value::U8(byte) => self.builder.ins().iconst(I8, byte),
                        Value::U16(int) => self.builder.ins().iconst(I8, int.0 as i64),
                        Value::TapePtr(uint) => self.builder.ins().iconst(I8, uint),
                        Value::Bool(bool) => self.builder.ins().iconst(I8, bool as i64),
                        Value::Val(value, _ty) => {
                            let (value, ty) = self.values[&value];
                            if ty != I8 {
                                self.builder.ins().ireduce(I8, value)
                            } else {
                                value
                            }
                        }
                    };

                    self.builder
                        .ins()
                        .stack_store(value, bytes_slot, offset as i32);
                }

                // Get the address of the stack slot and the number of arguments in it
                let bytes_ptr = self.builder.ins().stack_addr(self.ptr_type, bytes_slot, 0);
                let bytes_len = self
                    .builder
                    .ins()
                    .iconst(self.ptr_type, output.values().len() as i64);

                // Call the output function
                let (output_func, state_ptr) = (self.output_function(), self.state_ptr());
                let output_call = self
                    .builder
                    .ins()
                    .call(output_func, &[state_ptr, bytes_ptr, bytes_len]);
                let output_result = self.builder.inst_results(output_call)[0];

                // If the call to output failed (and therefore returned true),
                // branch to the io error handler
                let error_handler = self.io_error_handler();
                self.builder.ins().brnz(output_result, error_handler, &[]);

                // Otherwise if the call didn't fail, we don't have any work to do
                self.builder.ins().jump(call_prelude, &[]);
                self.builder.switch_to_block(call_prelude);
            }
        }
    }
}
