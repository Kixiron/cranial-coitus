use crate::{
    ir::{CmpKind, Pretty, PrettyConfig},
    jit::{
        basic_block::{Instruction, Load, Output, RValue, Scanl, Scanr, Type, Value},
        codegen::{assert_type, Codegen},
        ffi, State,
    },
    utils::AssertNone,
    values::{Cell, Ptr},
};
use cranelift::prelude::{
    types::{B1, I16, I32, I64, I8},
    InstBuilder, IntCC, MemFlags, StackSlotData, StackSlotKind, Type as ClifType,
    Value as ClifValue,
};
use std::{cmp::Ordering, ops::Not};

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
                        self.resize_int(value, ty, I8)
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
                                self.resize_int(value, ty, I8)
                            }
                        };
                        let rhs = match cmp.rhs() {
                            Value::U8(byte) => self.builder.ins().iconst(I8, byte),
                            Value::U16(int) => self.builder.ins().iconst(I16, int.0 as i64),
                            Value::TapePtr(uint) => self.builder.ins().iconst(I8, uint),
                            Value::Bool(bool) => self.builder.ins().iconst(I8, bool as i64),
                            Value::Val(value, _) => {
                                let (value, ty) = self.values[&value];
                                self.resize_int(value, ty, I8)
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
                        tracing::warn!("codegen negated unsigned value");

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
                    RValue::Not(not) => self.codegen_bitwise_not(not.value()),
                    RValue::BitNot(bit_not) => self.codegen_bitwise_not(bit_not.value()),

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
                                        // TODO: Proper wrapping
                                        self.resize_int(value, ty, I8)
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
                                        // TODO: Proper wrapping
                                        self.resize_int(value, ty, I16)
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
                                        // TODO: Proper wrapping?
                                        self.resize_int(value, ty, I32)
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
                                        self.resize_int(value, ty, expected_ty)
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
                            Value::TapePtr(uint) => self.builder.ins().iconst(I64, uint),
                            Value::Bool(_) => unreachable!(),
                            Value::Val(value, _) => {
                                let (value, ty) = self.values[&value];
                                self.resize_int(value, ty, I64)
                            }
                        };
                        let rhs = match sub.rhs() {
                            Value::U8(byte) => self.builder.ins().iconst(I64, byte),
                            Value::U16(int) => self.builder.ins().iconst(I64, int.0 as i64),
                            Value::TapePtr(uint) => self.builder.ins().iconst(I64, uint),
                            Value::Bool(_) => unreachable!(),
                            Value::Val(value, _ty) => {
                                let (value, ty) = self.values[&value];
                                self.resize_int(value, ty, I64)
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
                                self.resize_int(value, ty, I64)
                            }
                        };
                        let rhs = match mul.rhs() {
                            Value::U8(byte) => self.builder.ins().iconst(I64, byte),
                            Value::U16(int) => self.builder.ins().iconst(I64, int.0 as i64),
                            Value::TapePtr(uint) => self.builder.ins().iconst(I64, uint),
                            Value::Bool(_) => unreachable!(),
                            Value::Val(value, _ty) => {
                                let (value, ty) = self.values[&value];
                                self.resize_int(value, ty, I64)
                            }
                        };

                        // TODO: Optimize to `.imul_imm()` when either side is an immediate value
                        (self.builder.ins().imul(lhs, rhs), I64)
                    }

                    // TODO: Bounds checking & wrapping on pointers
                    RValue::Load(load) => self.codegen_load(load),

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

                    RValue::Scanr(scanr) => self.codegen_scanr(scanr),

                    RValue::Scanl(scanl) => self.codegen_scanl(scanl),
                };

                self.values
                    .insert(assign.value(), value)
                    .debug_unwrap_none();
            }

            Instruction::Output(output) => self.codegen_output(output),
        }
    }

    fn codegen_scanr(&mut self, scanr: &Scanr) -> (ClifValue, ClifType) {
        assert_type::<unsafe extern "fastcall" fn(*const State, u16, u16, u8) -> u32>(
            ffi::scanr_wrapping,
        );
        assert_type::<unsafe extern "fastcall" fn(*const State, u16, u16, u8) -> u32>(
            ffi::scanr_non_wrapping,
        );

        let ptr = match scanr.ptr() {
            Value::U8(byte) => self.builder.ins().iconst(I16, byte),
            Value::U16(int) => self.builder.ins().iconst(I16, int.0 as i64),
            Value::TapePtr(uint) => self.builder.ins().iconst(I16, uint),
            Value::Bool(_) => unreachable!(),
            Value::Val(value, _ty) => {
                let (value, ty) = self.values[&value];
                self.resize_int(value, ty, I16)
            }
        };
        let step = match scanr.step() {
            Value::U8(byte) => self.builder.ins().iconst(I16, byte),
            Value::U16(int) => self.builder.ins().iconst(I16, int.0 as i64),
            Value::TapePtr(uint) => self.builder.ins().iconst(I16, uint),
            Value::Bool(_) => unreachable!(),
            Value::Val(value, _ty) => {
                let (value, ty) = self.values[&value];
                self.resize_int(value, ty, I16)
            }
        };
        let needle = match scanr.needle() {
            Value::U8(byte) => self.builder.ins().iconst(I8, byte),
            Value::U16(int) => self.builder.ins().iconst(I8, int.0 as i64),
            Value::TapePtr(uint) => self.builder.ins().iconst(I8, uint),
            Value::Bool(_) => unreachable!(),
            Value::Val(value, _ty) => {
                let (value, ty) = self.values[&value];
                self.resize_int(value, ty, I8)
            }
        };

        // Create a block to house the stuff that happens after the function
        // call and associated error check
        let call_prelude = self.builder.create_block();

        // Call the input function
        let (scanr_func, state_ptr) = (self.scanr_function(), self.state_ptr());
        let scanr_call = self
            .builder
            .ins()
            .call(scanr_func, &[state_ptr, ptr, step, needle]);
        let scanr_value = self.builder.inst_results(scanr_call)[0];

        // If the call to scanr failed (and therefore returned usize::MAX),
        // branch to the scan error handler
        let error_handler = self.scan_error_handler();
        let u32_max = self.builder.ins().iconst(I32, u32::MAX as i64);
        self.builder
            .ins()
            .br_icmp(IntCC::Equal, scanr_value, u32_max, error_handler, &[]);

        // Otherwise if the call didn't fail, so we use the return value as the pointer
        self.builder.ins().jump(call_prelude, &[]);
        self.builder.switch_to_block(call_prelude);

        (scanr_value, I32)
    }

    fn codegen_scanl(&mut self, scanl: &Scanl) -> (ClifValue, ClifType) {
        assert_type::<unsafe extern "fastcall" fn(*const State, u16, u16, u8) -> u32>(
            ffi::scanl_wrapping,
        );
        assert_type::<unsafe extern "fastcall" fn(*const State, u16, u16, u8) -> u32>(
            ffi::scanl_non_wrapping,
        );

        let ptr = match scanl.ptr() {
            Value::U8(byte) => self.builder.ins().iconst(I16, byte),
            Value::U16(int) => self.builder.ins().iconst(I16, int.0 as i64),
            Value::TapePtr(uint) => self.builder.ins().iconst(I16, uint),
            Value::Bool(_) => unreachable!(),
            Value::Val(value, _ty) => {
                let (value, ty) = self.values[&value];
                self.resize_int(value, ty, I16)
            }
        };
        let step = match scanl.step() {
            Value::U8(byte) => self.builder.ins().iconst(I16, byte),
            Value::U16(int) => self.builder.ins().iconst(I16, int.0 as i64),
            Value::TapePtr(uint) => self.builder.ins().iconst(I16, uint),
            Value::Bool(_) => unreachable!(),
            Value::Val(value, _ty) => {
                let (value, ty) = self.values[&value];
                self.resize_int(value, ty, I16)
            }
        };
        let needle = match scanl.needle() {
            Value::U8(byte) => self.builder.ins().iconst(I8, byte),
            Value::U16(int) => self.builder.ins().iconst(I8, int.0 as i64),
            Value::TapePtr(uint) => self.builder.ins().iconst(I8, uint),
            Value::Bool(_) => unreachable!(),
            Value::Val(value, _ty) => {
                let (value, ty) = self.values[&value];
                self.resize_int(value, ty, I8)
            }
        };

        // Create a block to house the stuff that happens after the function
        // call and associated error check
        let call_prelude = self.builder.create_block();

        // Call the input function
        let (scanl_func, state_ptr) = (self.scanl_function(), self.state_ptr());
        let scanr_call = self
            .builder
            .ins()
            .call(scanl_func, &[state_ptr, ptr, step, needle]);
        let scanl_value = self.builder.inst_results(scanr_call)[0];

        // If the call to scanl failed (and therefore returned usize::MAX),
        // branch to the scan error handler
        let error_handler = self.scan_error_handler();
        let u32_max = self.builder.ins().iconst(I32, u32::MAX as i64);
        self.builder
            .ins()
            .br_icmp(IntCC::Equal, scanl_value, u32_max, error_handler, &[]);

        // Otherwise if the call didn't fail, so we use the return value as the pointer
        self.builder.ins().jump(call_prelude, &[]);
        self.builder.switch_to_block(call_prelude);

        (scanl_value, I32)
    }

    fn codegen_bitwise_not(&mut self, value: Value) -> (ClifValue, ClifType) {
        match value {
            Value::U8(byte) => (self.builder.ins().iconst(I8, !byte), I8),
            Value::U16(int) => (self.builder.ins().iconst(I16, int.not().0 as i64), I16),
            Value::TapePtr(uint) => (self.builder.ins().iconst(I32, !uint), I32),
            Value::Bool(bool) => (self.builder.ins().bconst(B1, !bool), B1),
            Value::Val(value, _ty) => {
                let (value, ty) = self.values[&value];
                (self.builder.ins().bnot(value), ty)
            }
        }
    }

    fn codegen_load(&mut self, load: &Load) -> (ClifValue, ClifType) {
        let tape_start = self.tape_start();

        // Get the value to store
        let (ptr, offset) = match load.ptr() {
            Value::U8(byte) => (tape_start, byte.into_ptr(self.tape_len).value() as i32),

            Value::U16(int) => (tape_start, Ptr::new(int.0, self.tape_len).value() as i32),

            Value::TapePtr(uint) => (tape_start, uint.value() as i32),

            Value::Val(offset, _ty) => {
                let (value, ty) = self.values[&offset];
                let offset = self.resize_int(value, ty, I64);

                // TODO: Overflow handling
                // TODO: Analysis to figure out if certain adds can overflow
                let ptr = self.builder.ins().iadd(tape_start, offset);

                (ptr, 0)
            }

            Value::Bool(_) => unreachable!(),
        };

        // All of our loads are known to be aligned and not trap
        let flags = MemFlags::trusted();
        let loaded = self.builder.ins().load(I8, flags, ptr, offset);

        // Loads from the program tape always return bytes
        (loaded, I8)
    }

    fn codegen_output(&mut self, output: &Output) {
        assert_type::<unsafe extern "fastcall" fn(*mut State, *const u8, usize) -> bool>(
            ffi::output,
        );

        // Create a block to house the stuff that happens after the function
        // call and associated error check
        let call_prelude = self.builder.create_block();

        // If every value is a constant we can create a static array instead of
        // building one on the stack
        let bytes_ptr = if output.values().iter().all(Value::is_const) {
            let values = output.values().iter().map(|&value| {
                match value {
                    Value::U8(byte) => byte,
                    Value::U16(int) => Cell::from(int.0),
                    Value::TapePtr(ptr) => ptr.into_cell(),
                    Value::Bool(_) => unreachable!("cannot pass a boolean to an output call"),
                    Value::Val(_, _) => unreachable!("all values are const"),
                }
                .into_inner()
            });

            // Create a static value containing all the bytes to be output
            let static_data = self
                .static_readonly_slice(values)
                .expect("failed to create slice for output");

            // Get a pointer to the static data
            self.builder.ins().symbol_value(self.ptr_type, static_data)

        // Otherwise we build a stack value (is there a more efficient way to do this?)
        } else {
            // Create a stack slot to hold the arguments
            let bytes_slot = self.builder.create_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                output.values().len() as u32,
            ));

            // Store the arguments into the stack slot
            for (offset, &value) in output.values().iter().enumerate() {
                let value = match value {
                    Value::U8(byte) => self.builder.ins().iconst(I8, byte),
                    Value::U16(int) => self.builder.ins().iconst(I8, Cell::from(int)),
                    Value::TapePtr(uint) => self.builder.ins().iconst(I8, uint.into_cell()),
                    Value::Bool(_) => unreachable!("cannot pass a boolean to an output call"),
                    Value::Val(value, _ty) => {
                        let (value, ty) = self.values[&value];
                        self.resize_int(value, ty, I8)
                    }
                };

                self.builder
                    .ins()
                    .stack_store(value, bytes_slot, offset as i32);
            }

            self.builder.ins().stack_addr(self.ptr_type, bytes_slot, 0)
        };

        // Get the address of the stack slot and the number of arguments in it
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

    #[track_caller]
    fn resize_int(&mut self, value: ClifValue, from: ClifType, to: ClifType) -> ClifValue {
        #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
        enum Type {
            I8 = 0,
            I16 = 1,
            I32 = 2,
            I64 = 3,
        }

        impl From<ClifType> for Type {
            #[track_caller]
            fn from(ty: ClifType) -> Self {
                if ty == I8 {
                    Self::I8
                } else if ty == I16 {
                    Self::I16
                } else if ty == I32 {
                    Self::I32
                } else if ty == I64 {
                    Self::I64
                } else {
                    panic!("invalid type given to resize_int: {:?}", ty)
                }
            }
        }

        // TODO: Cache the resized versions of things so we only do it once
        let (from_ty, to_ty) = (Type::from(from), Type::from(to));
        match from_ty.cmp(&to_ty) {
            Ordering::Less => self.builder.ins().uextend(to, value),
            Ordering::Equal => value,
            Ordering::Greater => self.builder.ins().ireduce(to, value),
        }
    }
}
