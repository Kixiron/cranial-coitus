#[macro_use]
mod ffi;
mod basic_block;
mod block_builder;
mod block_visitor;
// mod cir_jit;
mod cir_to_bb;
// mod coloring;
mod disassemble;
// mod liveliness;
mod memory;
// mod regalloc;

pub use ffi::State;
pub use memory::Executable;

use crate::{
    ir::{Block, Pretty, PrettyConfig},
    jit::basic_block::{Instruction, RValue, Terminator, Value},
    utils::AssertNone,
    values::{Cell, Ptr},
};
use anyhow::Result;
use cranelift::{
    codegen::{
        ir::{
            types::{B1, I16, I32, I64, I8},
            InstBuilder,
        },
        Context,
    },
    frontend::{FunctionBuilder, FunctionBuilderContext},
    prelude::{isa::CallConv, AbiParam, IntCC, MemFlags, TrapCode},
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataContext, Linkage, Module};
use std::{collections::BTreeMap, mem::transmute, ops::Not, slice};

/// The function produced by jit code
pub type JitFunction = unsafe extern "fastcall" fn(*mut State, *mut u8, *const u8) -> u8;

const RETURN_SUCCESS: Cell = Cell::zero();

const RETURN_IO_FAILURE: Cell = Cell::new(101);

pub struct Jit {
    /// The function builder context, which is reused across multiple
    /// `FunctionBuilder` instances
    builder_context: FunctionBuilderContext,

    /// The main Cranelift context, which holds the state for codegen. Cranelift
    /// separates this from `Module` to allow for parallel compilation, with a
    /// context per thread, though this isn't in the simple demo here
    ctx: Context,

    /// The data context, which is to data objects what `ctx` is to functions
    data_ctx: DataContext,

    /// The module, with the jit backend, which manages the JIT'd
    /// functions
    module: JITModule,

    /// The length of the (program/turing) tape we're targeting
    tape_len: u16,
}

impl Jit {
    /// Create a new jit
    pub fn new(tape_len: u16) -> Self {
        let mut builder = JITBuilder::new(cranelift_module::default_libcall_names());

        // Add external functions to the module so they're accessible within
        // generated code
        builder.symbols([
            ("input", ffi::input as *const u8),
            ("output", ffi::output as *const u8),
            (
                "io_error_encountered",
                ffi::io_error_encountered as *const u8,
            ),
        ]);

        let module = JITModule::new(builder);
        let mut ctx = module.make_context();
        ctx.set_disasm(true);

        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx,
            data_ctx: DataContext::new(),
            module,
            tape_len,
        }
    }

    /// Compile a block of CIR into an executable buffer
    pub fn compile(mut self, block: &Block) -> Result<(Executable, String, String, String)> {
        let ssa_ir = {
            // Translate CIR into SSA form
            let blocks = cir_to_bb::translate(block);

            // We use SystemV here since it's one of the only calling conventions
            // that both Rust and Cranelift support (Rust functions must use the
            // `sysv64` convention to match this)
            //
            // TODO: I'd really rather use the C calling convention or something,
            //       but as of now I don't think cranelift supports it
            self.ctx.func.signature.call_conv = CallConv::WindowsFastcall;

            // Get the target pointer type
            let ptr = self.module.target_config().pointer_type();
            let ptr_param = AbiParam::new(ptr);

            // State pointer
            self.ctx.func.signature.params.push(ptr_param);

            // Data start pointer
            self.ctx.func.signature.params.push(ptr_param);

            // Data end pointer
            self.ctx.func.signature.params.push(ptr_param);

            // Return a byte from the function
            self.ctx.func.signature.returns.push(AbiParam::new(I8));

            // Translate the SSA ir into cranelift ir
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
            let mut values = BTreeMap::new();

            // Create the input and output functions as well as the io error function
            // Note: These could be lazily initialized, but I'm not sure it really matters
            let input_function = {
                // extern "sysv64" fn(*mut State) -> u16
                let mut sig = self.module.make_signature();
                sig.params.push(ptr_param);
                sig.returns.push(AbiParam::new(I16));
                sig.call_conv = CallConv::WindowsFastcall;

                let callee = self
                    .module
                    .declare_function("input", Linkage::Import, &sig)?;
                self.module.declare_func_in_func(callee, builder.func)
            };
            let output_function = {
                // extern "sysv64" fn(*mut State, u8) -> bool
                let mut sig = self.module.make_signature();
                sig.params.push(ptr_param);
                sig.params.push(AbiParam::new(I8));
                sig.returns.push(AbiParam::new(B1));
                sig.call_conv = CallConv::WindowsFastcall;

                let callee = self
                    .module
                    .declare_function("output", Linkage::Import, &sig)?;
                self.module.declare_func_in_func(callee, builder.func)
            };
            let io_error_function = {
                // extern "sysv64" fn(*mut State) -> bool
                let mut sig = self.module.make_signature();
                sig.params.push(ptr_param);
                sig.returns.push(AbiParam::new(B1));
                sig.call_conv = CallConv::WindowsFastcall;

                let callee =
                    self.module
                        .declare_function("io_error_encountered", Linkage::Import, &sig)?;
                self.module.declare_func_in_func(callee, builder.func)
            };

            let entry = builder.create_block();

            // Create all blocks within the program so that we can later reference them
            let mut ssa_blocks = BTreeMap::new();
            ssa_blocks.extend(
                blocks
                    .iter()
                    .map(|block| (block.id(), builder.create_block())),
            );

            // Get the function's parameters
            let [state, tape_start, tape_end]: [_; 3] = {
                builder.switch_to_block(entry);

                // Add the function parameters to the entry block
                builder.append_block_params_for_function_params(entry);
                builder.ins().jump(ssa_blocks[&blocks[0].id()], &[]);
                builder.seal_block(entry);

                // Retrieve the values of the function parameters
                builder
                    .block_params(entry)
                    .try_into()
                    .expect("got more than three function parameters")
            };

            // A lazily-constructed block to handle IO errors
            let (io_error_handler, mut io_handler_used) = (builder.create_block(), false);

            // let mut block_params = Vec::with_capacity(blocks.len() * 2);
            // for block in &blocks {
            //     for inst in block {
            //         if let Instruction::Assign(assign) = inst {
            //             block_params.extend(
            //                 assign
            //                     .phi_targets()
            //                     .iter()
            //                     .map(|&value| (block.id(), value)),
            //             );
            //         }
            //     }
            // }

            for block in &blocks {
                // Switch to the basic block for the current block
                let ssa_block = ssa_blocks[&block.id()];
                builder.switch_to_block(ssa_block);

                // TODO: Get block inputs by looking at phi targets

                // Translate the instructions within the block
                for inst in block {
                    match inst {
                        Instruction::Store(store) => {
                            // Get the value to store
                            let value = match store.value() {
                                Value::U8(byte) => builder.ins().iconst(I8, byte),
                                Value::U16(int) => builder.ins().iconst(I16, int.0 as i64),
                                Value::TapePtr(uint) => builder.ins().iconst(I8, uint),
                                Value::Val(value, _ty) => {
                                    let (value, ty) = values[&value];
                                    if ty == I8 {
                                        value
                                    } else {
                                        builder.ins().ireduce(I8, value)
                                    }
                                }
                                Value::Bool(_) => unreachable!(),
                            };

                            match store.ptr() {
                                // We can optimize stores with const-known offsets to use
                                // constant offsets instead of dynamic ones
                                Value::U8(byte) => {
                                    builder.ins().store(
                                        MemFlags::new(),
                                        value,
                                        tape_start,
                                        byte.into_ptr(self.tape_len),
                                    );
                                }
                                Value::U16(int) => {
                                    builder.ins().store(
                                        MemFlags::new(),
                                        value,
                                        tape_start,
                                        Ptr::new(int.0, self.tape_len),
                                    );
                                }
                                Value::TapePtr(uint) => {
                                    builder
                                        .ins()
                                        .store(MemFlags::new(), value, tape_start, uint);
                                }

                                // TODO: Bounds checking & wrapping on pointers
                                Value::Val(offset, _ty) => {
                                    let offset = {
                                        let (value, ty) = values[&offset];
                                        if ty != I64 {
                                            builder.ins().uextend(I64, value)
                                        } else {
                                            value
                                        }
                                    };
                                    let pointer = builder.ins().iadd(tape_start, offset);

                                    // // Tape pointer bounds checking
                                    // let inbounds = builder.ins().icmp(
                                    //     // `tape_end` points to the *last* element of the tape
                                    //     IntCC::UnsignedLessThanOrEqual,
                                    //     pointer,
                                    //     tape_end,
                                    // );
                                    //
                                    // // Note: Doesn't wrap properly :/
                                    // let checked_pointer = builder.ins(inbounds, pointer, tape_end).select();

                                    builder.ins().store(MemFlags::new(), value, pointer, 0);
                                }

                                Value::Bool(_) => unreachable!(),
                            }
                        }

                        Instruction::Assign(assign) => {
                            let value = match assign.rval() {
                                RValue::Eq(eq) => {
                                    let lhs = match eq.lhs() {
                                        Value::U8(byte) => builder.ins().iconst(I8, byte),
                                        Value::U16(int) => builder.ins().iconst(I16, int.0 as i64),
                                        Value::TapePtr(uint) => builder.ins().iconst(I8, uint),
                                        Value::Bool(bool) => builder.ins().iconst(I8, bool as i64),
                                        Value::Val(value, _ty) => {
                                            let (value, ty) = values[&value];
                                            if ty != I8 {
                                                builder.ins().ireduce(I8, value)
                                            } else {
                                                value
                                            }
                                        }
                                    };
                                    let rhs = match eq.rhs() {
                                        Value::U8(byte) => builder.ins().iconst(I8, byte),
                                        Value::U16(int) => builder.ins().iconst(I16, int.0 as i64),
                                        Value::TapePtr(uint) => builder.ins().iconst(I8, uint),
                                        Value::Bool(bool) => builder.ins().iconst(I8, bool as i64),
                                        Value::Val(value, _ty) => {
                                            let (value, ty) = values[&value];
                                            if ty != I8 {
                                                builder.ins().ireduce(I8, value)
                                            } else {
                                                value
                                            }
                                        }
                                    };

                                    // TODO: Optimize to `.icmp_imm()` when either side is an immediate value
                                    (builder.ins().icmp(IntCC::Equal, lhs, rhs), B1)
                                }

                                RValue::Phi(phi) => {
                                    // Figure out the phi value's type
                                    let ty = match phi.lhs() {
                                        Value::U8(_) => I8,
                                        Value::U16(_) => I16,
                                        Value::TapePtr(_) => I32,
                                        Value::Bool(_) => B1,
                                        Value::Val(value, _ty) => values[&value].1,
                                    };

                                    // Create the phi value
                                    let phi = builder
                                        .append_block_param(builder.current_block().unwrap(), ty);

                                    (phi, ty)
                                }

                                RValue::Neg(neg) => {
                                    let (value, ty) = match neg.value() {
                                        Value::U8(byte) => (builder.ins().iconst(I8, byte), I8),
                                        Value::U16(int) => {
                                            (builder.ins().iconst(I16, int.not().0 as i64), I16)
                                        }
                                        Value::TapePtr(uint) => {
                                            (builder.ins().iconst(I32, uint), I32)
                                        }
                                        Value::Bool(_) => unreachable!("cannot negate a boolean"),
                                        Value::Val(value, _ty) => values[&value],
                                    };

                                    (builder.ins().ineg(value), ty)
                                }

                                // TODO: What cranelift does is pretty clever tbh, we should do this
                                //       for CIR and our bb form since all you need is type info to
                                //       do this properly so we can just have a single "not" node
                                RValue::Not(not) => match not.value() {
                                    Value::U8(byte) => (builder.ins().iconst(I8, !byte), I8),
                                    Value::U16(int) => {
                                        (builder.ins().iconst(I16, int.not().0 as i64), I16)
                                    }
                                    Value::TapePtr(uint) => (builder.ins().iconst(I32, !uint), I32),
                                    Value::Bool(bool) => (builder.ins().bconst(B1, !bool), B1),
                                    Value::Val(value, _ty) => {
                                        let (value, ty) = values[&value];
                                        (builder.ins().bnot(value), ty)
                                    }
                                },
                                RValue::BitNot(bit_not) => match bit_not.value() {
                                    Value::U8(byte) => (builder.ins().iconst(I8, !byte), I8),
                                    Value::U16(int) => {
                                        (builder.ins().iconst(I16, int.not().0 as i64), I16)
                                    }
                                    Value::TapePtr(uint) => (builder.ins().iconst(I32, !uint), I32),
                                    Value::Bool(bool) => (builder.ins().bconst(B1, !bool), B1),
                                    Value::Val(value, _ty) => {
                                        let (value, ty) = values[&value];
                                        (builder.ins().bnot(value), ty)
                                    }
                                },

                                RValue::Add(add) => {
                                    let lhs = match add.lhs() {
                                        Value::U8(byte) => builder.ins().iconst(I64, byte),
                                        Value::U16(int) => builder.ins().iconst(I64, int.0 as i64),
                                        Value::TapePtr(uint) => builder.ins().iconst(I64, uint),
                                        Value::Bool(_) => unreachable!(),
                                        Value::Val(value, _ty) => {
                                            let (value, ty) = values[&value];
                                            if ty == I64 {
                                                value
                                            } else if ty == I8 || ty == I32 {
                                                builder.ins().uextend(I64, value)
                                            } else {
                                                panic!("{}", ty)
                                            }
                                        }
                                    };
                                    let rhs = match add.rhs() {
                                        Value::U8(byte) => builder.ins().iconst(I64, byte),
                                        Value::U16(int) => builder.ins().iconst(I64, int.0 as i64),
                                        Value::TapePtr(uint) => builder.ins().iconst(I64, uint),
                                        Value::Bool(_) => unreachable!(),
                                        Value::Val(value, _ty) => {
                                            let (value, ty) = values[&value];
                                            if ty == I64 {
                                                value
                                            } else if ty == I8 || ty == I32 {
                                                builder.ins().uextend(I64, value)
                                            } else {
                                                panic!("{}", ty)
                                            }
                                        }
                                    };

                                    // TODO: Optimize to `.iadd_imm()` when either side is an immediate value
                                    (builder.ins().iadd(lhs, rhs), I64)
                                }

                                RValue::Sub(sub) => {
                                    let lhs = match sub.lhs() {
                                        Value::U8(byte) => builder.ins().iconst(I64, byte),
                                        Value::U16(int) => builder.ins().iconst(I64, int.0 as i64),
                                        Value::TapePtr(uint) => builder.ins().iconst(I32, uint),
                                        Value::Bool(_) => unreachable!(),
                                        Value::Val(value, _ty) => {
                                            let (value, ty) = values[&value];
                                            if ty == I64 {
                                                value
                                            } else if ty == I8 || ty == I32 {
                                                builder.ins().uextend(I64, value)
                                            } else {
                                                panic!("{}", ty)
                                            }
                                        }
                                    };
                                    let rhs = match sub.rhs() {
                                        Value::U8(byte) => builder.ins().iconst(I64, byte),
                                        Value::U16(int) => builder.ins().iconst(I64, int.0 as i64),
                                        Value::TapePtr(uint) => builder.ins().iconst(I64, uint),
                                        Value::Bool(_) => unreachable!(),
                                        Value::Val(value, _ty) => {
                                            let (value, ty) = values[&value];
                                            if ty == I64 {
                                                value
                                            } else if ty == I8 || ty == I32 {
                                                builder.ins().uextend(I64, value)
                                            } else {
                                                panic!("{}", ty)
                                            }
                                        }
                                    };

                                    // TODO: Optimize to `.isub_imm()` when either side is an immediate value
                                    (builder.ins().isub(lhs, rhs), I64)
                                }

                                RValue::Mul(mul) => {
                                    let lhs = match mul.lhs() {
                                        Value::U8(byte) => builder.ins().iconst(I64, byte),
                                        Value::U16(int) => builder.ins().iconst(I64, int.0 as i64),
                                        Value::TapePtr(uint) => builder.ins().iconst(I64, uint),
                                        Value::Bool(_) => unreachable!(),
                                        Value::Val(value, _ty) => {
                                            let (value, ty) = values[&value];
                                            if ty == I64 {
                                                value
                                            } else if ty == I8 || ty == I32 {
                                                builder.ins().uextend(I64, value)
                                            } else {
                                                panic!("{}", ty)
                                            }
                                        }
                                    };
                                    let rhs = match mul.rhs() {
                                        Value::U8(byte) => builder.ins().iconst(I64, byte),
                                        Value::U16(int) => builder.ins().iconst(I64, int.0 as i64),
                                        Value::TapePtr(uint) => builder.ins().iconst(I64, uint),
                                        Value::Bool(_) => unreachable!(),
                                        Value::Val(value, _ty) => {
                                            let (value, ty) = values[&value];
                                            if ty == I64 {
                                                value
                                            } else if ty == I8 || ty == I32 {
                                                builder.ins().uextend(I64, value)
                                            } else {
                                                panic!("{}", ty)
                                            }
                                        }
                                    };

                                    // TODO: Optimize to `.imul_imm()` when either side is an immediate value
                                    (builder.ins().imul(lhs, rhs), I64)
                                }

                                // TODO: Bounds checking & wrapping on pointers
                                RValue::Load(load) => {
                                    // Get the value to store
                                    let offset = match load.ptr() {
                                        Value::U8(byte) => builder.ins().iconst(I64, byte),
                                        Value::U16(int) => builder
                                            .ins()
                                            .iconst(I64, Ptr::new(int.0, self.tape_len)),
                                        Value::TapePtr(uint) => builder.ins().iconst(I64, uint),
                                        Value::Val(offset, _ty) => {
                                            let (value, ty) = values[&offset];
                                            if ty == I64 {
                                                value
                                            } else if ty == I8 || ty == I32 {
                                                builder.ins().uextend(I64, value)
                                            } else {
                                                panic!("{}", ty)
                                            }
                                        }
                                        Value::Bool(_) => unreachable!(),
                                    };
                                    let pointer = builder.ins().iadd(tape_start, offset);

                                    // TODO: Optimize this to use a constant offset instead of add when possible
                                    (builder.ins().load(I8, MemFlags::new(), pointer, 0), I8)
                                }

                                RValue::Input(_) => {
                                    // Create a block to house the stuff that happens after the function
                                    // call and associated error check
                                    let call_prelude = builder.create_block();

                                    // Call the input function
                                    let input_call = builder.ins().call(input_function, &[state]);
                                    let input_result = builder.inst_results(input_call)[0];

                                    // Shift the input value left by 8 bits to get the error boolean
                                    let input_failed = builder.ins().ireduce(I8, input_result);

                                    // If the call to input failed (and therefore returned true),
                                    // branch to the io error handler
                                    builder.ins().brnz(input_failed, io_error_handler, &[]);
                                    io_handler_used = true;

                                    // Otherwise if the call didn't fail, so we use the return value from the
                                    // call to the input function by discarding the high 8 bits that used
                                    // to contain the error status of the function call
                                    builder.ins().jump(call_prelude, &[]);
                                    builder.switch_to_block(call_prelude);

                                    let input_result = builder.ins().ushr_imm(input_result, 8);
                                    (builder.ins().ireduce(I8, input_result), I8)
                                }
                            };

                            values.insert(assign.value(), value).debug_unwrap_none();
                        }

                        Instruction::Output(output) => {
                            // Create a block to house the stuff that happens after the function
                            // call and associated error check
                            let call_prelude = builder.create_block();

                            // Get the argument to the function
                            let value = match output.value() {
                                Value::U8(byte) => builder.ins().iconst(I8, byte),
                                Value::U16(int) => builder.ins().iconst(I8, int.0 as i64),
                                Value::TapePtr(uint) => builder.ins().iconst(I8, uint),
                                Value::Bool(bool) => builder.ins().iconst(I8, bool as i64),
                                Value::Val(value, _ty) => {
                                    let (value, ty) = values[&value];
                                    if ty != I8 {
                                        builder.ins().ireduce(I8, value)
                                    } else {
                                        value
                                    }
                                }
                            };

                            // Call the output function
                            let output_call = builder.ins().call(output_function, &[state, value]);
                            let output_result = builder.inst_results(output_call)[0];

                            // If the call to output failed (and therefore returned true),
                            // branch to the io error handler
                            builder.ins().brnz(output_result, io_error_handler, &[]);
                            io_handler_used = true;

                            // Otherwise if the call didn't fail, we don't have any work to do
                            builder.ins().jump(call_prelude, &[]);
                            builder.switch_to_block(call_prelude);
                        }
                    }
                }

                match block.terminator() {
                    // FIXME: Block arguments
                    Terminator::Jump(target) => {
                        let params: Vec<_> = {
                            let idx = blocks
                                .binary_search_by_key(target, |block| block.id())
                                .unwrap();

                            blocks[idx]
                                .instructions()
                                .iter()
                                .filter_map(|inst| {
                                    inst.as_assign()
                                        .and_then(|assign| assign.rval().as_phi())
                                        .and_then(|phi| {
                                            if phi.lhs_src() == block.id() {
                                                Some(phi.lhs())
                                            } else if phi.rhs_src() == block.id() {
                                                Some(phi.rhs())
                                            } else {
                                                tracing::warn!(
                                                    ?inst,
                                                    "found phi node without the jump source {} as a source: {}",
                                                    block.id(),
                                                    inst.pretty_print(PrettyConfig::minimal()),
                                                );

                                                None
                                            }
                                        })
                                        .and_then(|value| match value {
                                            Value::U8(byte) => {
                                                Some(builder.ins().iconst(I8, byte))
                                            }
                                            Value::U16(int) => Some(builder.ins().iconst(I8, int.0 as i64)),
                                            Value::TapePtr(uint) => {
                                                Some(builder.ins().iconst(I32, uint))
                                            }
                                            Value::Bool(bool) => Some(builder.ins().bconst(B1, bool)),
                                            // FIXME: Sometimes the value doesn't exist???
                                            Value::Val(value, _ty) => values.get(&value).map(|&(value, _)| value),
                                        })
                                })
                                .collect()
                        };

                        builder.ins().jump(ssa_blocks[target], &params);
                    }

                    // FIXME: Block arguments
                    Terminator::Branch(branch) => {
                        let cond = match branch.condition() {
                            Value::U8(byte) => builder.ins().iconst(I8, byte),
                            Value::U16(int) => builder.ins().iconst(I16, int.0 as i64),
                            Value::TapePtr(uint) => builder.ins().iconst(I32, uint),
                            Value::Bool(bool) => builder.ins().bconst(B1, bool),
                            Value::Val(value, _ty) => values[&value].0,
                        };

                        let true_params: Vec<_> = {
                            let idx = blocks
                                .binary_search_by_key(&branch.true_jump(), |block| block.id())
                                .unwrap();

                            blocks[idx]
                                .instructions()
                                .iter()
                                .filter_map(|inst| {
                                    inst.as_assign()
                                        .and_then(|assign| assign.rval().as_phi())
                                        .and_then(|phi| {
                                            if phi.lhs_src() == block.id() {
                                                Some(phi.lhs())
                                            } else if phi.rhs_src() == block.id() {
                                                Some(phi.rhs())
                                            } else {
                                                tracing::warn!(
                                                    ?inst,
                                                    "found phi node without the jump source {} as a source: {}",
                                                    block.id(),
                                                    inst.pretty_print(PrettyConfig::minimal()),
                                                );

                                                None
                                            }
                                        })
                                        .map(|value| match value {
                                            Value::U8(byte) => {
                                                builder.ins().iconst(I8, byte)
                                            }
                                            Value::U16(int) => builder.ins().iconst(I16, int.0 as i64),
                                            Value::TapePtr(uint) => {
                                                builder.ins().iconst(I32, uint)
                                            }
                                            Value::Bool(bool) => builder.ins().bconst(B1, bool),
                                            Value::Val(value, _ty) => values[&value].0,
                                        })
                                })
                                .collect()
                        };

                        // Jump to the true branch if the condition is true (or non-zero)
                        builder
                            .ins()
                            .brnz(cond, ssa_blocks[&branch.true_jump()], &true_params);

                        let false_params: Vec<_> = {
                            let idx = blocks
                                .binary_search_by_key(&branch.false_jump(), |block| block.id())
                                .unwrap();

                            blocks[idx]
                                .instructions()
                                .iter()
                                .filter_map(|inst| {
                                    inst.as_assign()
                                        .and_then(|assign| assign.rval().as_phi())
                                        .and_then(|phi| {
                                            if phi.lhs_src() == block.id() {
                                                Some(phi.lhs())
                                            } else if phi.rhs_src() == block.id() {
                                                Some(phi.rhs())
                                            } else {
                                                tracing::warn!(
                                                    ?inst,
                                                    "found phi node without the jump source {} as a source: {}",
                                                    block.id(),
                                                    inst.pretty_print(PrettyConfig::minimal()),
                                                );

                                                None
                                            }
                                        })
                                        .map(|value| match value {
                                            Value::U8(byte) => {
                                                builder.ins().iconst(I8, byte)
                                            }
                                        Value::U16(int) => builder.ins().iconst(I16, int.0 as i64),
                                            Value::TapePtr(uint) => {
                                                builder.ins().iconst(I32, uint)
                                            }
                                            Value::Bool(bool) => builder.ins().bconst(B1, bool),
                                            Value::Val(value, _ty) => values[&value].0,
                                        })
                                })
                                .collect()
                        };

                        // Otherwise jump to the false branch
                        builder
                            .ins()
                            .jump(ssa_blocks[&branch.false_jump()], &false_params);
                    }

                    &Terminator::Return(value) => {
                        let value = match value {
                            Value::U8(byte) => builder.ins().iconst(I8, byte),
                            Value::U16(int) => builder.ins().iconst(I8, int.0 as i64),
                            Value::TapePtr(uint) => builder.ins().iconst(I8, uint),
                            Value::Bool(bool) => builder.ins().iconst(I8, bool as i64),
                            Value::Val(value, _ty) => {
                                let (value, ty) = values[&value];
                                if ty != I8 {
                                    builder.ins().ireduce(I8, value)
                                } else {
                                    value
                                }
                            }
                        };

                        builder.ins().return_(&[value]);
                    }

                    Terminator::Unreachable => {
                        builder.ins().trap(TrapCode::UnreachableCodeReached);
                    }

                    Terminator::Error => unreachable!(),
                }
            }

            // If the io handler block was ever used, construct it
            if io_handler_used {
                builder.switch_to_block(io_error_handler);

                // Call the io error reporting function
                builder.ins().call(io_error_function, &[state]);

                // Return with an error code
                let return_code = builder.ins().iconst(I8, RETURN_IO_FAILURE);
                builder.ins().return_(&[return_code]);

                // Mark this block as cold since it's only for error handling
                builder.set_cold_block(io_error_handler);
                builder.seal_block(io_error_handler);
            }

            // Seal all basic blocks
            // FIXME: We should seal blocks as it becomes possible to since it's more efficient
            builder.seal_all_blocks();

            tracing::debug!("generated cranelift ir: {}", builder.func);

            // Finalize the generated code
            builder.finalize();

            // Create a formatted version of the ssa ir we generated
            blocks.pretty_print(PrettyConfig::minimal())
        };

        // Next, declare the function to jit. Functions must be declared
        // before they can be called, or defined
        let function_id =
            self.module
                .declare_function("jit_entry", Linkage::Export, &self.ctx.func.signature)?;

        // Define the function within the jit
        let code_len = self.module.define_function(function_id, &mut self.ctx)?;

        // Create a formatted version of the cranelift ir we generated
        let clif_ir = format!("{}", self.ctx.func);

        // Now that compilation is finished, we can clear out the context state.
        self.module.clear_context(&mut self.ctx);

        // Finalize the functions which we just defined, which resolves any
        // outstanding relocations (patching in addresses, now that they're
        // available).
        self.module.finalize_definitions();

        // We can now retrieve a pointer to the machine code.
        let code = self.module.get_finalized_function(function_id);

        // Disassemble the generated instructions
        let code_bytes = unsafe { slice::from_raw_parts(code, code_len.size as usize) };
        let disassembly = disassemble::disassemble(code_bytes);

        let function = unsafe { transmute::<*const u8, JitFunction>(code) };
        let executable = Executable::new(function, self.module);

        Ok((executable, clif_ir, ssa_ir, disassembly))
    }
}
