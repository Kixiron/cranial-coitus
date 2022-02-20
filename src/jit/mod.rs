#[macro_use]
mod ffi;
mod basic_block;
mod block_builder;
mod block_visitor;
// mod cir_jit;
mod cir_to_bb;
// mod coloring;
// mod disassemble;
// mod liveliness;
mod memory;
// mod regalloc;

pub use ffi::State;
pub use memory::Executable;

use crate::{
    ir::Block,
    jit::basic_block::{Instruction, RValue, Terminator, Value},
    utils::AssertNone,
};
use anyhow::Result;
use cranelift::{
    codegen::{
        ir::{
            types::{B1, B8, I16, I32, I64, I8},
            InstBuilder,
        },
        Context,
    },
    frontend::{FunctionBuilder, FunctionBuilderContext},
    prelude::{isa::CallConv, AbiParam, IntCC, MemFlags, TrapCode},
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataContext, Linkage, Module};
use std::{collections::BTreeMap, mem::transmute};

/// The function produced by jit code
pub type JitFunction = unsafe extern "fastcall" fn(*mut State, *mut u8, *const u8) -> u8;

const RETURN_SUCCESS: i64 = 0;

const RETURN_IO_FAILURE: i64 = 101;

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
    tape_len: usize,
}

impl Jit {
    /// Create a new jit
    pub fn new(tape_len: usize) -> Self {
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

        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            data_ctx: DataContext::new(),
            module,
            tape_len,
        }
    }

    /// Compile a block of CIR into an executable buffer
    pub fn compile(&mut self, block: &Block) -> Result<(Executable, String)> {
        {
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
                                Value::Byte(byte) => builder.ins().iconst(I8, byte as i64),
                                Value::Uint(uint) => builder.ins().iconst(I8, uint as i64),
                                Value::Val(value) => {
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
                                Value::Byte(byte) => {
                                    // Wrap the offset to the tape's address space
                                    let offset = (byte as usize).rem_euclid(self.tape_len);

                                    builder.ins().store(
                                        MemFlags::new(),
                                        value,
                                        tape_start,
                                        offset as i32,
                                    );
                                }
                                Value::Uint(uint) => {
                                    // Wrap the offset to the tape's address space
                                    let offset = (uint as usize).rem_euclid(self.tape_len);

                                    builder.ins().store(
                                        MemFlags::new(),
                                        value,
                                        tape_start,
                                        offset as i32,
                                    );
                                }

                                // TODO: Bounds checking & wrapping on pointers
                                Value::Val(offset) => {
                                    let offset = {
                                        let (value, ty) = values[&offset];
                                        if ty != I64 {
                                            builder.ins().bitcast(I64, value)
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
                                        Value::Byte(byte) => builder.ins().iconst(I8, byte as i64),
                                        Value::Uint(uint) => builder.ins().iconst(I8, uint as i64),
                                        Value::Bool(bool) => builder.ins().iconst(I8, bool as i64),
                                        Value::Val(value) => {
                                            let (value, ty) = values[&value];
                                            if ty != I8 {
                                                builder.ins().bitcast(I8, value)
                                            } else {
                                                value
                                            }
                                        }
                                    };
                                    let rhs = match eq.rhs() {
                                        Value::Byte(byte) => builder.ins().iconst(I8, byte as i64),
                                        Value::Uint(uint) => builder.ins().iconst(I8, uint as i64),
                                        Value::Bool(bool) => builder.ins().iconst(I8, bool as i64),
                                        Value::Val(value) => {
                                            let (value, ty) = values[&value];
                                            if ty != I8 {
                                                builder.ins().bitcast(I8, value)
                                            } else {
                                                value
                                            }
                                        }
                                    };

                                    // TODO: Optimize to `.icmp_imm()` when either side is an immediate value
                                    (builder.ins().icmp(IntCC::Equal, lhs, rhs), B1)
                                }

                                RValue::Phi(_) => todo!(),
                                RValue::Neg(_) => todo!(),
                                RValue::Not(_) => todo!(),
                                RValue::BitNot(_) => todo!(),

                                RValue::Add(add) => {
                                    let lhs = match add.lhs() {
                                        Value::Byte(byte) => builder.ins().iconst(I64, byte as i64),
                                        Value::Uint(uint) => builder.ins().iconst(I64, uint as i64),
                                        Value::Bool(bool) => builder.ins().iconst(I64, bool as i64),
                                        Value::Val(value) => {
                                            let (value, ty) = values[&value];
                                            if ty == I64 {
                                                value
                                            } else if ty == I8 {
                                                builder.ins().uextend(I64, value)
                                            } else {
                                                panic!("{}", ty)
                                            }
                                        }
                                    };
                                    let rhs = match add.rhs() {
                                        Value::Byte(byte) => builder.ins().iconst(I8, byte as i64),
                                        Value::Uint(uint) => builder.ins().iconst(I32, uint as i64),
                                        Value::Bool(bool) => builder.ins().iconst(I8, bool as i64),
                                        Value::Val(value) => {
                                            let (value, ty) = values[&value];
                                            if ty == I64 {
                                                value
                                            } else if ty == I8 {
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
                                        Value::Byte(byte) => builder.ins().iconst(I8, byte as i64),
                                        Value::Uint(uint) => builder.ins().iconst(I32, uint as i64),
                                        Value::Bool(bool) => builder.ins().iconst(I8, bool as i64),
                                        Value::Val(value) => {
                                            let (value, ty) = values[&value];
                                            if ty == I64 {
                                                value
                                            } else if ty == I8 {
                                                builder.ins().uextend(I64, value)
                                            } else {
                                                panic!("{}", ty)
                                            }
                                        }
                                    };
                                    let rhs = match sub.rhs() {
                                        Value::Byte(byte) => builder.ins().iconst(I8, byte as i64),
                                        Value::Uint(uint) => builder.ins().iconst(I32, uint as i64),
                                        Value::Bool(bool) => builder.ins().iconst(I8, bool as i64),
                                        Value::Val(value) => {
                                            let (value, ty) = values[&value];
                                            if ty == I64 {
                                                value
                                            } else if ty == I8 {
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
                                        Value::Byte(byte) => builder.ins().iconst(I64, byte as i64),
                                        Value::Uint(uint) => builder.ins().iconst(I64, uint as i64),
                                        Value::Bool(bool) => builder.ins().iconst(I64, bool as i64),
                                        Value::Val(value) => {
                                            let (value, ty) = values[&value];
                                            if ty == I64 {
                                                value
                                            } else if ty == I8 {
                                                builder.ins().uextend(I64, value)
                                            } else {
                                                panic!("{}", ty)
                                            }
                                        }
                                    };
                                    let rhs = match mul.rhs() {
                                        Value::Byte(byte) => builder.ins().iconst(I64, byte as i64),
                                        Value::Uint(uint) => builder.ins().iconst(I64, uint as i64),
                                        Value::Bool(bool) => builder.ins().iconst(I64, bool as i64),
                                        Value::Val(value) => {
                                            let (value, ty) = values[&value];
                                            if ty == I64 {
                                                value
                                            } else if ty == I8 {
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
                                        Value::Byte(byte) => builder.ins().iconst(I64, byte as i64),
                                        Value::Uint(uint) => builder.ins().iconst(I64, uint as i64),
                                        Value::Val(offset) => {
                                            let (value, ty) = values[&offset];
                                            if ty == I64 {
                                                value
                                            } else if ty == I8 {
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
                                Value::Byte(byte) => builder.ins().iconst(I8, byte as i64),
                                Value::Uint(uint) => builder.ins().iconst(I8, uint as i64),
                                Value::Bool(bool) => builder.ins().iconst(I8, bool as i64),
                                Value::Val(value) => {
                                    let (value, ty) = values[&value];
                                    if ty != I8 {
                                        builder.ins().bitcast(I8, value)
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
                        builder.ins().jump(ssa_blocks[target], &[]);
                    }

                    // FIXME: Block arguments
                    Terminator::Branch(branch) => {
                        let cond = match branch.condition() {
                            Value::Byte(byte) => builder.ins().iconst(I8, byte as i64),
                            Value::Uint(uint) => builder.ins().iconst(I32, uint as i64),
                            Value::Bool(bool) => builder.ins().bconst(B1, bool),
                            Value::Val(value) => values[&value].0,
                        };

                        // Jump to the true branch if the condition is true (or non-zero)
                        builder
                            .ins()
                            .brnz(cond, ssa_blocks[&branch.true_jump()], &[]);

                        // Otherwise jump to the false branch
                        builder.ins().jump(ssa_blocks[&branch.false_jump()], &[]);
                    }

                    &Terminator::Return(value) => {
                        let value = match value {
                            Value::Byte(byte) => builder.ins().iconst(I8, byte as i64),
                            Value::Uint(uint) => builder.ins().iconst(I8, uint as i64),
                            Value::Bool(bool) => builder.ins().iconst(I8, bool as i64),
                            Value::Val(value) => {
                                let (value, ty) = values[&value];
                                if ty != I8 {
                                    builder.ins().bitcast(I8, value)
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

            // Finalize the generated code
            builder.finalize();
        }

        // Next, declare the function to jit. Functions must be declared
        // before they can be called, or defined
        let function_id =
            self.module
                .declare_function("jit_entry", Linkage::Export, &self.ctx.func.signature)?;

        // Define the function to jit. This finishes compilation, although
        // there may be outstanding relocations to perform. Currently, jit
        // cannot finish relocations until all functions to be called are
        // defined. For this toy demo for now, we'll just finalize the
        // function below.
        self.module.define_function(function_id, &mut self.ctx)?;

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
        let function = Executable::new(unsafe { transmute::<*const u8, JitFunction>(code) });

        // TODO: Disassembly of jit code

        Ok((function, clif_ir))
    }
}

/*
pub use memory::{CodeBuffer, Executable};

use crate::{
    ir::{Block, Pretty, PrettyConfig},
    jit::{
        basic_block::{
            Add, Assign, BasicBlock, BlockId, Blocks, Input, Instruction, Load, Output, RValue,
            Store, Terminator, ValId, Value,
        },
        ffi::State,
        liveliness::{BlockLifetime, Liveliness, LivelinessAnalysis, Location},
        regalloc::{Regalloc, StackSlot},
    },
    utils::AssertNone,
};
use iced_x86::code_asm::{CodeAssembler, *};
use std::{borrow::Cow, collections::BTreeMap};

type AsmResult<T> = Result<T, IcedError>;

const BITNESS: u32 = 64;


// https://docs.microsoft.com/en-us/cpp/build/x64-calling-convention?view=msvc-170#callercallee-saved-registers
// Note: Cannot preserve rsp here since it makes things go bananas
const CALLEE_SAVED_REGISTERS: &[AsmRegister64] = &[rbx, rbp, rdi, rsi, r12, r13, r14, r15];
const CALLEE_REGISTER_OFFSET: usize = CALLEE_SAVED_REGISTERS.len() * 8;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Operand {
    Byte(u8),
    Uint(u32),
    Bool(bool),
    Stack(StackSlot),
    Register(AsmRegister64),
}

impl Operand {
    #[allow(dead_code)]
    pub fn is_zero(&self) -> bool {
        match *self {
            Self::Byte(byte) => byte == 0,
            Self::Uint(uint) => uint == 0,
            Self::Bool(_) => todo!(),
            Self::Stack(_) | Self::Register(_) => false,
        }
    }

    pub fn map_stack<F>(&self, map: F) -> Self
    where
        F: FnOnce(StackSlot) -> StackSlot,
    {
        match *self {
            Self::Stack(addr) => Self::Stack(map(addr)),
            operand => operand,
        }
    }
}

pub struct Jit {
    asm: CodeAssembler,
    epilogue: CodeLabel,
    io_failure: CodeLabel,
    has_io_functions: bool,
    values: BTreeMap<ValId, Operand>,
    regalloc: Regalloc,
    comments: BTreeMap<usize, Vec<String>>,
    named_labels: BTreeMap<usize, Cow<'static, str>>,
    block_labels: BTreeMap<BlockId, CodeLabel>,
    // TODO: Control flow graph + liveliness gives us the ability to
    //       get the current set of live variables at any given point
    liveliness: Liveliness,
    current_location: Location,
    phi_destinations: BTreeMap<ValId, AsmRegister64>,
}

impl Jit {
    pub fn new(_tape_len: usize) -> AsmResult<Self> {
        let mut asm = CodeAssembler::new(BITNESS)?;
        let epilogue = asm.create_label();
        let io_failure = asm.create_label();

        Ok(Self {
            asm,
            epilogue,
            io_failure,
            has_io_functions: false,
            values: BTreeMap::new(),
            regalloc: Regalloc::new(),
            comments: BTreeMap::new(),
            named_labels: BTreeMap::new(),
            block_labels: BTreeMap::new(),
            liveliness: Liveliness::new(),
            current_location: Location::new(BlockId::new(u32::MAX), None),
            phi_destinations: BTreeMap::new(),
        })
    }

    #[tracing::instrument(skip_all)]
    pub fn assemble(&mut self, block: &Block) -> AsmResult<(Executable<CodeBuffer>, String)> {
        let blocks = cir_to_bb::translate(block);
        println!(
            "SSA form:\n{}",
            blocks.pretty_print(PrettyConfig::minimal()),
        );

        self.liveliness = LivelinessAnalysis::new().run(&blocks);
        println!("liveliness: {:#?}", self.liveliness);

        self.prologue()?;

        self.named_label(".BODY");

        self.create_block_labels(&blocks);
        for (idx, block) in blocks.iter().enumerate() {
            *self.current_location.block_mut() = block.id();

            let next_block = blocks.get(idx + 1).map(BasicBlock::id);
            self.assemble_block(block, next_block)?;
        }

        // Only build the IO handler if there's IO functions
        if self.has_io_functions {
            self.build_io_failure()?;
        }

        self.epilogue()?;

        let code_buffer = {
            // The maximum size of an instruction is 15 bytes, so we (virtually) allocate the most memory we could possibly use
            let maximum_possible_size = self.asm.instructions().len() * 15;
            let mut code_buffer = CodeBuffer::new(maximum_possible_size).unwrap();

            let code = self.asm.assemble(code_buffer.as_ptr() as u64)?;

            debug_assert!(code_buffer.len() <= code.capacity());
            code_buffer.copy_from_slice(&code).unwrap();

            code_buffer.executable().unwrap()
        };
        let pretty = self.disassemble();

        Ok((code_buffer, pretty))
    }

    fn assemble_block(&mut self, block: &BasicBlock, next_block: Option<BlockId>) -> AsmResult<()> {
        let label = *self.block_labels.get(&block.id()).unwrap();
        self.set_label(label);

        self.add_comment(format!("start {}", block.id()));
        // self.asm.nop()?;

        for (&value, lifetime) in &self.liveliness.lifetimes {
            if let Ok(idx) = lifetime
                .blocks
                .binary_search_by_key(&block.id(), |&(block, _)| block)
            {
                match lifetime.blocks[idx].1 {
                    BlockLifetime::Whole => {
                        if !self.values.contains_key(&value) {
                            let dest = self.allocate_register()?;
                            self.values.insert(value, Operand::Register(dest));
                        }
                    }
                    BlockLifetime::Span(..) => todo!(),
                }
            }
        }

        for (idx, inst) in block.iter().enumerate() {
            *self.current_location.inst_mut() = Some(idx);

            // // Allocate space for all declared variables
            // for decl in self.liveliness.declarations_at(self.current_location) {
            //     let dest = self.allocate_register()?;
            //     self.values
            //         .insert(decl, Operand::Register(dest))
            //         .debug_unwrap_none();
            // }

            self.assemble_inst(inst)?;
        }

        *self.current_location.inst_mut() = None;
        self.assemble_terminator(block.terminator(), next_block)?;

        Ok(())
    }

    fn assemble_inst(&mut self, inst: &Instruction) -> AsmResult<()> {
        self.add_comment(inst.pretty_print(PrettyConfig::minimal()));

        match inst {
            Instruction::Store(store) => self.assemble_store(store),
            Instruction::Assign(assign) => self.assemble_assign(assign),
            Instruction::Output(output) => self.assemble_output(output),
        }
    }

    #[allow(dead_code)]
    fn add(&mut self, lhs: AsmRegister64, rhs: i32) -> AsmResult<()> {
        if rhs == 0 {
            Ok(())
        } else if rhs == 1 {
            self.asm.inc(lhs)
        } else {
            self.asm.add(lhs, rhs)
        }
    }

    fn assemble_store(&mut self, store: &Store) -> AsmResult<()> {
        let ptr = match self.get_value(store.ptr()) {
            Operand::Byte(byte) => Operand::Uint(byte as u32),
            Operand::Bool(bool) => Operand::Uint(bool as u32),
            operand => operand,
        };
        let value = match self.get_value(store.value()) {
            Operand::Byte(byte) => Operand::Uint(byte as u32),
            Operand::Bool(bool) => Operand::Uint(bool as u32),
            operand => operand,
        };

        match (ptr, value) {
            (Operand::Uint(ptr), Operand::Uint(value)) => {
                let temp = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp, tape_ptr)?;

                self.asm.mov(byte_ptr(temp + ptr), value)?;

                self.deallocate_register(temp);
            }

            (Operand::Uint(ptr), Operand::Stack(value)) => {
                let temp1 = self.allocate_register()?;
                let temp2 = self.allocate_register()?;

                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp1, tape_ptr)?;

                let value = self.stack_offset(value);
                self.asm.mov(temp2, value)?;

                self.asm.mov(byte_ptr(temp1 + ptr), temp2)?;

                self.deallocate_register(temp2);
                self.deallocate_register(temp1);
            }

            (Operand::Uint(ptr), Operand::Register(value)) => {
                let temp = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp, tape_ptr)?;

                self.asm.mov(byte_ptr(temp + ptr), value)?;

                self.deallocate_register(temp);
            }

            (Operand::Stack(ptr), Operand::Uint(value)) => {
                let temp = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp, tape_ptr)?;

                let ptr = self.stack_offset(ptr);
                self.asm.add(temp, ptr)?;
                self.asm.mov(byte_ptr(temp), value)?;

                self.deallocate_register(temp);
            }
            (Operand::Stack(ptr), Operand::Stack(value)) => {
                let temp1 = self.allocate_register()?;
                let temp2 = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp1, tape_ptr)?;

                let (ptr, value) = (self.stack_offset(ptr), self.stack_offset(value));
                self.asm.add(temp1, ptr)?;
                self.asm.mov(temp2, value)?;
                self.asm.mov(byte_ptr(temp1), temp2)?;

                self.deallocate_register(temp1);
                self.deallocate_register(temp2);
            }
            (Operand::Stack(ptr), Operand::Register(value)) => {
                let temp = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp, tape_ptr)?;

                let ptr = self.stack_offset(ptr);
                self.asm.add(temp, ptr)?;
                self.asm.mov(byte_ptr(temp), value)?;

                self.deallocate_register(temp);
            }

            (Operand::Register(ptr), Operand::Uint(value)) => {
                let temp = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp, tape_ptr)?;

                self.asm.add(temp, ptr)?;
                self.asm.mov(byte_ptr(temp), value)?;

                self.deallocate_register(temp);
            }
            (Operand::Register(ptr), Operand::Stack(value)) => {
                let temp1 = self.allocate_register()?;
                let temp2 = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp1, tape_ptr)?;

                self.asm.add(temp1, ptr)?;
                let value = self.stack_offset(value);
                self.asm.mov(temp2, value)?;
                self.asm.mov(byte_ptr(temp1), temp2)?;

                self.deallocate_register(temp1);
                self.deallocate_register(temp2);
            }
            (Operand::Register(ptr), Operand::Register(value)) => {
                let temp = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp, tape_ptr)?;

                self.asm.add(temp, ptr)?;
                self.asm.mov(byte_ptr(temp), value)?;

                self.deallocate_register(temp);
            }

            // We removed Byte and Bool from the possible operand types
            (Operand::Byte(_) | Operand::Bool(_), _) | (_, Operand::Byte(_) | Operand::Bool(_)) => {
                unreachable!()
            }
        }

        Ok(())
    }

    fn assemble_assign(&mut self, assign: &Assign) -> AsmResult<()> {
        let dest = match self.get_var(assign.value()) {
            Operand::Register(dest) => dest,
            Operand::Stack(slot) => {
                let dest = self.allocate_register()?;
                self.asm.mov(dest, self.stack_offset(slot))?;
                dest
            }
            _ => unreachable!(),
        };

        match assign.rval() {
            RValue::Eq(eq) => {
                let (lhs, rhs) = (self.get_value(eq.lhs()), self.get_value(eq.rhs()));

                match (lhs, rhs) {
                    (Operand::Byte(lhs), Operand::Byte(rhs)) => {
                        self.asm.mov(dest, lhs as i64)?;
                        self.asm.cmp(dest, rhs as i32)?;
                    }
                    (Operand::Byte(lhs), Operand::Uint(rhs)) => {
                        self.asm.mov(dest, lhs as i64)?;
                        self.asm.cmp(dest, rhs as i32)?;
                    }
                    (Operand::Byte(lhs), Operand::Bool(rhs)) => {
                        self.asm.mov(dest, lhs as i64)?;
                        self.asm.cmp(dest, rhs as i32)?;
                    }
                    (Operand::Byte(lhs), Operand::Stack(rhs)) => {
                        let rhs = self.stack_offset(rhs);
                        self.asm.mov(dest, lhs as i64)?;
                        self.asm.cmp(dest, rhs)?;
                    }
                    (Operand::Byte(lhs), Operand::Register(rhs)) => {
                        self.asm.mov(dest, lhs as i64)?;
                        self.asm.cmp(dest, rhs)?;
                    }

                    (Operand::Uint(_), Operand::Byte(_)) => todo!(),
                    (Operand::Uint(_), Operand::Uint(_)) => todo!(),
                    (Operand::Uint(_), Operand::Bool(_)) => todo!(),
                    (Operand::Uint(_), Operand::Stack(_)) => todo!(),
                    (Operand::Uint(_), Operand::Register(_)) => todo!(),

                    (Operand::Bool(_), Operand::Byte(_)) => todo!(),
                    (Operand::Bool(_), Operand::Uint(_)) => todo!(),
                    (Operand::Bool(_), Operand::Bool(_)) => todo!(),
                    (Operand::Bool(_), Operand::Stack(_)) => todo!(),
                    (Operand::Bool(_), Operand::Register(_)) => todo!(),

                    (Operand::Stack(_), Operand::Byte(_)) => todo!(),
                    (Operand::Stack(_), Operand::Uint(_)) => todo!(),
                    (Operand::Stack(_), Operand::Bool(_)) => todo!(),
                    (Operand::Stack(_), Operand::Stack(_)) => todo!(),
                    (Operand::Stack(_), Operand::Register(_)) => todo!(),

                    (Operand::Register(lhs), Operand::Byte(rhs)) => {
                        self.asm.cmp(lhs, rhs as i32)?;
                    }
                    (Operand::Register(lhs), Operand::Uint(rhs)) => {
                        self.asm.cmp(lhs, rhs as i32)?;
                    }
                    (Operand::Register(lhs), Operand::Bool(rhs)) => {
                        self.asm.cmp(lhs, rhs as i32)?;
                    }
                    (Operand::Register(lhs), Operand::Stack(rhs)) => {
                        let rhs = self.stack_offset(rhs);
                        self.asm.cmp(lhs, rhs)?;
                    }
                    (Operand::Register(lhs), Operand::Register(rhs)) => self.asm.cmp(lhs, rhs)?,
                }

                // Set al to 1 if the operands are equal
                self.asm.setne(al)?;

                // Move the comparison result from al into the allocated
                // register with a zero sign extension
                self.asm.movzx(dest, al)?;
            }

            // RValue::Neg(_) => todo!(),
            RValue::Not(not) => {
                let value = self.get_value(not.value());
                // let dest = if let Value::Val(val) = not.value() {
                //     match value {
                //         Operand::Register(register)
                //             if self.liveliness.is_last_usage(val, self.current_location) =>
                //         {
                //             self.regalloc.allocate_overwrite(register);
                //             self.values.remove(&val);
                //             register
                //         }
                //
                //         _ => {
                //             let dest = self.allocate_register()?;
                //             self.move_to_reg(dest, value)?;
                //             dest
                //         }
                //     }
                // } else {
                //     let dest = self.allocate_register()?;
                //     self.move_to_reg(dest, value)?;
                //     dest
                // };

                // Perform a *logical* not on the destination value
                self.asm.xor(dest, 1)?;

                self.values
                    .insert(assign.value(), Operand::Register(dest))
                    .debug_unwrap_none();
            }

            RValue::Add(add) => self.assemble_add(dest, add, assign)?,

            RValue::Sub(sub) => {
                // let dest = self.allocate_register()?;
                let (lhs, rhs) = (self.get_value(sub.lhs()), self.get_value(sub.rhs()));

                self.move_to_reg(dest, lhs)?;
                match rhs {
                    Operand::Byte(byte) => self.asm.sub(dest, byte as i32)?,
                    Operand::Uint(uint) => self.asm.sub(dest, uint as i32)?,
                    Operand::Bool(bool) => self.asm.sub(dest, bool as i32)?,
                    Operand::Stack(slot) => {
                        let addr = self.stack_offset(slot);
                        self.asm.sub(dest, addr)?;
                    }
                    Operand::Register(reg) => self.asm.sub(dest, reg)?,
                }

                self.values
                    .insert(assign.value(), Operand::Register(dest))
                    .debug_unwrap_none();
            }

            // RValue::Mul(_) => todo!(),
            RValue::Load(load) => self.assemble_load(dest, load, assign)?,

            RValue::Input(input) => self.assemble_input(dest, input, assign)?,

            RValue::Phi(phi) => {}

            // RValue::BitNot(_) => todo!(),
            rvalue => todo!("{:?}", rvalue),
        }

        Ok(())
    }

    fn assemble_add(&mut self, dest: AsmRegister64, add: &Add, assign: &Assign) -> AsmResult<()> {
        let (lhs, rhs) = (self.get_value(add.lhs()), self.get_value(add.rhs()));

        self.move_to_reg(dest, lhs)?;

        match rhs {
            Operand::Byte(0) | Operand::Uint(0) => {}
            Operand::Byte(1) | Operand::Uint(1) => self.asm.inc(dest)?,

            Operand::Byte(byte) => self.asm.add(dest, byte as i32)?,
            Operand::Uint(uint) => self.asm.add(dest, uint as i32)?,

            Operand::Stack(slot) => {
                let addr = self.stack_offset(slot);
                self.asm.add(dest, addr)?;
            }

            Operand::Register(reg) => self.asm.add(dest, reg)?,

            Operand::Bool(_) => panic!("cannot add a boolean"),
        }

        // let mut dest = None;
        //
        // // Attempt to reuse the lhs's register if available
        // if let Operand::Register(reg) = lhs {
        //     if let Value::Val(val) = add.lhs() {
        //         if self.liveliness.is_last_usage(val, self.current_location) {
        //             self.regalloc.allocate_overwrite(reg);
        //             self.values.remove(&val);
        //             dest = Some((reg, rhs));
        //         }
        //     }
        // }
        //
        // // If we couldn't reuse the lhs's register, try to reuse the rhs's
        // if dest.is_none() {
        //     if let Operand::Register(reg) = rhs {
        //         if let Value::Val(val) = add.rhs() {
        //             if self.liveliness.is_last_usage(val, self.current_location) {
        //                 self.regalloc.allocate_overwrite(reg);
        //                 self.values.remove(&val);
        //                 dest = Some((reg, lhs));
        //             }
        //         }
        //     }
        // }
        //
        // // Currently `dest` contains one operand of the add and `operand` contains
        // // the other
        // let (dest, operand) = match dest {
        //     Some(dest) => dest,
        //     None => {
        //         let dest = self.allocate_register()?;
        //         self.move_to_reg(dest, lhs)?;
        //         (dest, rhs)
        //     }
        // };
        //
        // match operand {
        //     Operand::Byte(0) | Operand::Uint(0) => {}
        //     Operand::Byte(1) | Operand::Uint(1) => self.asm.inc(dest)?,
        //     Operand::Byte(byte) => self.asm.add(dest, byte as i32)?,
        //     Operand::Uint(uint) => self.asm.add(dest, uint as i32)?,
        //
        //     Operand::Bool(_) => panic!("cannot add a boolean"),
        //
        //     Operand::Stack(slot) => {
        //         let addr = self.stack_offset(slot);
        //         self.asm.add(dest, addr)?;
        //     }
        //
        //     Operand::Register(reg) => self.asm.add(dest, reg)?,
        // }
        //
        // self.values
        //     .insert(assign.value(), Operand::Register(dest))
        //     .debug_unwrap_none();

        Ok(())
    }

    fn assemble_load(
        &mut self,
        dest: AsmRegister64,
        load: &Load,
        assign: &Assign,
    ) -> AsmResult<()> {
        let ptr = self.get_value(load.ptr());
        let dest = match ptr {
            Operand::Byte(ptr) => {
                // let dest = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();

                self.asm.mov(dest, tape_ptr)?;
                self.asm
                    .mov(low_byte_register(dest), byte_ptr(dest + ptr as u32))?;

                dest
            }

            Operand::Uint(ptr) => {
                // let dest = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();

                self.asm.mov(dest, tape_ptr)?;
                self.asm
                    .mov(low_byte_register(dest), byte_ptr(dest + ptr as u32))?;

                dest
            }

            Operand::Bool(_) => panic!("cannot offset a pointer by a boolean"),

            Operand::Stack(slot) => {
                // let dest = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();
                let addr = self.stack_offset(slot);

                self.asm.mov(dest, tape_ptr)?;
                self.asm.add(dest, addr)?;
                self.asm.mov(low_byte_register(dest), byte_ptr(dest))?;

                dest
            }

            Operand::Register(reg) => {
                // let mut dest = None;
                //
                // // Attempt to reuse the register
                // if let Value::Val(val) = load.ptr() {
                //     if self.liveliness.is_last_usage(val, self.current_location) {
                //         self.regalloc.allocate_overwrite(reg);
                //         self.values.remove(&val);
                //         dest = Some(reg);
                //     }
                // }
                //
                // let dest = match dest {
                //     Some(dest) => dest,
                //     None => self.allocate_register()?,
                // };
                let tape_ptr = self.tape_start_ptr();

                self.asm.mov(dest, tape_ptr)?;
                self.asm
                    .mov(low_byte_register(dest), byte_ptr(dest + reg))?;

                dest
            }
        };

        // self.values
        //     .insert(assign.value(), Operand::Register(dest))
        //     .debug_unwrap_none();

        Ok(())
    }

    fn assemble_input(
        &mut self,
        input_reg: AsmRegister64,
        _input: &Input,
        assign: &Assign,
    ) -> AsmResult<()> {
        self.has_io_functions = true;

        // Move the state pointer into rcx
        self.asm.mov(rcx, self.state_ptr())?;

        // Setup the stack and save used registers
        let (stack_padding, slots) = self.before_win64_call()?;

        // Call the input function
        type_eq::<unsafe extern "win64" fn(state: *mut State) -> u16>(ffi::input);
        #[allow(clippy::fn_to_numeric_cast)]
        self.asm.mov(rax, ffi::input as u64)?;
        self.asm.call(rax)?;

        // Restore saved registers and deallocate stack space
        self.after_win64_call(stack_padding, slots)?;

        // `input()` returns a u16 within the rax register where the top six bytes are garbage, the
        // 7th byte is the input value (if the function succeeded) and the 8th byte is the status
        // code, 1 for error and 0 for success

        // // Allocate the register that'll hold the value gotten from the input
        // let input_reg = self.allocate_register()?;

        // Check if an IO error occurred
        self.asm.cmp(low_byte_register(rax), 0)?;

        // If so, jump to the io error label and exit
        self.asm.jne(self.io_failure)?;

        // Save rax to the input register
        self.asm.mov(input_reg, rax)?;
        // Shift right by one byte in order to keep only the input value
        self.asm.shr(input_reg, 8)?;

        // self.values
        //     .insert(assign.value(), Operand::Register(input_reg))
        //     .debug_unwrap_none();

        Ok(())
    }

    fn assemble_output(&mut self, output: &Output) -> AsmResult<()> {
        self.has_io_functions = true;

        // Setup the stack and save used registers
        let (stack_padding, slots) = self.before_win64_call()?;

        // Move the state pointer into rcx
        self.asm.mov(rcx, self.state_ptr() + stack_padding)?;

        // Move the given byte into rdx
        let value = self.get_value(output.value()).map_stack(|mut addr| {
            addr.offset += stack_padding as usize;
            addr
        });
        self.move_to_reg(rdx, value)?;

        // Call the output function
        type_eq::<unsafe extern "win64" fn(state: *mut State, byte: u64) -> u8>(ffi::output);
        #[allow(clippy::fn_to_numeric_cast)]
        self.asm.mov(rax, ffi::output as u64)?;
        self.asm.call(rax)?;

        // Restore saved registers and deallocate stack space
        self.after_win64_call(stack_padding, slots)?;

        // Check if an IO error occurred
        self.asm.cmp(low_byte_register(rax), 0)?;

        // If so, jump to the io error label and exit
        self.asm.jne(self.io_failure)?;

        Ok(())
    }

    fn assemble_terminator(
        &mut self,
        terminator: &Terminator,
        next_block: Option<BlockId>,
    ) -> AsmResult<()> {
        self.add_comment(terminator.pretty_print(PrettyConfig::minimal()));

        match terminator {
            Terminator::Error => panic!("encountered an error terminator"),
            Terminator::Unreachable => self.asm.ud2(),

            // If the next block is the target block, we can fall into it
            &Terminator::Jump(block) if Some(block) == next_block => Ok(()),
            Terminator::Jump(block) => {
                let jump_label = *self.block_labels.get(block).unwrap();
                self.asm.jmp(jump_label)
            }

            // TODO: Deallocate stack space and whatnot
            Terminator::Return(value) => {
                let value = self.get_value(*value);
                self.move_to_reg(rax, value)?;

                // Deallocate the stack space for the operand, we know that this is
                // the last usage for the value since the next instruction is a return
                if let Operand::Stack(slot) = value {
                    self.regalloc.free(&mut self.asm, slot)?;
                }

                // Return from the function
                self.asm.jmp(self.epilogue)
            }

            Terminator::Branch(branch) => {
                let cond = self.get_value(branch.condition());

                // Get both block's labels
                let (true_label, false_label) = (
                    *self.block_labels.get(&branch.true_jump()).unwrap(),
                    *self.block_labels.get(&branch.false_jump()).unwrap(),
                );

                // See <https://stackoverflow.com/a/54499552/9885253> for the logic behind the const-known
                // polyfills. Since our comparisons are currently pretty dumb, everything is an `x ≡ 0` so we
                // can just polyfill with things that non-destructively (in regards to register & stack state)
                // set the ZF to one or zero based on the result of the comparison, ZF=1 for equal and ZF=0 for
                // not equal
                match cond {
                    Operand::Byte(byte) => {
                        self.add_comment(format!("{} ≡ 0 = {}", byte, byte == 0));

                        if byte == 0 {
                            self.asm.jmp(true_label)?;
                        } else {
                            self.asm.jmp(false_label)?;
                        }

                        return Ok(());
                    }

                    Operand::Uint(uint) => {
                        self.add_comment(format!("{} ≡ 0 = {}", uint, uint == 0));

                        if uint == 0 {
                            self.asm.jmp(true_label)?;
                        } else {
                            self.asm.jmp(false_label)?;
                        }

                        return Ok(());
                    }

                    Operand::Bool(bool) => {
                        self.add_comment(format!("{} = {}", bool, bool));

                        if bool {
                            self.asm.jmp(true_label)?;
                        } else {
                            self.asm.jmp(false_label)?;
                        }

                        return Ok(());
                    }

                    Operand::Stack(slot) => {
                        let addr = self.stack_offset(slot);
                        self.asm.cmp(addr, 0)?;
                    }

                    Operand::Register(reg) => self.asm.cmp(reg, 0)?,
                }

                // If the next block is the true branch's block we can fall into it
                if Some(branch.true_jump()) == next_block {
                    self.asm.jnz(false_label)?;

                // If the next block is the false branch's block we can fall into it
                } else if Some(branch.false_jump()) == next_block {
                    self.asm.jz(true_label)?;

                // Otherwise we have to make a jump in both the true and false cases
                } else {
                    // If the condition is true, jump to the true label
                    self.asm.jz(true_label)?;

                    // Otherwise unconditionally jump to the false label
                    self.asm.jmp(false_label)?;
                }

                Ok(())
            }
        }
    }

    // See <https://stackoverflow.com/a/54499552/9885253>
    #[allow(dead_code)]
    fn set_zf_one(&mut self) -> AsmResult<()> {
        // If we have a free register available we can xor zero it in order to set ZF=1
        if let Some(free) = self.regalloc.peek_free_register().map(low_byte_register) {
            self.asm.xor(free, free)?;

        // Otherwise we'll compare a register against itself, which is always
        // true and results in ZF=1
        } else {
            let reg = low_byte_register(self.regalloc.least_recently_used_register());
            self.asm.cmp(reg, reg)?;
        }

        Ok(())
    }

    // See <https://stackoverflow.com/a/54499552/9885253>
    #[allow(dead_code)]
    fn set_zf_zero(&mut self) -> AsmResult<()> {
        // If we have a free register available we can use `or %reg, -1` to set ZF=0
        if let Some(free) = self.regalloc.peek_free_register().map(low_byte_register) {
            self.asm.or(free, -1)?;

        // Otherwise we'll test rsp against itself, rsp is known to be non-zero
        } else {
            self.asm.test(rsp, rsp)?;
        }

        Ok(())
    }

    /// Create labels for each basic block
    fn create_block_labels(&mut self, blocks: &Blocks) {
        for block in blocks {
            let label = self.create_label();
            self.block_labels
                .insert(block.id(), label)
                .debug_unwrap_none();
        }
    }

    fn move_to_reg(&mut self, dest: AsmRegister64, value: Operand) -> AsmResult<()> {
        match value {
            Operand::Byte(byte) => self.asm.mov(dest, byte as i64),
            Operand::Uint(uint) => self.asm.mov(dest, uint as i64),
            Operand::Bool(bool) => self.asm.mov(dest, bool as i64),
            Operand::Stack(slot) => {
                let ptr = self.stack_offset(slot);
                self.asm.mov(dest, ptr)
            }
            // If the source and destination registers are the same, do nothing
            Operand::Register(src) if src == dest => Ok(()),
            Operand::Register(src) => self.asm.mov(dest, src),
        }
    }

    #[track_caller]
    fn get_value(&mut self, value: Value) -> Operand {
        match value {
            Value::Byte(byte) => Operand::Byte(byte),
            Value::Uint(uint) => Operand::Uint(uint),
            Value::Bool(bool) => Operand::Bool(bool),
            Value::Val(var) => match self.values.get(&var).copied() {
                Some(operand) => operand,
                None => panic!(
                    "attempted to get the value of {:?}, but {} couldn't be found",
                    value, var,
                ),
            },
        }
    }

    #[track_caller]
    #[allow(dead_code)]
    fn get_var(&self, var: ValId) -> Operand {
        match self.values.get(&var) {
            Some(&operand) => operand,
            None => panic!(
                "attempted to get the value of {}, but it couldn't be found",
                var,
            ),
        }
    }

    /// Create the IO failure block
    // FIXME: Needs to deallocate all of the stack space from any given failure point
    //        as well as restore callee sided registers
    #[allow(clippy::fn_to_numeric_cast)]
    fn build_io_failure(&mut self) -> AsmResult<()> {
        self.set_label(self.io_failure);
        self.named_label(".IO_FAILURE");

        // Move the state pointer into the rcx register
        self.asm.mov(rcx, self.state_ptr())?;

        // The "most correct" method would be to call `self.after_win64_call()`
        // before the function call and `self.before_win64_call()` after we call
        // it, however it doesn't actually matter since we never use any variables
        // from in this code path, io failures diverge from the perspective of
        // the function. Instead, what we do is allocate the stack space for the
        // called function and align the stack before the function call and then
        // deallocate that space afterwards.

        // Allocate the required space by subtracting from rsp
        let stack_padding = self.regalloc.align_for_call() as i32;
        self.asm.sub(rsp, stack_padding)?;

        // Call the io error function
        type_eq::<unsafe extern "win64" fn(*mut State) -> bool>(ffi::io_error_encountered);
        self.asm.mov(rax, ffi::io_error_encountered as u64)?;
        self.asm.call(rax)?;

        // Deallocate the stack space allocated for the function call
        self.asm.add(rsp, stack_padding)?;

        // Move the return code into rax
        self.asm.mov(rax, RETURN_IO_FAILURE)?;

        // Return from the function

        // We technically would/could jump to the epilogue directly with
        // `self.asm.jmp(self.epilogue)?;`, but the io failure block should be
        // positioned directly before the epilogue block, meaning we can just
        // fall through into it. This will hold as long as `self.build_io_failure()`
        // is called directly before `self.epilogue()`

        Ok(())
    }

    /// Set up the function's prologue
    // TODO: Save & restore callee sided registers
    fn prologue(&mut self) -> AsmResult<()> {
        self.named_label(".PROLOGUE");

        // The function that called us must allocated 32 bytes of stack space
        // for parameters, regardless of whether or not we actually use 4
        // parameters
        //
        // Current stack state:
        // ┌──────────┬────────────┐
        // │ rsp + 32 │   unused   │
        // ├──────────┼────────────┤
        // │ rsp + 24 │   unused   │
        // ├──────────┼────────────┤
        // │ rsp + 16 │   unused   │
        // ├──────────┼────────────┤
        // │ rsp + 8  │   unused   │
        // └──────────┴────────────┘

        // rcx contains the state pointer
        self.asm
            .mov(self.state_ptr() - CALLEE_REGISTER_OFFSET, rcx)?;

        // rdx contains the tape's start pointer
        self.asm
            .mov(self.tape_start_ptr() - CALLEE_REGISTER_OFFSET, rdx)?;

        // r8 contains the tape's end pointer
        self.asm
            .mov(self.tape_end_ptr() - CALLEE_REGISTER_OFFSET, r8)?;

        // Current stack state:
        // ┌──────────┬────────────┐
        // │ rsp + 32 │   unused   │
        // ├──────────┼────────────┤
        // │ rsp + 24 │  tape end  │
        // ├──────────┼────────────┤
        // │ rsp + 16 │ tape start │
        // ├──────────┼────────────┤
        // │ rsp + 8  │ state ptr  │
        // └──────────┴────────────┘

        // TODO: We get one extra stack slot from the 32 bytes allocated by the caller,
        //       we should use it here
        // TODO: Only push/pop the registers we actually use
        for &register in CALLEE_SAVED_REGISTERS {
            self.asm.push(register)?;
        }

        // C = total_callee_regs * 8
        //
        // Current stack state:
        // ┌────────────────┬────────────────────────┐
        // │  rsp + C + 32  │         unused         │
        // ├────────────────┼────────────────────────┤
        // │  rsp + C + 24  │        tape end        │
        // ├────────────────┼────────────────────────┤
        // │  rsp + C + 16  │       tape start       │
        // ├────────────────┼────────────────────────┤
        // │  rsp + C + 8   │        state ptr       │
        // ├────────────────┼────────────────────────┤
        // │ rsp + C .. rsp │ non-volatile registers │
        // └────────────────┴────────────────────────┘

        Ok(())
    }

    // Expects that rax contains the function return code
    fn epilogue(&mut self) -> AsmResult<()> {
        self.set_label(self.epilogue);
        self.named_label(".EPILOGUE");

        for &register in CALLEE_SAVED_REGISTERS.iter().rev() {
            self.asm.pop(register)?;
        }

        self.asm.ret()?;

        Ok(())
    }

    /// Get a pointer to the `*mut State` stored on the stack
    fn state_ptr(&self) -> AsmMemoryOperand {
        qword_ptr(rsp + 8 + CALLEE_REGISTER_OFFSET + self.regalloc.virtual_rsp())
    }

    /// Get a pointer to the `*mut u8` stored on the stack
    fn tape_start_ptr(&self) -> AsmMemoryOperand {
        qword_ptr(rsp + 16 + CALLEE_REGISTER_OFFSET + self.regalloc.virtual_rsp())
    }

    /// Get a pointer to the `*const u8` stored on the stack
    fn tape_end_ptr(&self) -> AsmMemoryOperand {
        qword_ptr(rsp + 24 + CALLEE_REGISTER_OFFSET + self.regalloc.virtual_rsp())
    }

    fn stack_offset(&self, slot: StackSlot) -> AsmMemoryOperand {
        qword_ptr(rsp + self.regalloc.slot_offset(slot))
    }

    // /// Push a register's value to the stack
    // fn push(&mut self, register: AsmRegister64) -> AsmResult<StackSlot> {
    //     let slot = StackSlot::new(self.regalloc.virtual_rsp(), 8);
    //
    //     self.asm.push(register)?;
    //     self.regalloc.stack.virtual_rsp += 8;
    //
    //     Ok(slot)
    // }

    fn allocate_register(&mut self) -> AsmResult<AsmRegister64> {
        let (register, spilled) = self.regalloc.allocate(&mut self.asm, true)?;

        // Spill the spilled register to the stack
        if let Some(spilled) = spilled {
            self.spill_register(register, spilled);
        }

        Ok(register)
    }

    #[allow(dead_code)]
    fn allocate_specific(&mut self, register: AsmRegister64) -> AsmResult<AsmRegister64> {
        let (register, spilled) = self
            .regalloc
            .allocate_specific(&mut self.asm, register, true)?;

        // Spill the spilled register to the stack
        if let Some(spilled) = spilled {
            self.spill_register(register, spilled);
        }

        Ok(register)
    }

    #[allow(dead_code)]
    fn allocate_specific_clobbered(&mut self, register: AsmRegister64) -> AsmResult<AsmRegister64> {
        let (register, spilled) =
            self.regalloc
                .allocate_specific(&mut self.asm, register, false)?;

        // Spill the spilled register to the stack
        if let Some(spilled) = spilled {
            self.spill_register(register, spilled);
        }

        Ok(register)
    }

    fn spill_register(&mut self, register: AsmRegister64, spilled: StackSlot) {
        let mut replaced_old = false;
        for value in self.values.values_mut() {
            if *value == Operand::Register(register) {
                *value = Operand::Stack(spilled);
                replaced_old = true;

                break;
            }
        }

        debug_assert!(replaced_old);
    }

    fn deallocate_register(&mut self, register: AsmRegister64) {
        self.regalloc.deallocate(register);
    }

    #[allow(dead_code)]
    fn infinite_loop(&mut self) -> AsmResult<()> {
        // Add a nop to avoid having multiple labels for an instruction
        self.asm.nop()?;

        self.add_comment("infinite loop");

        let label = self.asm.create_label();
        self.set_label(label);
        self.asm.jmp(label)?;

        Ok(())
    }

    /// Saves all currently used registers and allocates 32 bytes of stack for the called function
    /// as well as aligning the stack to 16 bytes. Returns the total extra stack space that was allocated
    /// for both the parameters and alignment padding
    fn before_win64_call(&mut self) -> AsmResult<(i32, Vec<(AsmRegister64, StackSlot)>)> {
        let mut slots = Vec::new();

        let registers: Vec<_> = self.regalloc.used_volatile_registers().collect();
        for register in registers {
            let slot = self.regalloc.push(&mut self.asm, register)?;
            slots.push((register, slot));

            tracing::debug!(
                "pushed register {:?} to slot {:?} before win64 call",
                register,
                slot,
            );
        }

        let stack_padding = self.regalloc.align_for_call() as i32;

        // Allocate the required space by subtracting from rsp
        self.asm.sub(rsp, stack_padding)?;

        Ok((stack_padding, slots))
    }

    /// Restores all saved registers and deallocates the space that was allocated for the called function
    fn after_win64_call(
        &mut self,
        stack_padding: i32,
        slots: Vec<(AsmRegister64, StackSlot)>,
    ) -> AsmResult<()> {
        // Deallocate the stack space we allocated for both the function arguments
        // and the padding required to align the stack to 16 bytes
        self.asm.add(rsp, stack_padding)?;

        // Pop all used registers from the stack
        for (register, slot) in slots.into_iter().rev() {
            self.regalloc.pop(&mut self.asm, slot, register)?;
            tracing::debug!(
                "popped register {:?} from slot {:?} after win64 call",
                register,
                slot,
            );
        }

        // Remove all clobbered registers
        let registers: Vec<_> = self.regalloc.clobbered_registers().collect();
        for clobbered in registers {
            self.regalloc.deallocate(clobbered);
            tracing::debug!("deallocated clobbered register {:?}", clobbered);
        }

        Ok(())
    }

    fn add_comment<C>(&mut self, comment: C)
    where
        C: ToString,
    {
        let idx = self.asm.instructions().len();
        self.comments
            .entry(idx)
            .or_insert_with(|| Vec::with_capacity(1))
            .push(comment.to_string());
    }

    fn create_label(&mut self) -> CodeLabel {
        self.asm.create_label()
    }

    #[track_caller]
    fn named_label<N>(&mut self, name: N)
    where
        N: Into<Cow<'static, str>>,
    {
        self.named_labels
            .insert(self.asm.instructions().len(), name.into())
            .debug_unwrap_none();
    }

    #[track_caller]
    fn set_label(&mut self, mut label: CodeLabel) {
        self.asm.set_label(&mut label).unwrap();
    }
}

fn type_eq<T>(_: T) {}

#[allow(non_upper_case_globals)]
fn low_byte_register(register: AsmRegister64) -> AsmRegister8 {
    match register {
        rax => al,
        rbx => bl,
        rcx => cl,
        rdx => dl,
        rsp => spl,
        rbp => bpl,
        rsi => sil,
        rdi => dil,
        r8 => r8b,
        r9 => r9b,
        r10 => r10b,
        r11 => r11b,
        r12 => r12b,
        r13 => r13b,
        r14 => r14b,
        r15 => r15b,
        _ => unreachable!(),
    }
}
*/
