#![allow(dead_code)]

use crate::{
    ir::{
        Add, Assign, AssignTag, Block, Call, Const, Eq, Expr, Gamma, Instruction, LifetimeEnd,
        Load, Pretty, PrettyConfig, Store, Sub, Theta, Value, VarId, Variance,
    },
    jit::{
        cir_to_bb,
        ffi::{self, State},
        memory::{CodeBuffer, Executable},
        regalloc::{Regalloc, StackSlot},
    },
    utils::AssertNone,
};
use iced_x86::code_asm::{CodeAssembler, *};
use std::{borrow::Cow, collections::BTreeMap};

type AsmResult<T> = Result<T, IcedError>;

const BITNESS: u32 = 64;

const RETURN_SUCCESS: i64 = 0;

const RETURN_IO_FAILURE: i64 = 101;

pub struct Jit {
    asm: CodeAssembler,
    io_failure: CodeLabel,
    epilogue: CodeLabel,
    has_io_functions: bool,
    values: BTreeMap<VarId, Operand>,
    regalloc: Regalloc,
    comments: BTreeMap<usize, Vec<String>>,
    named_labels: BTreeMap<usize, Cow<'static, str>>,
    inputs: BTreeMap<VarId, Operand>,
    output_feedbacks: Vec<BTreeMap<VarId, VarId>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Operand {
    Register(AsmRegister64),
    Const(Const),
    Stack(StackSlot),
}

impl Jit {
    pub fn new(_tape_len: usize) -> AsmResult<Self> {
        let mut asm = CodeAssembler::new(BITNESS)?;
        let io_failure = asm.create_label();
        let epilogue = asm.create_label();

        Ok(Self {
            asm,
            io_failure,
            epilogue,
            has_io_functions: false,
            values: BTreeMap::new(),
            regalloc: Regalloc::new(),
            comments: BTreeMap::new(),
            named_labels: BTreeMap::new(),
            inputs: BTreeMap::new(),
            output_feedbacks: Vec::new(),
        })
    }

    #[tracing::instrument(skip_all)]
    pub fn assemble(&mut self, block: &Block) -> AsmResult<(Executable<CodeBuffer>, String)> {
        let blocks = cir_to_bb::translate(block);
        println!(
            "SSA form:\n{}",
            blocks.pretty_print(PrettyConfig::minimal()),
        );

        self.named_label(".PROLOGUE");
        self.prologue()?;

        self.named_label(".BODY");
        self.assemble_block(block)?;

        self.named_label(".EPILOGUE");
        // Set the return code
        self.asm.mov(rax, RETURN_SUCCESS)?;
        self.epilogue()?;

        // Only build the IO handler if there's IO functions
        if self.has_io_functions {
            self.build_io_failure()?;
        }

        let code_buffer = {
            // The maximum size of an instruction is 15 bytes, so we allocate the most memory we could possibly use
            let maximum_possible_size = self.asm.instructions().len() * 15;
            let mut code_buffer = CodeBuffer::new(maximum_possible_size).unwrap();

            let code = self.asm.assemble(code_buffer.as_ptr() as u64)?;

            debug_assert!(code_buffer.len() <= code.capacity());
            code_buffer.copy_from_slice(&code).unwrap();

            code_buffer.executable().unwrap()
        };
        let pretty = String::new(); // self.disassemble();

        Ok((code_buffer, pretty))
    }

    fn assemble_block(&mut self, block: &[Instruction]) -> AsmResult<()> {
        for inst in block {
            self.assemble_inst(inst)?;
        }

        Ok(())
    }

    fn assemble_inst(&mut self, inst: &Instruction) -> AsmResult<()> {
        match inst {
            Instruction::Call(call) => {
                self.add_comment(call.pretty_print(PrettyConfig::minimal()));
                self.assemble_call(call)?.debug_unwrap_none();

                Ok(())
            }

            Instruction::Assign(assign) => self.assemble_assign(assign),
            Instruction::Theta(theta) => self.assemble_theta(theta),
            Instruction::Gamma(gamma) => self.assemble_gamma(gamma),
            Instruction::Store(store) => self.assemble_store(store),
            Instruction::LifetimeEnd(lifetime) => self.lifetime_end(lifetime),
        }
    }

    #[tracing::instrument(skip(self))]
    fn lifetime_end(&mut self, lifetime: &LifetimeEnd) -> AsmResult<()> {
        // self.add_comment(lifetime.pretty_print(PrettyConfig::minimal()));

        if let Some(&value) = self.values.get(&lifetime.var) {
            match value {
                Operand::Register(register) => self.deallocate_register(register),
                Operand::Stack(slot) => self.regalloc.free(&mut self.asm, slot)?,
                // Don't need to do anything special to "deallocate" a constant
                Operand::Const(_) => {}
            }

            self.values.remove(&lifetime.var).unwrap();
        }

        Ok(())
    }

    fn assemble_call(&mut self, call: &Call) -> AsmResult<Option<AsmRegister64>> {
        match &*call.function {
            "input" => self.call_input(call).map(Some),
            "output" => self.call_output(call).map(|()| None),
            _ => unreachable!(),
        }
    }

    #[tracing::instrument(skip(self))]
    fn assemble_gamma(&mut self, gamma: &Gamma) -> AsmResult<()> {
        let gamma_cond = gamma.cond.pretty_print(PrettyConfig::minimal());
        self.add_comment(format!("if {} {{ ... }} else {{ ... }}", gamma_cond));

        let after_gamma = self.create_label();

        match self.get_value(gamma.cond) {
            Operand::Register(register) => self.asm.cmp(register, 0)?,
            Operand::Stack(offset) => self.asm.cmp(self.stack_offset(offset), 0)?,
            Operand::Const(_) => todo!(),
        }

        // If the true branch is empty and the false isn't
        if gamma.true_is_empty() && !gamma.false_is_empty() {
            // If the condition is true, skip the false branch's code
            self.asm.jz(after_gamma)?;

            // Build the false branch
            self.add_comment(format!(
                "[false branch] if {} {{ ... }} else {{ ... }}",
                gamma_cond,
            ));
            self.assemble_block(&gamma.false_branch)?;

        // If the true branch isn't empty and the false is
        } else if !gamma.true_is_empty() && gamma.false_is_empty() {
            // If the condition is false, skip the true branch's code
            self.asm.jnz(after_gamma)?;

            // Build the true branch
            self.add_comment(format!(
                "[true branch] if {} {{ ... }} else {{ ... }}",
                gamma_cond,
            ));
            self.assemble_block(&gamma.true_branch)?;

        // If both branches are full (TODO: if both branches are empty?)
        } else {
            // If the condition is false, jump to the false branch
            let false_branch = self.create_label();
            self.asm.jnz(false_branch)?;

            // Build the true branch
            self.add_comment(format!(
                "[true branch] if {} {{ ... }} else {{ ... }}",
                gamma_cond,
            ));
            self.assemble_block(&gamma.true_branch)?;
            // After we execute the true branch we jump to the code after the gamma
            self.asm.jmp(after_gamma)?;

            // Build the false branch
            self.add_comment(format!(
                "[false branch] if {} {{ ... }} else {{ ... }}",
                gamma_cond,
            ));
            self.set_label(false_branch);
            self.assemble_block(&gamma.false_branch)?;
        }

        self.add_comment(format!("[end] if {} {{ ... }} else {{ ... }}", gamma_cond));

        self.set_label(after_gamma);
        Ok(())
    }

    #[tracing::instrument(skip(self))]
    fn assemble_theta(&mut self, theta: &Theta) -> AsmResult<()> {
        self.add_comment("[header] do {{ ... }} while {{ ... }}");

        // A fix to keep us from making instructions with multiple labels
        self.asm.nop()?;

        // Create the theta's header
        for inst in &theta.body {
            if matches!(
                inst,
                Instruction::Assign(Assign {
                    tag: AssignTag::InputParam(_),
                    ..
                })
            ) {
                self.assemble_inst(inst)?;
            }
        }

        self.add_comment(format!(
            "do {{ ... }} while {{ {} }}",
            theta.cond.unwrap().pretty_print(PrettyConfig::minimal()),
        ));

        self.output_feedbacks.push(theta.output_feedback.clone());

        // Create a label for the theta's body
        let theta_body = self.create_label();
        self.set_label(theta_body);

        // Build the theta's body
        for inst in &theta.body {
            if !matches!(
                inst,
                Instruction::Assign(Assign {
                    tag: AssignTag::InputParam(_),
                    ..
                })
            ) {
                self.assemble_inst(inst)?;
            }
        }

        self.add_comment(format!(
            "[condition] do {{ ... }} while {{ {} }}",
            theta.cond.unwrap().pretty_print(PrettyConfig::minimal()),
        ));

        // Get the theta's condition
        let condition = self.get_value(theta.cond.unwrap());
        match condition {
            Operand::Register(register) => {
                self.asm.cmp(register, 0).unwrap();
                self.deallocate_register(register);
            }
            Operand::Stack(offset) => {
                // TODO: Use al + byte_ptr
                self.asm.mov(rax, self.stack_offset(offset))?;
                self.asm.and(rax, 1)?;
                self.regalloc.free(&mut self.asm, offset)?;
                self.asm.cmp(rax, 0)?;
            }
            Operand::Const(_) => todo!(),
        }
        match theta.cond.unwrap() {
            Value::Var(var) => self.values.remove(&var).debug_unwrap(),
            Value::Const(_) | Value::Missing => {}
        }

        // if let Some(freed) = self.regalloc.stack.free_vacant_slots() {
        //     self.asm.add(rsp, freed.get() as i32)?;
        // }

        // If the condition is true, jump to the beginning of the theta's body
        self.asm.je(theta_body)?;

        self.output_feedbacks.pop().debug_unwrap();

        self.add_comment(format!(
            "[end] do {{ ... }} while {{ {} }}",
            theta.cond.unwrap().pretty_print(PrettyConfig::minimal()),
        ));

        Ok(())
    }

    /// Invoke the input function to get a single byte from stdin
    #[tracing::instrument(skip(self))]
    #[allow(clippy::fn_to_numeric_cast)]
    fn call_input(&mut self, call: &Call) -> AsmResult<AsmRegister64> {
        debug_assert_eq!(call.function, "input");
        debug_assert_eq!(call.args.len(), 0);

        self.has_io_functions = true;

        // Move the state pointer into rcx
        self.asm.mov(rcx, self.state_ptr())?;

        // Setup the stack and save used registers
        let (stack_padding, slots) = self.before_win64_call()?;

        // Call the input function
        type_eq::<unsafe extern "win64" fn(state: *mut State) -> u16>(ffi::input);
        self.asm.mov(rax, ffi::input as u64)?;
        self.asm.call(rax)?;

        // Restore saved registers and deallocate stack space
        self.after_win64_call(stack_padding, slots)?;

        // `input()` returns a u16 within the rax register where the top six bytes are garbage, the
        // 7th byte is the input value (if the function succeeded) and the 8th byte is the status
        // code, 1 for error and 0 for success

        // Allocate the register that'll hold the value gotten from the input
        let input_reg = self.allocate_register()?;

        // Save rax to the input register
        self.asm.mov(input_reg, rax)?;

        // Check if an IO error occurred
        self.asm.cmp(low_register(rax), 0)?;

        // If so, jump to the io error label and exit
        self.asm.jnz(self.io_failure)?;

        // Shift right by one byte in order to keep only the input value
        self.asm.shr(input_reg, 8)?;

        // input_reg holds the input value
        Ok(input_reg)
    }

    /// Invoke the output function to print a single byte to stdout
    #[tracing::instrument(skip(self))]
    #[allow(clippy::fn_to_numeric_cast)]
    fn call_output(&mut self, call: &Call) -> AsmResult<()> {
        debug_assert_eq!(call.function, "output");
        debug_assert_eq!(call.args.len(), 1);

        self.has_io_functions = true;

        // Move the state pointer into rcx
        self.asm.mov(rcx, self.state_ptr())?;

        // Move the given byte into rdx
        match call.args[0] {
            Value::Var(var) => match self.get_var(var) {
                Operand::Register(register) => self.asm.mov(rdx, register)?,
                Operand::Stack(offset) => self.asm.mov(rdx, self.stack_offset(offset))?,
                Operand::Const(constant) => self
                    .asm
                    .mov(rdx, constant.convert_to_u8().unwrap() as u64)?,
            },

            Value::Const(constant) => self
                .asm
                .mov(rdx, constant.convert_to_u8().unwrap() as u64)?,

            Value::Missing => unreachable!(),
        };

        // Setup the stack and save used registers
        let (stack_padding, slots) = self.before_win64_call()?;

        // Call the output function
        type_eq::<unsafe extern "win64" fn(state: *mut State, byte: u64) -> u8>(ffi::output);
        self.asm.mov(rax, ffi::output as u64)?;
        self.asm.call(rax)?;

        // Restore saved registers and deallocate stack space
        self.after_win64_call(stack_padding, slots)?;

        // Check if an IO error occurred
        self.asm.cmp(low_register(rax), 0)?;
        // If so, jump to the io error label and exit
        self.asm.jnz(self.io_failure)?;

        Ok(())
    }

    #[tracing::instrument(skip(self))]
    fn assemble_assign(&mut self, assign: &Assign) -> AsmResult<()> {
        self.add_comment(assign.pretty_print(PrettyConfig::minimal()));

        // Handle input params
        if matches!(assign.tag, AssignTag::InputParam(Variance::Variant { .. })) {
            let dest = self.allocate_register()?;

            let value = match assign.value {
                Expr::Value(value) => self.get_value(value),
                _ => unreachable!(),
            };
            match value {
                Operand::Register(reg) => self.asm.mov(dest, reg)?,
                Operand::Const(value) => {
                    self.asm.mov(dest, value.convert_to_u32().unwrap() as i64)?;
                }
                Operand::Stack(slot) => self.asm.mov(dest, byte_ptr(self.stack_offset(slot)))?,
            }

            self.inputs
                .insert(assign.var, Operand::Register(dest))
                .debug_unwrap_none();
            self.values
                .insert(assign.var, Operand::Register(dest))
                .debug_unwrap_none();

        // Handle output params
        // FIXME: Will cause issues for loops with non-feedback outputs
        } else if assign.is_output_param() {
            let value = match assign.value {
                Expr::Value(value) => value,
                _ => unreachable!(),
            };
            let value = self.get_value(value);

            let output_reg = self.allocate_register()?;

            if let Some(&input) = self
                .output_feedbacks
                .last()
                .and_then(|feedback| feedback.get(&assign.var))
            {
                match self.get_input(input) {
                    Operand::Register(feedback) => match value {
                        Operand::Register(reg) => self.asm.mov(feedback, reg)?,
                        Operand::Const(value) => self
                            .asm
                            .mov(feedback, value.convert_to_u32().unwrap() as i64)?,
                        Operand::Stack(slot) => self.asm.mov(feedback, self.stack_offset(slot))?,
                    },
                    Operand::Stack(_) | Operand::Const(_) => unreachable!(),
                }
            }

            match value {
                Operand::Register(reg) => self.asm.mov(output_reg, reg)?,
                Operand::Const(value) => self
                    .asm
                    .mov(output_reg, value.convert_to_u32().unwrap() as i64)?,
                Operand::Stack(slot) => self.asm.mov(output_reg, self.stack_offset(slot))?,
            }

            self.values
                .insert(assign.var, Operand::Register(output_reg));

        // Handle normal assignments
        } else {
            match &assign.value {
                Expr::Eq(eq) => {
                    let register = self.assemble_eq(eq)?;
                    self.values
                        .insert(assign.var, Operand::Register(register))
                        .debug_unwrap_none();
                }

                Expr::Add(add) => {
                    let register = self.assemble_add(add)?;
                    self.values
                        .insert(assign.var, Operand::Register(register))
                        .debug_unwrap_none();
                }

                Expr::Sub(sub) => {
                    let register = self.assemble_sub(sub)?;
                    self.values
                        .insert(assign.var, Operand::Register(register))
                        .debug_unwrap_none();
                }

                Expr::Mul(_) => todo!(),

                Expr::Not(not) => {
                    let dest = self.allocate_register()?;

                    // Get the input value
                    match self.get_value(not.value) {
                        Operand::Register(register) => self.asm.mov(dest, register)?,
                        Operand::Stack(offset) => self.asm.mov(dest, self.stack_offset(offset))?,
                        Operand::Const(constant) => {
                            let value = constant.convert_to_u32().unwrap();
                            self.asm.mov(dest, value as i64)?;
                        }
                    }

                    // Perform a *logical* not on it
                    // FIXME: This will break for stuff that needs actual bitwise not
                    //        instead of just logical not
                    self.asm.xor(dest, 1)?;

                    self.values
                        .insert(assign.var, Operand::Register(dest))
                        .debug_unwrap_none();
                }

                Expr::Neg(neg) => {
                    let dest = self.allocate_register()?;

                    // Get the input value
                    match self.get_value(neg.value) {
                        Operand::Register(register) => self.asm.mov(dest, register)?,
                        Operand::Stack(offset) => self.asm.mov(dest, self.stack_offset(offset))?,
                        Operand::Const(constant) => {
                            let value = constant.convert_to_u32().unwrap();
                            self.asm.mov(dest, value as i64)?;
                        }
                    }

                    // Perform twos compliant negation on it
                    self.asm.neg(dest)?;

                    self.values
                        .insert(assign.var, Operand::Register(dest))
                        .debug_unwrap_none();
                }

                // TODO: `mov reg, [reg]` is legal, we can skip clobbering rax in some scenarios
                Expr::Load(load) => self.assemble_load(load, assign)?,

                Expr::Call(call) => {
                    let input_reg = self.assemble_call(call)?.unwrap();
                    self.values
                        .insert(assign.var, Operand::Register(input_reg))
                        .debug_unwrap_none();
                }

                Expr::Value(value) => {
                    match *value {
                        Value::Var(var) => {
                            match self.get_var(var) {
                                Operand::Register(register) => {
                                    let dest = self.allocate_register()?;
                                    self.asm.mov(dest, register)?;

                                    self.values.insert(assign.var, Operand::Register(dest));
                                    // .debug_unwrap_none();
                                }

                                Operand::Const(constant) => {
                                    self.values
                                        .insert(assign.var, Operand::Const(constant))
                                        .debug_unwrap_none();
                                }

                                Operand::Stack(offset) => {
                                    let dest = self.allocate_register()?;
                                    self.asm.mov(dest, self.stack_offset(offset))?;

                                    self.values
                                        .insert(assign.var, Operand::Register(dest))
                                        .debug_unwrap_none();
                                }
                            }
                        }

                        Value::Const(constant) => {
                            self.values
                                .insert(assign.var, Operand::Const(constant))
                                .debug_unwrap_none();
                        }

                        Value::Missing => todo!(),
                    }
                }
            }
        }

        Ok(())
    }

    fn assemble_load(&mut self, load: &Load, assign: &Assign) -> AsmResult<()> {
        let dest = self.allocate_register()?;

        match self.get_value(load.ptr) {
            Operand::Register(register) => {
                // Get the start pointer from the stack
                self.asm.mov(dest, self.tape_start_ptr())?;

                // FIXME: Subtraction??
                self.asm.add(dest, register)?;

                // Dereference the pointer and store its value in the destination register
                self.asm.mov(al, byte_ptr(dest))?;
                self.asm.movzx(dest, al)?;
            }

            Operand::Stack(offset) => {
                // Get the start pointer from the stack
                self.asm.mov(dest, self.tape_start_ptr())?;

                // FIXME: Subtraction??
                self.asm.add(dest, self.stack_offset(offset))?;

                // Dereference the pointer and store its value in the destination register
                self.asm.mov(al, byte_ptr(dest))?;
                self.asm.movzx(dest, al)?;
            }

            Operand::Const(constant) => {
                let offset = constant.convert_to_u32().unwrap();

                // Zero out the target register
                self.asm.xor(dest, dest)?;

                // Get the start pointer from the stack, add the offset to
                // it and load the value to the destination register
                self.asm.mov(rax, self.tape_start_ptr())?;

                if offset != 0 {
                    self.asm.add(rax, offset as i32)?;
                }

                self.asm.mov(low_register(dest), byte_ptr(rax))?;
            }
        }

        self.values
            .insert(assign.var, Operand::Register(dest))
            .debug_unwrap_none();

        Ok(())
    }

    // TODO: `mov [reg], reg` is legal, we can skip clobbering rax in some scenarios
    #[tracing::instrument(skip(self))]
    fn assemble_store(&mut self, store: &Store) -> AsmResult<()> {
        self.add_comment(store.pretty_print(PrettyConfig::minimal()));

        // Get the start pointer from the stack
        self.asm.mov(rax, self.tape_start_ptr())?;

        // Offset the tape pointer
        match self.get_value(store.ptr) {
            // FIXME: Subtraction??
            Operand::Register(register) => self.asm.add(rax, register)?,

            // FIXME: Subtraction??
            Operand::Stack(offset) => self.asm.add(rax, self.stack_offset(offset))?,

            Operand::Const(constant) => {
                let offset = constant.convert_to_u32().unwrap();

                if offset != 0 {
                    self.asm.add(rax, offset as i32)?;
                }
            }
        }

        // Store the given value to the given pointer
        match self.get_value(store.value) {
            Operand::Register(register) => self.asm.mov(byte_ptr(rax), low_register(register))?,

            Operand::Stack(offset) => {
                self.asm.mov(rcx, self.stack_offset(offset))?;
                self.asm.mov(byte_ptr(rax), cl)?;
            }

            Operand::Const(value) => {
                let value = value.convert_to_u8().unwrap();

                if value != 0 {
                    self.asm.mov(byte_ptr(rax), value as i32)?;
                }
            }
        }

        Ok(())
    }

    fn assemble_add(&mut self, add: &Add) -> AsmResult<AsmRegister64> {
        // FIXME: These could actually be optimized a lot with register reuse
        match (self.get_value(add.lhs), self.get_value(add.rhs)) {
            (Operand::Register(lhs), Operand::Register(rhs)) => {
                let dest = self.allocate_register()?;

                self.asm.mov(dest, lhs)?;
                // FIXME: Subtracting registers?
                self.asm.add(dest, rhs)?;

                Ok(dest)
            }

            (Operand::Register(lhs), Operand::Const(rhs)) => {
                let rhs = rhs.convert_to_u32().unwrap();

                let dest = self.allocate_register()?;
                self.asm.mov(dest, lhs)?;

                match rhs {
                    0 => {}
                    1 => self.asm.inc(dest)?,
                    rhs => self.asm.add(dest, rhs as i32)?,
                }

                Ok(dest)
            }

            (Operand::Const(lhs), Operand::Register(rhs)) => {
                let dest = self.allocate_register()?;
                let lhs = lhs.convert_to_u32().unwrap();

                if lhs == 0 {
                    self.asm.mov(dest, rhs)?;
                    Ok(dest)
                } else {
                    self.asm.mov(dest, rhs)?;
                    self.asm.add(dest, lhs as i32)?;

                    Ok(dest)
                }
            }

            (Operand::Const(lhs), Operand::Const(rhs)) => {
                let (lhs, rhs) = (lhs.convert_to_u32().unwrap(), rhs.convert_to_u32().unwrap());
                let sum = lhs + rhs;

                let dest = self.allocate_register()?;
                self.asm.mov(dest, sum as i64)?;

                Ok(dest)
            }

            (Operand::Register(_), Operand::Stack(_)) => todo!(),
            (Operand::Const(_), Operand::Stack(_)) => todo!(),
            (Operand::Stack(_), Operand::Register(_)) => todo!(),

            (Operand::Stack(offset), Operand::Const(rhs)) => {
                let rhs = rhs.convert_to_u32().unwrap();

                let dest = self.allocate_register()?;
                self.asm.mov(dest, self.stack_offset(offset))?;

                match rhs {
                    0 => {}
                    1 => self.asm.inc(dest)?,
                    rhs => self.asm.add(dest, rhs as i32)?,
                }

                Ok(dest)
            }

            (Operand::Stack(_), Operand::Stack(_)) => todo!(),
        }
    }

    fn assemble_sub(&mut self, sub: &Sub) -> AsmResult<AsmRegister64> {
        // FIXME: These could actually be optimized a lot with register reuse
        match (self.get_value(sub.lhs), self.get_value(sub.rhs)) {
            (Operand::Register(lhs), Operand::Register(rhs)) => {
                let dest = self.allocate_register()?;

                self.asm.mov(dest, lhs)?;
                self.asm.sub(dest, rhs)?;

                Ok(dest)
            }

            (Operand::Register(lhs), Operand::Const(rhs)) => {
                let rhs = rhs.convert_to_u32().unwrap();

                let dest = self.allocate_register()?;
                self.asm.mov(dest, lhs)?;

                match rhs {
                    0 => {}
                    1 => self.asm.dec(dest)?,
                    rhs => self.asm.sub(dest, rhs as i32)?,
                }

                Ok(dest)
            }

            (Operand::Const(lhs), Operand::Register(rhs)) => {
                let dest = self.allocate_register()?;
                let lhs = lhs.convert_to_u32().unwrap();

                if lhs == 0 {
                    self.asm.mov(dest, rhs)?;
                    Ok(dest)
                } else {
                    self.asm.mov(dest, lhs as i64)?;
                    self.asm.sub(dest, rhs)?;

                    Ok(dest)
                }
            }

            (Operand::Const(lhs), Operand::Const(rhs)) => {
                let (lhs, rhs) = (lhs.convert_to_u32().unwrap(), rhs.convert_to_u32().unwrap());
                let sum = lhs - rhs;

                let dest = self.allocate_register()?;
                self.asm.mov(dest, sum as i64)?;

                Ok(dest)
            }

            (Operand::Register(_), Operand::Stack(_)) => todo!(),
            (Operand::Const(_), Operand::Stack(_)) => todo!(),
            (Operand::Stack(_), Operand::Register(_)) => todo!(),

            (Operand::Stack(offset), Operand::Const(rhs)) => {
                let rhs = rhs.convert_to_u32().unwrap();

                let dest = self.allocate_register()?;
                self.asm.mov(dest, self.stack_offset(offset))?;

                match rhs {
                    0 => {}
                    1 => self.asm.dec(dest)?,
                    rhs => self.asm.sub(dest, rhs as i32)?,
                }

                Ok(dest)
            }

            (Operand::Stack(_), Operand::Stack(_)) => todo!(),
        }
    }

    // FIXME: It's inefficient to always store comparisons in registers when we could
    //        just keep them in al 99% of the time
    fn assemble_eq(&mut self, eq: &Eq) -> AsmResult<AsmRegister64> {
        let dest = self.allocate_register()?;

        match (self.get_value(eq.lhs), self.get_value(eq.rhs)) {
            (Operand::Register(lhs), Operand::Register(rhs)) => {
                self.asm.cmp(lhs, rhs)?;

                // Set al to 1 if the operands are equal
                self.asm.setne(al)?;

                // Move the comparison result from al into the allocated
                // register with a zero sign extension
                self.asm.movzx(dest, al)?;
            }

            (Operand::Register(lhs), Operand::Const(rhs)) => {
                self.asm.cmp(lhs, rhs.convert_to_u32().unwrap() as i32)?;

                // Set al to 1 if the operands are equal
                self.asm.setne(al)?;

                // Move the comparison result from al into the allocated
                // register with a zero sign extension
                self.asm.movzx(dest, al)?;
            }

            (Operand::Const(lhs), Operand::Register(rhs)) => {
                self.asm.cmp(rhs, lhs.convert_to_u32().unwrap() as i32)?;

                // Set al to 1 if the operands are equal
                self.asm.setne(al)?;

                // Move the comparison result from al into the allocated
                // register with a zero sign extension
                self.asm.movzx(dest, al)?;
            }

            (Operand::Const(lhs), Operand::Const(rhs)) => {
                let are_equal = lhs.convert_to_u32().unwrap() == rhs.convert_to_u32().unwrap();
                self.asm.mov(dest, are_equal as i64)?;
            }

            (Operand::Register(_), Operand::Stack(_)) => todo!(),
            (Operand::Const(_), Operand::Stack(_)) => todo!(),
            (Operand::Stack(_), Operand::Register(_)) => todo!(),
            (Operand::Stack(_), Operand::Const(_)) => todo!(),
            (Operand::Stack(_), Operand::Stack(_)) => todo!(),
        }

        Ok(dest)
    }

    #[track_caller]
    fn get_value(&mut self, value: Value) -> Operand {
        match value {
            Value::Const(constant) => Operand::Const(constant),
            Value::Var(var) => self.values.get(&var).copied().unwrap_or_else(|| {
                panic!(
                    "attempted to get the value of {:?}, but {} couldn't be found",
                    value, var,
                )
            }),
            Value::Missing => panic!("tried to jit a missing value"),
        }
    }

    #[track_caller]
    fn get_var(&self, var: VarId) -> Operand {
        match self.values.get(&var) {
            Some(&operand) => operand,
            None => panic!(
                "attempted to get the value of {}, but it couldn't be found",
                var,
            ),
        }
    }

    #[track_caller]
    fn get_input(&self, input: VarId) -> Operand {
        match self.inputs.get(&input) {
            Some(&operand) => operand,
            None => panic!(
                "attempted to get the input value {}, but it couldn't be found",
                input,
            ),
        }
    }

    /// Create the IO failure block
    // FIXME: Needs to deallocate all of the stack space from any given failure point
    //        as well as restore callee sided registers
    #[allow(clippy::fn_to_numeric_cast)]
    fn build_io_failure(&mut self) -> AsmResult<()> {
        self.set_label(self.io_failure);

        self.add_comment("handle IO errors");

        // Move the state pointer into the rcx register
        self.asm.mov(rcx, self.state_ptr())?;

        // Setup the stack and save used registers
        let (stack_padding, slots) = self.before_win64_call()?;

        // Call the io error function
        type_eq::<unsafe extern "win64" fn(*mut State) -> bool>(ffi::io_error_encountered);
        self.asm.mov(rax, ffi::io_error_encountered as u64)?;
        self.asm.call(rax)?;

        // Restore saved registers and deallocate stack space
        self.after_win64_call(stack_padding, slots)?;

        // Set the return code
        self.asm.mov(rax, RETURN_IO_FAILURE)?;

        // Return from the function
        self.asm.ret()?;

        Ok(())
    }

    /// Set up the function's prologue
    fn prologue(&mut self) -> AsmResult<()> {
        // rcx contains the state pointer
        self.asm.mov(self.state_ptr(), rcx)?;

        // rdx contains the tape's start pointer
        self.asm.mov(self.tape_start_ptr(), rdx)?;

        // r8 contains the tape's end pointer
        self.asm.mov(self.tape_end_ptr(), r8)?;

        // // Push all non-volatile registers
        // for &register in NONVOLATILE_REGISTERS {
        //     let slot = self.push(register)?;
        //     self.non_volatile_registers.push((register, slot));
        // }

        Ok(())
    }

    /// Set up the function's epilog
    // FIXME: Needs to deallocate all of the live stack space
    //        as well as restore callee sided registers
    fn epilogue(&mut self) -> AsmResult<()> {
        self.set_label(self.epilogue);

        let stack_size = self.regalloc.free_stack();
        if stack_size != 0 {
            self.asm.add(rsp, stack_size as i32)?;
        }

        // Return from the function
        self.asm.ret()?;

        Ok(())
    }

    /// Get a pointer to the `*mut State` stored on the stack
    fn state_ptr(&self) -> AsmMemoryOperand {
        rsp + 8 + self.regalloc.virtual_rsp()
    }

    /// Get a pointer to the `*mut u8` stored on the stack
    fn tape_start_ptr(&self) -> AsmMemoryOperand {
        rsp + 16 + self.regalloc.virtual_rsp()
    }

    /// Get a pointer to the `*const u8` stored on the stack
    fn tape_end_ptr(&self) -> AsmMemoryOperand {
        rsp + 24 + self.regalloc.virtual_rsp()
    }

    fn stack_offset(&self, slot: StackSlot) -> AsmMemoryOperand {
        rsp + self.regalloc.slot_offset(slot)
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

    fn infinite_loop(&mut self) -> AsmResult<()> {
        // Add a nop to avoid having multiple labels for an instruction
        self.asm.nop()?;

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
fn low_register(register: AsmRegister64) -> AsmRegister8 {
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
