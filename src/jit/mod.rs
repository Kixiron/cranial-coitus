#[macro_use]
mod ffi;
mod disassemble;
mod memory;
mod regalloc;

use crate::{
    ir::{
        Add, Assign, Block, Call, Const, Eq, Expr, Gamma, Instruction, Pretty, PrettyConfig, Store,
        Theta, Value, VarId,
    },
    jit::{
        ffi::State,
        memory::CodeBuffer,
        regalloc::{Regalloc, StackSlot, NONVOLATILE_REGISTERS},
    },
    utils::{self, AssertNone},
};
use iced_x86::code_asm::{CodeAssembler, *};
use std::{
    borrow::Cow,
    collections::BTreeMap,
    io,
    panic::{self, AssertUnwindSafe},
};

type AsmResult<T> = Result<T, IcedError>;

const BITNESS: u32 = 64;

const RETURN_SUCCESS: i64 = 0;

const RETURN_IO_FAILURE: i64 = 101;

pub struct Jit {
    asm: CodeAssembler,
    io_failure: CodeLabel,
    epilogue: CodeLabel,
    has_io_functions: bool,
    tape_len: usize,
    values: BTreeMap<VarId, Operand>,
    regalloc: Regalloc,
    non_volatile_registers: Vec<(AsmRegister64, StackSlot)>,
    comments: BTreeMap<usize, Vec<String>>,
    named_labels: BTreeMap<usize, Cow<'static, str>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Operand {
    Register(AsmRegister64),
    Const(Const),
    Stack(StackSlot),
}

impl Jit {
    pub fn new(tape_len: usize) -> AsmResult<Self> {
        let mut asm = CodeAssembler::new(BITNESS)?;
        let io_failure = asm.create_label();
        let epilogue = asm.create_label();

        Ok(Self {
            asm,
            io_failure,
            epilogue,
            has_io_functions: false,
            tape_len,
            values: BTreeMap::new(),
            regalloc: Regalloc::new(),
            non_volatile_registers: Vec::new(),
            comments: BTreeMap::new(),
            named_labels: BTreeMap::new(),
        })
    }

    #[tracing::instrument(skip_all)]
    pub fn assemble(&mut self, block: &Block) -> AsmResult<String> {
        self.named_label(self.asm.instructions().len(), ".PROLOGUE");
        self.prologue()?;

        self.named_label(self.asm.instructions().len(), ".BODY");
        self.assemble_block(block)?;

        self.named_label(self.asm.instructions().len(), ".EPILOGUE");
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

            debug_assert!(code_buffer.len() >= code.len());
            code_buffer[..code.len()].copy_from_slice(&code);

            // TODO: Shrink code_buffer to the used size?
            code_buffer
        };

        let assembly = self.disassemble();
        println!("{}", assembly);
        std::fs::write(
            concat!(env!("CARGO_MANIFEST_DIR"), "/output.asm"),
            &assembly,
        )
        .unwrap();

        {
            let code_buffer = code_buffer.executable().unwrap();

            let mut tape = vec![0x00; self.tape_len];
            let tape_len = tape.len();

            let (stdin, stdout) = (io::stdin(), io::stdout());
            let mut state = State::new(stdin.lock(), stdout.lock());

            let jit_return = panic::catch_unwind(AssertUnwindSafe(|| unsafe {
                let start = tape.as_mut_ptr();
                let end = start.add(tape_len);

                println!(
                    "state = {}, start = {}, end = {}",
                    &state as *const _ as usize, start as usize, end as usize,
                );

                log_registers!();
                let value = code_buffer.call(&mut state, start, end);
                log_registers!();

                value
            }));

            println!("\njitted function returned {:?}", jit_return);
            println!("tape: {:?}", utils::debug_collapse(&tape));
        }

        Ok(assembly)
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
            Instruction::Store(store) => {
                self.add_comment(store.pretty_print(PrettyConfig::minimal()));
                self.assemble_store(store)
            }
            Instruction::LifetimeEnd(lifetime) => {
                self.add_comment(lifetime.pretty_print(PrettyConfig::minimal()));

                // if let Some(&value) = self.values.get(&lifetime.var) {
                //     match value {
                //         // FIXME: Register deallocation
                //         Operand::Register(register) => {} // self.deallocate_register(register),
                //
                //         Operand::Stack(slot) => {
                //             self.values.remove(&lifetime.var).unwrap();
                //             self.regalloc.free(&mut self.asm, slot)?;
                //         }
                //
                //         // Don't need to do anything special to "deallocate" a constant
                //         Operand::Const(_) => {
                //             self.values.remove(&lifetime.var).unwrap();
                //         }
                //     }
                // }

                Ok(())
            }
        }
    }

    fn assemble_call(&mut self, call: &Call) -> AsmResult<Option<AsmRegister64>> {
        match &*call.function {
            "input" => self.call_input(call).map(Some),
            "output" => self.call_output(call).map(|()| None),
            _ => unreachable!(),
        }
    }

    fn assemble_gamma(&mut self, gamma: &Gamma) -> AsmResult<()> {
        let after_gamma = self.create_label();

        match self.get_value(gamma.cond) {
            Operand::Register(register) => self.asm.cmp(register, 0)?,
            Operand::Const(constant) => self.asm.mov(al, constant.as_bool().unwrap() as i32)?,
            Operand::Stack(offset) => self.asm.cmp(self.stack_offset(offset), 0)?,
        }

        // If the true branch is empty and the false isn't
        if gamma.true_is_empty() && !gamma.false_is_empty() {
            // If the condition is true, skip the false branch's code
            self.asm.jz(after_gamma)?;

            // Build the false branch
            self.assemble_block(&gamma.false_branch)?;

        // If the true branch isn't empty and the false is
        } else if !gamma.true_is_empty() && gamma.false_is_empty() {
            // If the condition is false, skip the true branch's code
            self.asm.jnz(after_gamma)?;

            // Build the true branch
            self.assemble_block(&gamma.true_branch)?;

        // If both branches are full (TODO: if both branches are empty?)
        } else {
            // If the condition is false, jump to the false branch
            let false_branch = self.create_label();
            self.asm.jnz(false_branch)?;

            // Build the true branch
            self.assemble_block(&gamma.true_branch)?;
            // After we execute the true branch we jump to the code after the gamma
            self.asm.jmp(after_gamma)?;

            // Build the false branch
            self.set_label(false_branch);
            self.assemble_block(&gamma.false_branch)?;
        }

        self.set_label(after_gamma);
        Ok(())
    }

    fn assemble_theta(&mut self, theta: &Theta) -> AsmResult<()> {
        // A fix to keep us from making instructions with multiple labels
        self.asm.nop()?;

        // Create a label for the head of the theta's body
        let theta_head = self.create_label();
        self.set_label(theta_head);

        // Build the theta's body
        self.assemble_block(&theta.body)?;

        // Get the theta's condition
        let condition = self.get_value(theta.cond.unwrap());
        match condition {
            Operand::Register(register) => self.asm.cmp(register, 0)?,
            Operand::Const(constant) => self.asm.cmp(al, constant.as_bool().unwrap() as i32)?,
            Operand::Stack(offset) => self.asm.cmp(self.stack_offset(offset), 0i32)?,
        }

        self.regalloc.stack.free_vacant_slots();

        // If the condition is true, jump to the beginning of the theta's body
        self.asm.jz(theta_head)?;

        Ok(())
    }

    /// Invoke the input function to get a single byte from stdin
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

        // Save rax to rcx
        self.asm.mov(rcx, rax)?;

        // Take only the lowest bit of rcx in order to get the success code
        self.asm.and(rcx, 0x0000_0000_0000_0001)?;

        // Check if an IO error occurred
        self.asm.cmp(rcx, 0)?;

        // If so, jump to the io error label and exit
        self.asm.jnz(self.io_failure)?;

        // Shift right by one byte in order to keep only the input value
        self.asm.shr(rax, 8)?;

        // Otherwise rax holds the input value
        let input_reg = self.allocate_register()?;
        self.asm.mov(input_reg, rax)?;

        Ok(input_reg)
    }

    /// Invoke the output function to print a single byte to stdout
    #[allow(clippy::fn_to_numeric_cast)]
    fn call_output(&mut self, call: &Call) -> AsmResult<()> {
        debug_assert_eq!(call.function, "output");
        debug_assert_eq!(call.args.len(), 1);

        self.has_io_functions = true;

        // Move the state pointer into rcx
        self.asm.mov(rcx, self.state_ptr())?;

        // Move the given byte into rdx
        match call.args[0] {
            Value::Var(var) => match *self.values.get(&var).unwrap() {
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
        type_eq::<unsafe extern "win64" fn(state: *mut State, byte: u64) -> bool>(ffi::output);
        self.asm.mov(rax, ffi::output as u64)?;
        self.asm.call(rax)?;

        // Restore saved registers and deallocate stack space
        self.after_win64_call(stack_padding, slots)?;

        // Check if an IO error occurred
        self.asm.cmp(rax, 0)?;
        // If so, jump to the io error label and exit
        self.asm.jnz(self.io_failure)?;

        Ok(())
    }

    fn assemble_assign(&mut self, assign: &Assign) -> AsmResult<()> {
        self.add_comment(assign.pretty_print(PrettyConfig::minimal()));

        match &assign.value {
            Expr::Eq(eq) => {
                let register = self.assemble_eq(eq)?;
                self.values
                    .insert(assign.var, Operand::Register(register))
                    .debug_unwrap_none();

                Ok(())
            }

            Expr::Add(add) => {
                let register = self.assemble_add(add)?;
                self.values
                    .insert(assign.var, Operand::Register(register))
                    .debug_unwrap_none();

                Ok(())
            }

            Expr::Mul(_) => Ok(()),

            Expr::Not(not) => {
                let dest = self.allocate_register()?;

                // Get the input value
                match self.get_value(not.value) {
                    Operand::Register(register) => self.asm.mov(dest, register)?,
                    Operand::Stack(offset) => self.asm.mov(dest, self.stack_offset(offset))?,
                    Operand::Const(constant) => {
                        let value = constant.convert_to_i32().unwrap();
                        self.asm.mov(dest, value as i64)?;
                    }
                }

                // Perform a logical not on it
                self.asm.not(dest)?;

                self.values
                    .insert(assign.var, Operand::Register(dest))
                    .debug_unwrap_none();

                Ok(())
            }

            Expr::Neg(neg) => {
                let dest = self.allocate_register()?;

                // Get the input value
                match self.get_value(neg.value) {
                    Operand::Register(register) => self.asm.mov(dest, register)?,
                    Operand::Stack(offset) => self.asm.mov(dest, self.stack_offset(offset))?,
                    Operand::Const(constant) => {
                        let value = constant.convert_to_i32().unwrap();
                        self.asm.mov(dest, value as i64)?;
                    }
                }

                // Perform twos compliant negation on it
                self.asm.neg(dest)?;

                self.values
                    .insert(assign.var, Operand::Register(dest))
                    .debug_unwrap_none();

                Ok(())
            }

            Expr::Load(load) => {
                let destination = self.allocate_register()?;

                // Get the start pointer from the stack
                self.asm.mov(rax, self.tape_start_ptr())?;

                // Offset the tape pointer
                match self.get_value(load.ptr) {
                    // FIXME: Subtraction??
                    Operand::Register(register) => self.asm.add(rax, register)?,
                    // FIXME: Subtraction??
                    Operand::Stack(offset) => self.asm.add(rax, self.stack_offset(offset))?,

                    Operand::Const(constant) => {
                        let offset = constant.convert_to_i32().unwrap();

                        if offset != 0 {
                            // FIXME: Get the actual tape pointer's register
                            if offset.is_negative() {
                                self.asm.sub(rax, offset.abs())?;
                            } else {
                                self.asm.add(rax, offset)?;
                            }
                        }
                    }
                }

                // Dereference the pointer and store its value in the destination register
                self.asm.mov(destination, byte_ptr(rax))?;

                self.values
                    .insert(assign.var, Operand::Register(destination))
                    .debug_unwrap_none();

                Ok(())
            }

            Expr::Call(call) => {
                let input_reg = self.assemble_call(call)?.unwrap();
                self.values
                    .insert(assign.var, Operand::Register(input_reg))
                    .debug_unwrap_none();

                Ok(())
            }

            Expr::Value(value) => {
                match *value {
                    Value::Var(var) => match *self.values.get(&var).unwrap() {
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
                    },

                    Value::Const(constant) => {
                        self.values
                            .insert(assign.var, Operand::Const(constant))
                            .debug_unwrap_none();
                    }

                    Value::Missing => todo!(),
                }

                Ok(())
            }
        }
    }

    fn assemble_store(&mut self, store: &Store) -> AsmResult<()> {
        // Get the start pointer from the stack
        self.asm.mov(rax, self.tape_start_ptr())?;

        // Offset the tape pointer
        match self.get_value(store.ptr) {
            // FIXME: Subtraction??
            Operand::Register(register) => self.asm.add(rax, register)?,

            // FIXME: Subtraction??
            Operand::Stack(offset) => self.asm.add(rax, self.stack_offset(offset))?,

            Operand::Const(constant) => {
                let offset = constant.convert_to_i32().unwrap();

                if offset != 0 {
                    if offset.is_negative() {
                        self.asm.sub(rax, offset.abs())?;
                    } else {
                        self.asm.add(rax, offset)?;
                    }
                }
            }
        }

        // Store the given value to the given pointer
        match self.get_value(store.value) {
            Operand::Register(register) => self.asm.mov(byte_ptr(rax), register)?,

            Operand::Stack(offset) => {
                let temp = self.allocate_register()?;

                self.asm.mov(temp, self.stack_offset(offset))?;
                self.asm.mov(byte_ptr(rax), temp)?;

                self.deallocate_register(temp);
            }

            Operand::Const(value) => self
                .asm
                .mov(byte_ptr(rax), value.convert_to_u8().unwrap() as i32)?,
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
                let rhs = rhs.convert_to_i32().unwrap();

                let dest = self.allocate_register()?;
                self.asm.mov(dest, lhs)?;

                if rhs != 0 {
                    match rhs {
                        0 => unreachable!(),

                        1 => self.asm.inc(dest)?,
                        -1 => self.asm.dec(dest)?,

                        // Positive numbers turn into an add
                        rhs if rhs > 0 => self.asm.add(dest, rhs)?,

                        // Negative numbers turn into subtraction
                        rhs => self.asm.sub(dest, rhs.abs())?,
                    }
                }

                Ok(dest)
            }

            (Operand::Const(lhs), Operand::Register(rhs)) => {
                let dest = self.allocate_register()?;
                let lhs = lhs.convert_to_i32().unwrap();

                if lhs == 0 {
                    self.asm.mov(dest, rhs)?;
                    Ok(dest)
                } else {
                    self.asm.mov(dest, lhs as i64)?;
                    self.asm.add(dest, rhs)?;

                    Ok(dest)
                }
            }

            (Operand::Const(lhs), Operand::Const(rhs)) => {
                let (lhs, rhs) = (lhs.convert_to_i32().unwrap(), rhs.convert_to_i32().unwrap());
                let sum = lhs + rhs;

                let dest = self.allocate_register()?;
                self.asm.mov(dest, sum as i64)?;

                Ok(dest)
            }

            (Operand::Register(_), Operand::Stack(_)) => todo!(),
            (Operand::Const(_), Operand::Stack(_)) => todo!(),
            (Operand::Stack(_), Operand::Register(_)) => todo!(),

            (Operand::Stack(offset), Operand::Const(rhs)) => {
                let rhs = rhs.convert_to_i32().unwrap();

                let dest = self.allocate_register()?;
                self.asm.mov(dest, self.stack_offset(offset))?;

                if rhs != 0 {
                    match rhs {
                        0 => unreachable!(),

                        1 => self.asm.inc(dest)?,
                        -1 => self.asm.dec(dest)?,

                        // Positive numbers turn into an add
                        rhs if rhs > 0 => self.asm.add(dest, rhs)?,

                        // Negative numbers turn into subtraction
                        rhs => self.asm.sub(dest, rhs.abs())?,
                    }
                }

                Ok(dest)
            }

            (Operand::Stack(_), Operand::Stack(_)) => todo!(),
        }
    }

    fn stack_offset(&self, slot: StackSlot) -> AsmMemoryOperand {
        rsp + self.regalloc.slot_offset(slot)
    }

    // FIXME: It's inefficient to always store comparisons in registers when we could
    //        just keep them in al 99% of the time
    fn assemble_eq(&mut self, eq: &Eq) -> AsmResult<AsmRegister64> {
        let dest = self.allocate_register()?;

        // FIXME: There's opportunities for register reuse here as well
        match (self.get_value(eq.lhs), self.get_value(eq.rhs)) {
            (Operand::Register(lhs), Operand::Register(rhs)) => {
                self.asm.cmp(lhs, rhs)?;

                // Move the comparison result from al into the allocated
                // register with a zero sign extension
                self.asm.movzx(dest, al)?;
            }

            (Operand::Register(lhs), Operand::Const(rhs)) => {
                self.asm.cmp(lhs, rhs.convert_to_i32().unwrap())?;

                // Move the comparison result from al into the allocated
                // register with a zero sign extension
                self.asm.movzx(dest, al)?;
            }

            (Operand::Const(lhs), Operand::Register(rhs)) => {
                self.asm.mov(dest, lhs.convert_to_i32().unwrap() as i64)?;
                self.asm.cmp(dest, rhs)?;

                // Move the comparison result from al into the allocated
                // register with a zero sign extension
                self.asm.movzx(dest, al)?;
            }

            (Operand::Const(lhs), Operand::Const(rhs)) => {
                let are_equal = lhs.convert_to_i32().unwrap() == rhs.convert_to_i32().unwrap();
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

    fn get_value(&mut self, value: Value) -> Operand {
        match value {
            Value::Var(var) => *self.values.get(&var).unwrap(),

            Value::Const(constant) => Operand::Const(constant),

            Value::Missing => todo!(),
        }
    }

    /// Create the IO failure block
    #[allow(clippy::fn_to_numeric_cast)]
    fn build_io_failure(&mut self) -> AsmResult<()> {
        self.set_label(self.io_failure);

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

        // Jump to the function's epilogue
        self.asm.jmp(self.epilogue)?;

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
    fn epilogue(&mut self) -> AsmResult<()> {
        self.asm.nop()?;
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

    // /// Push a register's value to the stack
    // fn push(&mut self, register: AsmRegister64) -> AsmResult<StackSlot> {
    //     let slot = StackSlot::new(self.regalloc.virtual_rsp(), 8);

    //     self.asm.push(register)?;
    //     self.regalloc.stack.virtual_rsp += 8;

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
    fn named_label<N>(&mut self, index: usize, name: N)
    where
        N: Into<Cow<'static, str>>,
    {
        self.named_labels
            .insert(index, name.into())
            .debug_unwrap_none();
    }

    #[track_caller]
    fn set_label(&mut self, mut label: CodeLabel) {
        self.asm.set_label(&mut label).unwrap();
    }
}

fn type_eq<T>(_: T) {}
