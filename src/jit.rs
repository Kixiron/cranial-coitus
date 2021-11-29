use crate::{
    ir::{
        Add, Assign, AssignTag, Block, Call, Const, Eq, Expr, Gamma, Instruction, Pretty,
        PrettyConfig, Store, Theta, Value, VarId,
    },
    utils::{self, AssertNone},
};
use core::slice;
use iced_x86::{
    code_asm::{CodeAssembler, *},
    FlowControl, Formatter, Instruction as X86Instruction, MasmFormatter, SymbolResolver,
    SymbolResult,
};
use std::{
    ascii,
    collections::{BTreeMap, VecDeque},
    io::{self, Read, StdinLock, StdoutLock, Write},
    mem::{size_of, transmute},
    ops::{Deref, DerefMut},
    panic::{self, AssertUnwindSafe},
    ptr::{self, NonNull},
};
use winapi::um::{
    errhandlingapi::GetLastError,
    handleapi::CloseHandle,
    memoryapi::{VirtualAlloc, VirtualFree, VirtualProtect},
    processthreadsapi::{FlushInstructionCache, GetCurrentProcess},
    winnt::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE, PAGE_READWRITE},
};

type AsmResult<T> = Result<T, IcedError>;

const BITNESS: u32 = 64;

const RETURN_SUCCESS: i64 = 0;

const RETURN_IO_FAILURE: i64 = 101;

const NONVOLATILE_REGISTERS: &[AsmRegister64] = &[rbx, rbp, rdi, rsi, rsp, r12, r13, r14, r15];

const VOLATILE_REGISTERS: &[AsmRegister64] = &[rax, rcx, rdx, r8, r9, r10, r11];

struct RegisterLru {
    cache: VecDeque<AsmRegister64>,
    length: usize,
}

impl RegisterLru {
    pub fn new(length: usize) -> Self {
        Self {
            cache: VecDeque::with_capacity(length),
            length,
        }
    }

    pub fn push(&mut self, register: AsmRegister64) -> Option<AsmRegister64> {
        // If the register is already in the cache, move it to the front
        if let Some(idx) = self.cache.iter().position(|&reg| reg == register) {
            if idx != 0 {
                self.cache.remove(idx).debug_unwrap();
                self.cache.push_front(register);
            }

            None

        // If the cache has space, push it to the front
        } else if self.cache.len() < self.length {
            self.cache.push_front(register);
            None

        // Otherwise evict the least recently used register and push the given
        // register to the front of the cache
        } else {
            let evicted = self.cache.pop_back().unwrap();
            self.cache.push_front(register);

            Some(evicted)
        }
    }

    pub fn pop(&mut self, register: AsmRegister64) {
        // If the register exists within the cache, evict it
        if let Some(idx) = self.cache.iter().position(|&reg| reg == register) {
            self.cache.remove(idx).debug_unwrap();
        }
    }

    pub fn evict(&mut self) -> AsmRegister64 {
        self.cache.pop_back().unwrap()
    }
}

pub struct Codegen {
    asm: CodeAssembler,
    io_failure: CodeLabel,
    has_io_functions: bool,
    tape_len: usize,
    values: BTreeMap<VarId, Operand>,
    registers: Vec<(AsmRegister64, Option<()>)>,
    register_lru: RegisterLru,
    comments: BTreeMap<usize, Vec<String>>,
    stack_top: usize,
    stack_bottom: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Operand {
    Register(AsmRegister64),
    Const(Const),
    Stack(usize),
}

impl Codegen {
    pub fn new(tape_len: usize) -> AsmResult<Self> {
        let mut asm = CodeAssembler::new(BITNESS)?;
        let io_failure = asm.create_label();

        let registers = vec![
            (r8, None),
            (r9, None),
            (r10, None),
            (r11, None),
            (r12, None),
            (r13, None),
            (r14, None),
            (r15, None),
        ];
        let register_lru = RegisterLru::new(registers.len());

        Ok(Self {
            asm,
            io_failure,
            has_io_functions: false,
            tape_len,
            values: BTreeMap::new(),
            registers,
            register_lru,
            comments: BTreeMap::new(),
            stack_top: 0,
            stack_bottom: 32,
        })
    }

    pub fn assemble(&mut self, block: &Block) -> AsmResult<String> {
        let prologue_start = self.asm.instructions().len();
        self.prologue()?;

        let body_start = self.asm.instructions().len();
        self.assemble_block(block)?;

        let epilogue_start = self.asm.instructions().len();
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

        let assembly = {
            let mut formatter = MasmFormatter::with_options(Some(Box::new(Resolver)), None);

            // Set up the formatter's options
            {
                let options = formatter.options_mut();

                // Set up hex formatting
                options.set_uppercase_hex(true);
                options.set_hex_prefix("0x");
                options.set_hex_suffix("");

                // Make operand formatting pretty
                options.set_space_after_operand_separator(true);
                options.set_space_between_memory_add_operators(true);
                options.set_space_between_memory_mul_operators(true);
                options.set_scale_before_index(false);
            }

            // Collect all jump targets
            let mut labels = Vec::new();

            for inst in self.asm.instructions() {
                if matches!(
                    inst.flow_control(),
                    FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch,
                ) {
                    labels.push(inst.near_branch_target());
                }
            }

            // Sort and deduplicate the jump targets
            labels.sort_unstable();
            labels.dedup();

            // Name each jump target in increasing order
            let labels: BTreeMap<u64, String> = labels
                .into_iter()
                .enumerate()
                .map(|(idx, address)| (address, format!("LBL_{}", idx)))
                .collect();

            // Format all instructions
            let (mut output, mut is_indented) = (String::new(), false);
            for (idx, inst) in self.asm.instructions().iter().enumerate() {
                let address = inst.ip();

                // Create pseudo labels for points of interest
                let mut pseudo_label = None;
                if idx == prologue_start {
                    pseudo_label = Some(".PROLOGUE:\n");
                } else if idx == body_start {
                    pseudo_label = Some(".BODY:\n");
                } else if idx == epilogue_start {
                    pseudo_label = Some(".EPILOGUE:\n");
                }

                if let Some(label) = pseudo_label {
                    is_indented = true;

                    if idx != 0 {
                        output.push('\n');
                    }
                    output.push_str(label);
                }

                // If the current address is jumped to, add a label to the output text
                if let Some(label) = labels.get(&address) {
                    is_indented = true;

                    if idx != 0
                        && idx != prologue_start
                        && idx != body_start
                        && idx != epilogue_start
                    {
                        output.push('\n');
                    }

                    output.push('.');
                    output.push_str(label);
                    output.push_str(":\n");
                }

                // Display any comments
                if let Some(comments) = self.comments.get(&idx) {
                    for comment in comments {
                        if is_indented {
                            output.push_str("  ");
                        }

                        output.push_str("; ");
                        output.push_str(comment);
                        output.push('\n');
                    }
                }

                // Indent the line if needed
                if is_indented {
                    output.push_str("  ");
                }

                // If this is a branch instruction we want to replace the branch address with
                // our human readable label
                if matches!(
                    inst.flow_control(),
                    FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch,
                ) {
                    // Use the label name if we can find it
                    if let Some(label) = labels.get(&inst.near_branch_target()) {
                        let mnemonic = format!("{:?}", inst.mnemonic()).to_lowercase();
                        output.push_str(&mnemonic);
                        output.push_str(" .");
                        output.push_str(label);

                    // Otherwise fall back to normal formatting
                    } else {
                        tracing::warn!(
                            "failed to get branch label for {} (address: {:#x})",
                            inst,
                            inst.near_branch_target(),
                        );

                        formatter.format(inst, &mut output);
                    }

                // Otherwise format the instruction into the output buffer
                } else {
                    formatter.format(inst, &mut output);
                }

                // Add a newline between each instruction (and a trailing one)
                output.push('\n');
            }

            println!("{}", output);
            output
        };

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

                code_buffer.call(&mut state, start, end)
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

                if let Some(&value) = self.values.get(&lifetime.var) {
                    match value {
                        Operand::Register(register) => self.deallocate_register(register),
                        Operand::Stack(_) => {
                            // TODO: Deallocate from stack
                        }
                        Operand::Const(_) => {}
                    }
                }

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
        let mut after_gamma = self.create_label();

        match self.get_value(gamma.cond) {
            Operand::Register(register) => self.asm.cmp(register, 0)?,
            Operand::Const(constant) => self.asm.mov(al, constant.as_bool().unwrap() as i32)?,
            Operand::Stack(offset) => self
                .asm
                .cmp(rsp + (self.stack_top - (self.stack_bottom + offset)), 0)?,
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
            let mut false_branch = self.create_label();
            self.asm.jnz(false_branch)?;

            // Build the true branch
            self.assemble_block(&gamma.true_branch)?;
            // After we execute the true branch we jump to the code after the gamma
            self.asm.jmp(after_gamma)?;

            // Build the false branch
            self.asm.set_label(&mut false_branch).unwrap();
            self.assemble_block(&gamma.false_branch)?;
        }

        self.asm.set_label(&mut after_gamma).unwrap();
        Ok(())
    }

    fn assemble_theta(&mut self, theta: &Theta) -> AsmResult<()> {
        // A fix to keep us from making instructions with multiple labels
        self.asm.nop()?;

        // Create a label for the head of the theta's body
        let mut theta_head = self.create_label();
        self.asm.set_label(&mut theta_head).unwrap();

        // Build the theta's body
        self.assemble_block(&theta.body)?;

        // Get the theta's condition
        let condition = self.get_value(theta.cond.unwrap());
        match condition {
            Operand::Register(register) => self.asm.cmp(register, 0)?,
            Operand::Const(constant) => self.asm.mov(al, constant.as_bool().unwrap() as i32)?,
            Operand::Stack(offset) => self
                .asm
                .cmp(rsp + (self.stack_top - (self.stack_bottom + offset)), 0)?,
        }

        // If the condition is true, jump to the beginning of the theta's body
        self.asm.jz(theta_head)?;

        // Deallocate all registers & stack space from the theta's body
        for inst in &theta.body {
            if let &Instruction::Assign(Assign {
                var,
                // TODO: What to do with `AssignTag::InputParam(_)`?
                tag: AssignTag::None,
                ..
            }) = inst
            {
                match self.get_value(Value::Var(var)) {
                    Operand::Register(register) => self.deallocate_register(register),
                    Operand::Stack(_) => {
                        // TODO: Deallocate stack slot?
                    }
                    Operand::Const(_) => {}
                }

                self.values.remove(&var).debug_unwrap();
            }
        }

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
        let stack_allocation = self.before_win64_call()?;

        // Call the input function
        type_eq::<unsafe extern "win64" fn(state: *mut State) -> u16>(input);
        self.asm.mov(rax, input as u64)?;
        self.asm.call(rax)?;

        // Restore saved registers and deallocate stack space
        self.after_win64_call(stack_allocation)?;

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
                Operand::Stack(offset) => self
                    .asm
                    .mov(rdx, rsp + (self.stack_top - (self.stack_bottom + offset)))?,
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
        let stack_allocation = self.before_win64_call()?;

        // Call the output function
        type_eq::<unsafe extern "win64" fn(state: *mut State, byte: u8) -> bool>(output);
        self.asm.mov(rax, output as u64)?;
        self.asm.call(rax)?;

        // Restore saved registers and deallocate stack space
        self.after_win64_call(stack_allocation)?;

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
                    Operand::Stack(offset) => self
                        .asm
                        .mov(dest, rsp + (self.stack_top - (self.stack_bottom + offset)))?,
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
                    Operand::Stack(offset) => self
                        .asm
                        .mov(dest, rsp + (self.stack_top - (self.stack_bottom + offset)))?,
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
                    Operand::Stack(offset) => self
                        .asm
                        .add(rax, rsp + (self.stack_top - (self.stack_bottom + offset)))?,

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
                match value {
                    &Value::Var(var) => {
                        self.values
                            .insert(assign.var, *self.values.get(&var).unwrap());
                    }
                    &Value::Const(constant) => {
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
            Operand::Stack(offset) => self
                .asm
                .add(rax, rsp + (self.stack_top - (self.stack_bottom + offset)))?,

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

                self.asm
                    .mov(temp, rsp + (self.stack_top - (self.stack_bottom + offset)))?;
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
                if rhs == 0 {
                    Ok(lhs)
                } else {
                    let dest = self.allocate_register()?;
                    self.asm.mov(dest, lhs)?;

                    match rhs {
                        0 => unreachable!(),

                        1 => self.asm.inc(dest)?,
                        -1 => self.asm.dec(dest)?,

                        // Positive numbers turn into an add
                        rhs if rhs > 0 => self.asm.add(dest, rhs)?,

                        // Negative numbers turn into subtraction
                        rhs => self.asm.sub(dest, rhs.abs())?,
                    }

                    Ok(dest)
                }
            }

            (Operand::Const(lhs), Operand::Register(rhs)) => {
                let lhs = lhs.convert_to_i32().unwrap();
                if lhs == 0 {
                    Ok(rhs)
                } else {
                    let dest = self.allocate_register()?;

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
            (Operand::Stack(_), Operand::Const(_)) => todo!(),
            (Operand::Stack(_), Operand::Stack(_)) => todo!(),
        }
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
        self.asm.set_label(&mut self.io_failure).unwrap();

        // Move the state pointer into the rcx register
        self.asm.mov(rcx, self.state_ptr())?;

        // Setup the stack and save used registers
        let stack_allocation = self.before_win64_call()?;

        // Call the io error function
        type_eq::<unsafe extern "win64" fn(*mut State) -> bool>(io_error_encountered);
        self.asm.mov(rax, io_error_encountered as u64)?;
        self.asm.call(rax)?;

        // Restore saved registers and deallocate stack space
        self.after_win64_call(stack_allocation)?;

        // Move the io error code into the rax register to be used as the exit code
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

        // Push all non-volatile registers
        // for &register in NONVOLATILE_REGISTERS {
        //     self.push(register)?;
        // }

        Ok(())
    }

    /// Set up the function's epilog
    fn epilogue(&mut self) -> AsmResult<()> {
        // Pop all non-volatile registers
        // for &register in NONVOLATILE_REGISTERS.iter().rev() {
        //     self.pop(register)?;
        // }

        // Return zero as the return code
        self.asm.mov(rax, RETURN_SUCCESS)?;

        // Return from the function
        self.asm.ret()?;

        Ok(())
    }

    /// Get a pointer to the `*mut State` stored on the stack
    fn state_ptr(&self) -> AsmMemoryOperand {
        (rsp + 8) + self.stack_top
    }

    /// Get a pointer to the `*mut u8` stored on the stack
    fn tape_start_ptr(&self) -> AsmMemoryOperand {
        (rsp + 16) + self.stack_top
    }

    /// Get a pointer to the `*const u8` stored on the stack
    fn tape_end_ptr(&self) -> AsmMemoryOperand {
        (rsp + 24) + self.stack_top
    }

    /// Push a register's value to the stack
    fn push(&mut self, register: AsmRegister64) -> AsmResult<()> {
        self.stack_top += size_of::<u64>();
        self.asm.push(register)?;

        Ok(())
    }

    /// Pop a value from the stack and into a register
    fn pop(&mut self, register: AsmRegister64) -> AsmResult<()> {
        self.stack_top -= size_of::<u64>();
        self.asm.pop(register)?;

        Ok(())
    }

    // TODO: Stack spilling
    // TODO: Value lifetimes, when can we kill what?
    fn allocate_register(&mut self) -> AsmResult<AsmRegister64> {
        self.registers
            .iter_mut()
            .find_map(|(register, used)| {
                if used.is_none() {
                    self.register_lru.push(*register);
                    *used = Some(());

                    Some(Ok(*register))
                } else {
                    None
                }
            })
            // Spill to the stack
            .unwrap_or_else(|| {
                let register = self.register_lru.evict();
                for operand in self.values.values_mut() {
                    if *operand == Operand::Register(register) {
                        // self.push(register)?;
                        self.stack_top += size_of::<u64>();
                        self.asm.push(register)?;
                        *operand = Operand::Stack(self.stack_top - size_of::<u64>());

                        break;
                    }
                }

                Ok(register)
            })
    }

    fn deallocate_register(&mut self, register: AsmRegister64) {
        self.register_lru.pop(register);

        for (reg, used) in &mut self.registers {
            if *reg == register {
                *used = None;
                return;
            }
        }
    }

    fn infinite_loop(&mut self) -> AsmResult<()> {
        let mut label = self.asm.create_label();
        self.asm.set_label(&mut label).unwrap();
        self.asm.jmp(label)?;

        Ok(())
    }

    /// Collects all registers that are currently in-use and are volatile
    fn used_volatile_registers(&self) -> Vec<AsmRegister64> {
        self.registers
            .iter()
            // Filter out any unused registers
            .filter_map(|&(register, occupied)| occupied.map(|()| register))
            // We only want volatile registers, non-volatile registers must be preserved by the callee
            .filter(|register| VOLATILE_REGISTERS.contains(register))
            .collect()
    }

    /// Saves all currently used registers and allocates 32 bytes of stack for the called function
    /// as well as aligning the stack to 16 bytes. Returns the total extra stack space that was allocated
    /// for both the parameters and alignment padding
    fn before_win64_call(&mut self) -> AsmResult<i32> {
        // We start with an offset of 8 since `call` pushes the return address to the stack
        // https://www.felixcloutier.com/x86/call#operation
        let mut stack_offset = 8 + self.stack_top;

        // Push all currently used registers to the stack
        for register in self.used_volatile_registers() {
            self.asm.push(register)?;
            stack_offset += 8;
        }

        // Get the required padding to align the stack to 16 bytes
        // https://docs.microsoft.com/en-us/cpp/build/stack-usage?view=msvc-170#stack-allocation
        let stack_padding = (stack_offset + 32) % 16;

        // Reserve stack space for 4 arguments (must be allocated unconditionally for win64)
        // as well as the required padding to align the stack to 16 bytes
        // https://docs.microsoft.com/en-us/cpp/build/stack-usage?view=msvc-170#stack-allocation
        let stack_allocation = (32 + stack_padding) as i32;

        // Allocate the required space by subtracting from rsp
        self.asm.sub(rsp, stack_allocation)?;

        Ok(stack_allocation)
    }

    /// Restores all saved registers and deallocates the space that was allocated for the called function
    fn after_win64_call(&mut self, stack_allocation: i32) -> AsmResult<()> {
        // Deallocate the stack space we allocated for both the function arguments
        // and the padding required to align the stack to 16 bytes
        self.asm.add(rsp, stack_allocation)?;

        // Pop all used registers from the stack
        for register in self.used_volatile_registers().into_iter().rev() {
            self.asm.pop(register)?;
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
}

struct State<'a> {
    stdin: StdinLock<'a>,
    stdout: StdoutLock<'a>,
}

impl<'a> State<'a> {
    fn new(stdin: StdinLock<'a>, stdout: StdoutLock<'a>) -> Self {
        Self { stdin, stdout }
    }
}

struct Resolver;

impl SymbolResolver for Resolver {
    #[allow(clippy::fn_to_numeric_cast)]
    fn symbol(
        &mut self,
        _instruction: &X86Instruction,
        _operand: u32,
        _instruction_operand: Option<u32>,
        address: u64,
        _address_size: u32,
    ) -> Option<SymbolResult<'_>> {
        if address == io_error_encountered as u64 {
            Some(SymbolResult::with_str(address, "io_error_encountered"))
        } else if address == input as u64 {
            Some(SymbolResult::with_str(address, "input"))
        } else if address == output as u64 {
            Some(SymbolResult::with_str(address, "output"))
        } else {
            None
        }
    }
}

macro_rules! log_registers {
    () => {
        let (
            mut rcx_val,
            mut rdx_val,
            mut r8_val,
            mut r9_val,
            mut rax_val,
            mut rsp_val,
        ): (u64, u64, u64, u64, u64, u64);

        asm!(
            "mov {0}, rax",
            "mov {1}, rcx",
            "mov {2}, rdx",
            "mov {3}, r8",
            "mov {4}, r9",
            "mov {5}, rsp",
            out(reg) rax_val,
            out(reg) rcx_val,
            out(reg) rdx_val,
            out(reg) r8_val,
            out(reg) r9_val,
            out(reg) rsp_val,
            options(pure, nostack, readonly),
        );

        println!(
            "[{}:{}:{}]: rax = {}, rcx = {}, rdx = {}, r8 = {}, r9 = {}, rsp = {}",
            file!(),
            line!(),
            column!(),
            rax_val,
            rcx_val,
            rdx_val,
            r8_val,
            r9_val,
            rsp_val,
        );
    };
}

unsafe extern "win64" fn io_error_encountered(state: *mut State) -> bool {
    // log_registers!();
    // println!("state = {}", state as usize);

    let state = &mut *state;

    let io_failure_panicked = panic::catch_unwind(AssertUnwindSafe(|| {
        const IO_FAILURE_MESSAGE: &[u8] = b"encountered an io failure during execution";

        let write_failed = match state.stdout.write_all(IO_FAILURE_MESSAGE) {
            Ok(()) => false,
            Err(err) => {
                tracing::error!("failed to write to stdout during io failure: {:?}", err);
                true
            }
        };

        let flush_failed = match state.stdout.flush() {
            Ok(()) => false,
            Err(err) => {
                tracing::error!("failed to flush stdout during io failure: {:?}", err);
                true
            }
        };

        write_failed || flush_failed
    }));

    match io_failure_panicked {
        Ok(result) => result,
        Err(err) => {
            tracing::error!("ip failure panicked: {:?}", err);
            true
        }
    }
}

/// Returns a `u16` where the first byte is the input value and the second
/// byte is a 1 upon IO failure and a 0 upon success
unsafe extern "win64" fn input(state: *mut State) -> u16 {
    // log_registers!();
    // println!("state = {}", state as usize);

    let state = &mut *state;
    let mut value = 0;

    let input_panicked = panic::catch_unwind(AssertUnwindSafe(|| {
        // Flush stdout
        let flush_failed = match state.stdout.flush() {
            Ok(()) => false,
            Err(err) => {
                tracing::error!("failed to flush stdout while getting byte: {:?}", err);
                true
            }
        };

        // Read one byte from stdin
        let read_failed = match state.stdin.read_exact(slice::from_mut(&mut value)) {
            Ok(()) => false,
            Err(err) => {
                tracing::error!("getting byte from stdin failed: {:?}", err);
                true
            }
        };

        read_failed || flush_failed
    }));

    let failed = match input_panicked {
        Ok(result) => result,
        Err(err) => {
            tracing::error!("getting byte from stdin panicked: {:?}", err);
            true
        }
    };

    // println!("value = {}, failed = {}", value, failed);
    u16::from_be_bytes([value, failed as u8])
}

unsafe extern "win64" fn output(state: *mut State, byte: u8) -> bool {
    // log_registers!();
    // println!("state = {}, byte = {}", state as usize, byte);

    let state = &mut *state;
    let output_panicked = panic::catch_unwind(AssertUnwindSafe(|| {
        let write_result = if byte.is_ascii() {
            state.stdout.write_all(&[byte])
        } else {
            let escape = ascii::escape_default(byte);
            write!(&mut state.stdout, "{}", escape)
        };

        let _ = writeln!(&mut state.stdout, "output: {:#X}", byte);
        let _ = state.stdout.flush();

        match write_result {
            Ok(()) => false,
            Err(err) => {
                tracing::error!("writing byte to stdout failed: {:?}", err);
                true
            }
        }
    }));

    match output_panicked {
        Ok(result) => result,
        Err(err) => {
            tracing::error!("writing byte to stdout panicked: {:?}", err);
            true
        }
    }
}

fn type_eq<T>(_: T) {}

struct CodeBuffer {
    buffer: NonNull<[u8]>,
}

impl CodeBuffer {
    pub fn new(length: usize) -> Option<Self> {
        tracing::debug!("allocating a code buffer of {} bytes", length);

        if length == 0 {
            tracing::error!("tried to allocate code buffer of zero bytes");
            return None;
        }

        // Safety: VirtualAlloc allocates the requested memory initialized with zeroes
        let ptr = unsafe {
            VirtualAlloc(
                ptr::null_mut(),
                length,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };
        let ptr = NonNull::new(ptr.cast())?;
        let buffer = NonNull::slice_from_raw_parts(ptr, length);

        Some(Self { buffer })
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { self.buffer.as_ref() }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { self.buffer.as_mut() }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.buffer.as_ptr() as *const u8
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.buffer.as_mut_ptr() as *mut u8
    }

    pub fn executable(mut self) -> Option<Executable<Self>> {
        unsafe {
            // Give the page execute permissions
            let mut old_protection = 0;
            let protection_result = VirtualProtect(
                self.as_mut_ptr().cast(),
                self.len(),
                PAGE_EXECUTE,
                &mut old_protection,
            );

            if protection_result == 0 {
                let error_code = GetLastError();
                tracing::error!(
                    "failed to give code buffer PAGE_EXECUTE protection, error {}",
                    error_code,
                );

                return None;
            }

            // Get a pseudo handle to the current process
            let handle = GetCurrentProcess();

            // Flush the instruction cache
            let flush_result = FlushInstructionCache(handle, self.as_mut_ptr().cast(), self.len());

            // Closing the handle of the current process is a noop, but we do it anyways for correctness
            let close_handle_result = CloseHandle(handle);

            if flush_result == 0 {
                let error_code = GetLastError();
                tracing::error!("failed to flush instruction cache, error {}", error_code,);

                return None;
            } else if close_handle_result == 0 {
                let error_code = GetLastError();
                tracing::error!(
                    "failed to close handle to the current process, error {}",
                    error_code,
                );

                return None;
            }
        }

        Some(Executable::new(self))
    }
}

impl Deref for CodeBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for CodeBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Drop for CodeBuffer {
    fn drop(&mut self) {
        let free_result = unsafe { VirtualFree(self.as_mut_ptr().cast(), 0, MEM_RELEASE) };

        if free_result == 0 {
            tracing::error!(
                "failed to deallocate {} bytes at {:p} from jit",
                self.len(),
                self.as_mut_ptr(),
            );
        }
    }
}

#[repr(transparent)]
pub struct Executable<T>(T);

impl<T> Executable<T> {
    fn new(inner: T) -> Self {
        Self(inner)
    }
}

impl Executable<CodeBuffer> {
    pub unsafe fn call(&self, state: *mut State, start: *mut u8, end: *mut u8) -> u8 {
        let func: unsafe extern "win64" fn(*mut State, *mut u8, *const u8) -> u8 =
            transmute(self.0.as_ptr());

        func(state, start, end)
    }
}
