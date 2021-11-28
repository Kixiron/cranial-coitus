use crate::{
    ir::{
        Add, Assign, Block, Call, Const, Eq, Expr, Instruction, Pretty, PrettyConfig, Store, Value,
        VarId,
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
    collections::BTreeMap,
    io::{self, Read, StdinLock, StdoutLock, Write},
    mem::transmute,
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

const BITNESS: u32 = 64;

const RETURN_SUCCESS: i64 = 0;

const RETURN_IO_FAILURE: i64 = 101;

pub struct Codegen {
    asm: CodeAssembler,
    io_failure: CodeLabel,
    has_io_functions: bool,
    tape_len: usize,
    values: BTreeMap<VarId, Immediate>,
    registers: Vec<(AsmRegister64, Option<()>)>,
    comments: BTreeMap<usize, Vec<String>>,
}

#[derive(Debug, Clone, Copy)]
enum Immediate {
    Register(AsmRegister64),
    Const(Const),
    // TODO: Stack values
}

impl Codegen {
    pub fn new(tape_len: usize) -> Self {
        let mut asm = CodeAssembler::new(BITNESS).unwrap();
        let io_failure = asm.create_label();

        Self {
            asm,
            io_failure,
            has_io_functions: false,
            tape_len,
            values: BTreeMap::new(),
            registers: vec![
                (r8, None),
                (r9, None),
                (r10, None),
                (r11, None),
                (r12, None),
                (r13, None),
                (r14, None),
                (r15, None),
            ],
            comments: BTreeMap::new(),
        }
    }

    pub fn assemble(&mut self, block: &Block) -> Result<(), IcedError> {
        let prologue_start = self.asm.instructions().len();
        self.prologue()?;

        let body_start = self.asm.instructions().len();
        for inst in &**block {
            self.assemble_inst(inst)?;
        }

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

        Ok(())
    }

    fn assemble_inst(&mut self, inst: &Instruction) -> Result<(), IcedError> {
        match inst {
            Instruction::Call(call) => {
                self.add_comment(call.pretty_print(PrettyConfig::minimal()));
                self.assemble_call(call)?.debug_unwrap_none();

                Ok(())
            }

            Instruction::Assign(assign) => self.assemble_assign(assign),

            Instruction::Theta(theta) => Ok(()),

            Instruction::Gamma(gamma) => Ok(()),

            Instruction::Store(store) => {
                self.add_comment(store.pretty_print(PrettyConfig::minimal()));
                self.assemble_store(store)
            }
        }
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

    #[allow(clippy::fn_to_numeric_cast)]
    fn assemble_call(&mut self, call: &Call) -> Result<Option<AsmRegister64>, IcedError> {
        match &*call.function {
            "input" => {
                debug_assert!(call.args.is_empty());
                self.has_io_functions = true;

                // Move the state pointer into rcx
                self.asm.mov(rcx, rsp + 8)?;

                // Reserve stack space for the passed argument
                self.asm.sub(rsp, 40)?;

                for reg in self
                    .registers
                    .iter()
                    .filter_map(|&(reg, occupied)| occupied.map(|()| reg))
                {
                    self.asm.push(reg)?;
                }

                // Call the input function
                type_eq::<unsafe extern "win64" fn(state: *mut State) -> u16>(input);
                self.asm.mov(rax, input as u64)?;
                self.asm.call(rax)?;

                for reg in self
                    .registers
                    .iter()
                    .filter_map(|&(reg, occupied)| occupied.map(|()| reg))
                    .rev()
                {
                    self.asm.pop(reg)?;
                }

                // Deallocate the argument stack space
                self.asm.add(rsp, 40)?;

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
                let input_reg = self.allocate_register();
                self.asm.mov(input_reg, rax)?;

                Ok(Some(input_reg))
            }

            "output" => {
                debug_assert_eq!(call.args.len(), 1);
                self.has_io_functions = true;

                // Move the state pointer into rcx
                self.asm.mov(rcx, rsp + 8)?;

                // Move the given byte into rdx
                match call.args[0] {
                    Value::Var(var) => match *self
                        .values
                        .get(&var)
                        .unwrap_or(&Immediate::Const(Const::U8(0xFF)))
                    {
                        Immediate::Register(reg) => self.asm.mov(rdx, reg)?,
                        Immediate::Const(constant) => self
                            .asm
                            .mov(rdx, constant.convert_to_u8().unwrap() as u64)?,
                    },

                    Value::Const(constant) => self
                        .asm
                        .mov(rdx, constant.convert_to_u8().unwrap() as u64)?,

                    Value::Missing => todo!(),
                };

                // Reserve stack space for the passed arguments
                self.asm.sub(rsp, 40)?;

                for reg in self
                    .registers
                    .iter()
                    .filter_map(|&(reg, occupied)| occupied.map(|()| reg))
                {
                    self.asm.push(reg)?;
                }

                // Call the output function
                type_eq::<unsafe extern "win64" fn(state: *mut State, byte: u8) -> bool>(output);
                self.asm.mov(rax, output as u64)?;
                self.asm.call(rax)?;

                for reg in self
                    .registers
                    .iter()
                    .filter_map(|&(reg, occupied)| occupied.map(|()| reg))
                    .rev()
                {
                    self.asm.pop(reg)?;
                }

                // Deallocate the argument stack space
                self.asm.add(rsp, 40)?;

                // Check if an IO error occurred
                self.asm.cmp(rax, 0)?;
                // If so, jump to the io error label and exit
                self.asm.jnz(self.io_failure)?;

                Ok(None)
            }

            _ => todo!(),
        }
    }

    fn assemble_assign(&mut self, assign: &Assign) -> Result<(), IcedError> {
        self.add_comment(assign.pretty_print(PrettyConfig::minimal()));

        match &assign.value {
            Expr::Eq(eq) => self.eq(eq),

            Expr::Add(add) => {
                let add_reg = self.assemble_add(add)?;
                self.values
                    .insert(assign.var, Immediate::Register(add_reg))
                    .debug_unwrap_none();

                Ok(())
            }

            Expr::Mul(_) => Ok(()),
            Expr::Not(_) => Ok(()),
            Expr::Neg(_) => Ok(()),
            Expr::Load(load) => {
                let destination = self.allocate_register();

                // Get the start pointer from the stack
                self.asm.mov(rax, rsp + 16)?;

                // Offset the tape pointer
                let ptr = match self.get_value(load.ptr) {
                    Immediate::Register(reg) => {
                        // FIXME: Subtract from register??
                        // FIXME: Get the actual tape pointer's register
                        self.asm.add(rax, reg)?;

                        byte_ptr(rax)
                    }

                    Immediate::Const(constant) => {
                        let offset = constant.convert_to_i32().unwrap();

                        if offset != 0 {
                            // FIXME: Get the actual tape pointer's register
                            if offset.is_negative() {
                                self.asm.sub(rax, offset.abs())?;
                            } else {
                                self.asm.add(rax, offset)?;
                            }
                        }

                        byte_ptr(rax)
                    }
                };

                // Dereference the pointer and store its value in the destination register
                self.asm.mov(destination, ptr)?;

                self.values
                    .insert(assign.var, Immediate::Register(destination))
                    .debug_unwrap_none();

                Ok(())
            }

            Expr::Call(call) => {
                let input_reg = self.assemble_call(call)?.unwrap();
                self.values
                    .insert(assign.var, Immediate::Register(input_reg))
                    .debug_unwrap_none();

                Ok(())
            }

            Expr::Value(value) => {
                match value {
                    &Value::Var(var) => {
                        self.values
                            .insert(assign.var, *self.values.get(&var).unwrap())
                            .debug_unwrap_none();
                    }
                    &Value::Const(constant) => {
                        self.values
                            .insert(assign.var, Immediate::Const(constant))
                            .debug_unwrap_none();
                    }

                    Value::Missing => todo!(),
                }

                Ok(())
            }
        }
    }

    /// Set up the function's prologue
    fn prologue(&mut self) -> Result<(), IcedError> {
        // rcx contains the state pointer
        self.asm.mov(rsp + 8, rcx)?;

        // rdx contains the tape's start pointer
        self.asm.mov(rsp + 16, rdx)?;

        // r8 contains the tape's end pointer
        self.asm.mov(rsp + 24, r8)?;

        Ok(())
    }

    /// Set up the function's epilog
    fn epilogue(&mut self) -> Result<(), IcedError> {
        // Return zero as the return code
        self.asm.mov(rax, RETURN_SUCCESS)?;

        // Return from the function
        self.asm.ret()?;

        Ok(())
    }

    #[allow(clippy::fn_to_numeric_cast)]
    fn build_io_failure(&mut self) -> Result<(), IcedError> {
        self.asm.set_label(&mut self.io_failure)?;

        // Move the state pointer into the rcx register
        self.asm.mov(rcx, rsp + 8)?;

        // Reserve stack space for the state pointer
        self.asm.sub(rsp, 40)?;

        // Call the io error function
        type_eq::<unsafe extern "win64" fn(*mut State) -> bool>(io_error_encountered);
        self.asm.mov(rax, io_error_encountered as u64)?;
        self.asm.call(rax)?;

        // Deallocate the parameter stack space
        self.asm.add(rsp, 40)?;

        // Move the io error code into the rax register to be used as the exit code
        self.asm.mov(rax, RETURN_IO_FAILURE)?;

        // Return from the function
        self.asm.ret()?;

        Ok(())
    }

    fn assemble_store(&mut self, store: &Store) -> Result<(), IcedError> {
        // Get the start pointer from the stack
        self.asm.mov(rax, rsp + 16)?;

        // Offset the tape pointer
        let ptr = match self.get_value(store.ptr) {
            Immediate::Register(reg) => {
                // FIXME: Subtract from register??
                // FIXME: Get the actual tape pointer's register
                self.asm.add(rax, reg)?;

                byte_ptr(rax)
            }

            Immediate::Const(constant) => {
                let offset = constant.convert_to_i32().unwrap();

                if offset != 0 {
                    if offset.is_negative() {
                        self.asm.sub(rax, offset.abs())?;
                    } else {
                        self.asm.add(rax, offset)?;
                    }
                }

                byte_ptr(rax)
            }
        };

        // Store the given value to the given pointer
        match self.get_value(store.value) {
            Immediate::Register(value) => self.asm.mov(ptr, value)?,
            Immediate::Const(value) => self.asm.mov(ptr, value.convert_to_u8().unwrap() as i32)?,
        }

        Ok(())
    }

    fn get_value(&mut self, value: Value) -> Immediate {
        match value {
            Value::Var(var) => *self
                .values
                .get(&var)
                .unwrap_or(&Immediate::Const(Const::U8(0xFF))),
            Value::Const(constant) => Immediate::Const(constant),
            Value::Missing => todo!(),
        }
    }

    fn assemble_add(&mut self, add: &Add) -> Result<AsmRegister64, IcedError> {
        let dest = self.allocate_register();

        // FIXME: These could actually be optimized a lot with register reuse
        match (self.get_value(add.lhs), self.get_value(add.rhs)) {
            (Immediate::Register(lhs), Immediate::Register(rhs)) => {
                self.asm.mov(dest, lhs)?;
                // FIXME: Subtracting registers?
                self.asm.add(dest, rhs)?;
            }

            (Immediate::Register(lhs), Immediate::Const(rhs)) => {
                self.asm.mov(dest, lhs)?;

                let rhs = rhs.convert_to_i32().unwrap();
                if rhs.is_negative() {
                    self.asm.sub(dest, rhs.abs())?;
                } else {
                    self.asm.add(dest, rhs)?;
                }
            }

            (Immediate::Const(lhs), Immediate::Register(rhs)) => {
                self.asm.mov(dest, lhs.convert_to_i32().unwrap() as i64)?;
                self.asm.add(dest, rhs)?;
            }

            (Immediate::Const(lhs), Immediate::Const(rhs)) => {
                self.asm.mov(dest, lhs.convert_to_i32().unwrap() as i64)?;

                let rhs = rhs.convert_to_i32().unwrap();
                if rhs.is_negative() {
                    self.asm.sub(dest, rhs.abs())?;
                } else {
                    self.asm.add(dest, rhs)?;
                }
            }
        }

        Ok(dest)
    }

    fn eq(&mut self, eq: &Eq) -> Result<(), IcedError> {
        // FIXME: There's opportunities for register reuse here as well
        match (self.get_value(eq.lhs), self.get_value(eq.rhs)) {
            (Immediate::Register(lhs), Immediate::Register(rhs)) => self.asm.cmp(lhs, rhs),

            (Immediate::Register(lhs), Immediate::Const(rhs)) => {
                self.asm.cmp(lhs, rhs.convert_to_i32().unwrap())
            }

            (Immediate::Const(lhs), Immediate::Register(rhs)) => {
                let lhs_reg = self.allocate_register();
                self.asm
                    .mov(lhs_reg, lhs.convert_to_i32().unwrap() as i64)?;
                self.asm.cmp(lhs_reg, rhs)?;
                self.deallocate_register(lhs_reg);

                Ok(())
            }

            (Immediate::Const(lhs), Immediate::Const(rhs)) => {
                let lhs_reg = self.allocate_register();
                self.asm
                    .mov(lhs_reg, lhs.convert_to_i32().unwrap() as i64)?;
                self.asm.cmp(lhs_reg, rhs.convert_to_i32().unwrap())?;
                self.deallocate_register(lhs_reg);

                Ok(())
            }
        }
    }

    // TODO: Stack spilling
    fn allocate_register(&mut self) -> AsmRegister64 {
        self.registers
            .iter_mut()
            .find_map(|(register, used)| {
                if used.is_none() {
                    *used = Some(());
                    Some(*register)
                } else {
                    None
                }
            })
            .unwrap()
    }

    fn deallocate_register(&mut self, register: AsmRegister64) {
        for (reg, used) in &mut self.registers {
            if *reg == register {
                *used = None;
                return;
            }
        }
    }

    fn infinite_loop(&mut self) -> Result<(), IcedError> {
        let mut label = self.asm.create_label();
        self.asm.set_label(&mut label)?;
        self.asm.jmp(label)?;

        Ok(())
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
    log_registers!();
    println!("state = {}", state as usize);

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
    log_registers!();
    println!("state = {}", state as usize);

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

    u16::from_be_bytes([value, failed as u8])
}

unsafe extern "win64" fn output(state: *mut State, byte: u8) -> bool {
    log_registers!();
    println!("state = {}, byte = {}", state as usize, byte);

    let state = &mut *state;
    let output_panicked = panic::catch_unwind(AssertUnwindSafe(|| {
        let write_result = if byte.is_ascii() {
            state.stdout.write_all(&[byte])
        } else {
            let escape = ascii::escape_default(byte);
            write!(&mut state.stdout, "{}", escape)
        };

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
