use crate::ir::{Assign, Block, Call, Expr, Instruction, Value};
use core::slice;
use iced_x86::{
    code_asm::{CodeAssembler, *},
    Decoder, DecoderOptions, Formatter, Instruction as X86Instruction, MasmFormatter,
    SymbolResolver, SymbolResult,
};
use std::{
    ascii,
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
        }
    }

    pub fn assemble(&mut self, block: &Block) -> Result<(), IcedError> {
        self.prologue()?;

        for inst in &**block {
            self.assemble_inst(inst)?;
        }

        self.epilogue()?;

        // Only build the IO handler if there's IO functions
        if self.has_io_functions {
            build_io_failure(&mut self.asm, &mut self.io_failure)?;
        }

        let (code, code_buffer) = {
            // The maximum size of an instruction is 15 bytes, so we allocate the most memory we could possibly use
            let maximum_possible_size = self.asm.instructions().len() * 15;
            let mut code_buffer = CodeBuffer::new(maximum_possible_size).unwrap();

            let code = self.asm.assemble(code_buffer.as_ptr() as u64)?;

            debug_assert!(code_buffer.len() >= code.len());
            code_buffer[..code.len()].copy_from_slice(&code);

            // TODO: Shrink code_buffer to the used size?
            (code, code_buffer)
        };

        let disassembled = {
            let mut decoder = Decoder::new(BITNESS, &code, DecoderOptions::NONE);
            let mut formatter = MasmFormatter::with_options(Some(Box::new(Resolver)), None);

            {
                let options = formatter.options_mut();

                options.set_uppercase_hex(true);
                options.set_hex_prefix("0x");
                options.set_hex_suffix("");

                options.set_space_after_operand_separator(true);
                options.set_space_between_memory_add_operators(true);
                options.set_space_between_memory_mul_operators(true);
                options.set_scale_before_index(false);
            }

            let mut output = String::new();
            let mut inst = X86Instruction::default();

            while decoder.can_decode() {
                // Decode the instruction
                decoder.decode_out(&mut inst);

                // Format the instruction into the output buffer
                formatter.format(&inst, &mut output);
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
                    "state = {}, current = {}, start = {}, end = {}",
                    &state as *const _ as usize, start as usize, start as usize, end as usize,
                );

                code_buffer.call(&mut state, start, start, end)
            }));

            println!("\njitted function returned {:?}", jit_return);
        }

        Ok(())
    }

    fn assemble_inst(&mut self, inst: &Instruction) -> Result<(), IcedError> {
        match inst {
            Instruction::Call(call) => self.assemble_call(call),
            Instruction::Assign(assign) => self.assemble_assign(assign),
            Instruction::Theta(theta) => Ok(()),
            Instruction::Gamma(gamma) => Ok(()),
            Instruction::Store(store) => Ok(()),
        }
    }

    #[allow(clippy::fn_to_numeric_cast)]
    fn assemble_call(&mut self, call: &Call) -> Result<(), IcedError> {
        match &*call.function {
            "input" => {
                debug_assert!(call.args.is_empty());
                self.has_io_functions = true;

                self.asm.mov(rsp + 0x38, rdx)?;

                // Call the input function
                type_eq::<unsafe extern "win64" fn(state: *mut State) -> u16>(input);
                self.asm.mov(rax, input as u64)?;
                self.asm.call(rax)?;

                // Take args off the stack
                self.asm.mov(rcx, rsp + 0x30)?;
                self.asm.mov(rdx, rsp + 0x38)?;
                self.asm.mov(r8, rsp + 0x40)?;
                self.asm.mov(r9, rsp + 0x48)?;

                // `input()` returns a u16 within the rax register where the top six bytes are garbage, the
                // 7th byte is the input value (if the function succeeded) and the 8th byte is the status
                // code, 1 for error and 0 for success

                // Save rax to r15
                self.asm.mov(rax, r15)?;

                // Take only the lowest bit of r15 in order to get the success code
                self.asm.and(r15, 0x0000_0000_0000_0001)?;
                // Shift right by one byte in order to keep only the input value
                self.asm.shr(rax, 8)?;

                // Check if an IO error occurred
                self.asm.cmp(r15, 0)?;

                // If so, jump to the io error label and exit
                self.asm.jnz(self.io_failure)?;

                // Otherwise rax holds the input value
            }

            "output" => {
                debug_assert_eq!(call.args.len(), 1);

                self.has_io_functions = true;

                let arg = &call.args[0];
                let value = match *arg {
                    Value::Var(var) => 0,
                    Value::Const(constant) => constant.convert_to_u8().unwrap(),
                    Value::Missing => todo!(),
                };

                self.asm.mov(rsp + 0x38, rdx)?;

                // Call the output function
                type_eq::<unsafe extern "win64" fn(state: *mut State, byte: u8) -> bool>(output);
                self.asm.mov(rax, output as u64)?;
                self.asm.call(rax)?;

                // Take args off the stack
                self.asm.mov(rcx, rsp + 0x30)?;
                self.asm.mov(rdx, rsp + 0x38)?;
                self.asm.mov(r8, rsp + 0x40)?;
                self.asm.mov(r9, rsp + 0x48)?;

                // Check if an IO error occurred
                self.asm.cmp(al, 0)?;
                // If so, jump to the io error label and exit
                self.asm.jnz(self.io_failure)?;
            }

            _ => todo!(),
        }

        Ok(())
    }

    fn assemble_assign(&mut self, assign: &Assign) -> Result<(), IcedError> {
        match &assign.value {
            Expr::Eq(_) => Ok(()),
            Expr::Add(_) => Ok(()),
            Expr::Mul(_) => Ok(()),
            Expr::Not(_) => Ok(()),
            Expr::Neg(_) => Ok(()),
            Expr::Load(_) => Ok(()),
            Expr::Call(_) => Ok(()),
            Expr::Value(_) => Ok(()),
        }
    }

    /// Set up the function's epilog
    fn epilogue(&mut self) -> Result<(), IcedError> {
        // Return the callee registers
        self.asm.pop(rdx)?;
        self.asm.pop(r8)?;
        self.asm.pop(r9)?;

        // Return zero as the return code
        self.asm.mov(rax, RETURN_SUCCESS)?;

        // Return from the function
        self.asm.ret()?;

        Ok(())
    }

    /// Set up the function's prologue
    fn prologue(&mut self) -> Result<(), IcedError> {
        // Save callee registers
        self.asm.mov(rsp + 8, rcx)?;
        self.asm.push(rdx)?;
        self.asm.push(r8)?;
        self.asm.push(r9)?;

        Ok(())
    }
}

#[allow(clippy::fn_to_numeric_cast)]
fn build_io_failure(asm: &mut CodeAssembler, io_failure: &mut CodeLabel) -> Result<(), IcedError> {
    asm.set_label(io_failure)?;

    // Set up the stack for a function call
    asm.mov(rax, io_error_encountered as u64)?;

    // Call the io error function
    asm.call(rax)?;

    // Pop callee registers
    asm.mov(rcx, rsp + 0x30)?;
    asm.mov(rdx, rsp + 0x38)?;
    asm.mov(r8, rsp + 0x40)?;
    asm.mov(r9, rsp + 0x48)?;

    // Move the io error code into the rax register to be used as the exit code
    asm.mov(rax, RETURN_IO_FAILURE)?;
    asm.add(rsp, 0x28)?;

    // Return from the function
    asm.ret()?;

    Ok(())
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

unsafe extern "win64" fn io_error_encountered(state: *mut State) -> bool {
    panic::catch_unwind(|| {
        let state = &mut *state;

        let mut errored = state
            .stdout
            .write_all(b"encountered an io failure during execution")
            .is_err();
        errored |= state.stdout.flush().is_err();
        errored
    })
    .unwrap_or(true)
}

/// Returns a `u16` where the first byte is the input value and the second
/// byte is a 1 upon IO failure and a 0 upon success
unsafe extern "win64" fn input(state: *mut State) -> u16 {
    let state = &mut *state;
    let mut value = 0;

    let status = panic::catch_unwind(AssertUnwindSafe(|| {
        let flush_error = state.stdout.flush().is_err();
        state.stdin.read_exact(slice::from_mut(&mut value)).is_err() || flush_error
    }))
    .unwrap_or(true);

    u16::from_ne_bytes([value, status as u8])
}

unsafe extern "win64" fn output(state: *mut State, byte: u8) -> bool {
    let (mut rcx_val, mut rdx_val, mut r8_val, mut r9_val, mut rax_val): (u64, u64, u64, u64, u64);
    asm!(
        "mov rcx, {0}",
        "mov rdx, {1}",
        "mov r8, {2}",
        "mov r9, {3}",
        "mov rax, {4}",
        out(reg) rcx_val,
        out(reg) rdx_val,
        out(reg) r8_val,
        out(reg) r9_val,
        out(reg) rax_val,
        options(pure, nostack, readonly),
    );

    println!(
        "rax = {}, rcx = {}, rdx = {}, r8 = {}, r9 = {}",
        rax_val, rcx_val, rdx_val, r8_val, r9_val,
    );

    panic::catch_unwind(|| {
        let state = &mut *state;

        if byte.is_ascii() {
            state.stdout.write_all(&[byte]).is_err()
        } else {
            let escape = ascii::escape_default(byte);
            write!(&mut state.stdout, "{}", escape).is_err()
        }
    })
    .unwrap_or(true)
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
    pub unsafe fn call(
        &self,
        state: *mut State,
        current: *mut u8,
        start: *mut u8,
        end: *mut u8,
    ) -> u8 {
        let func: unsafe extern "win64" fn(*mut State, *mut u8, *mut u8, *const u8) -> u8 =
            transmute(self.0.as_ptr());

        func(state, current, start, end)
    }
}
