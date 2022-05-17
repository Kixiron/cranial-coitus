use crate::jit::{ffi::State, JitFunction};
use anyhow::{Context, Result};
use cranelift_jit::JITModule;
use std::{
    io::{self, Read, Write},
    panic::{self, AssertUnwindSafe},
    time::Instant,
};

pub struct Executable {
    func: JitFunction,
    module: Option<JITModule>,
}

impl Executable {
    pub(super) fn new(func: JitFunction, module: JITModule) -> Self {
        Self {
            func,
            module: Some(module),
        }
    }

    pub unsafe fn call(&self, state: *mut State, start: *mut u8, end: *mut u8) -> u8 {
        unsafe { (self.func)(state, start, end) }
    }

    pub unsafe fn execute(&self, tape: &mut [u8]) -> Result<u8> {
        let tape_len = tape.len();

        let (stdin, stdout, mut utf8_buffer) =
            (io::stdin(), io::stdout(), String::with_capacity(512));

        let start_ptr = tape.as_mut_ptr();
        let end_ptr = unsafe { start_ptr.add(tape_len) };

        let (mut stdin, mut stdout) = (stdin.lock(), stdout.lock());
        stdout
            .flush()
            .context("failed to flush stdout before jitted function execution")?;

        let mut state = State::new(
            &mut stdin,
            &mut stdout,
            &mut utf8_buffer,
            start_ptr.cast(),
            end_ptr.cast(),
        );

        let jit_start = Instant::now();
        let jit_return = panic::catch_unwind(AssertUnwindSafe(|| unsafe {
            self.call(&mut state, start_ptr, end_ptr)
        }))
        .map_err(|err| anyhow::anyhow!("jitted function panicked: {:?}", err));
        let elapsed = jit_start.elapsed();
        drop(stdin);

        stdout
            .flush()
            .context("failed to flush stdout after jitted function execution")?;
        drop(stdout);

        tracing::debug!(
            "jitted function returned {:?} in {:#?}",
            jit_return,
            elapsed,
        );

        jit_return
    }

    pub unsafe fn execute_into(
        &self,
        tape: &mut [u8],
        stdin: &mut dyn Read,
        stdout: &mut dyn Write,
    ) -> Result<u8> {
        let tape_len = tape.len();

        let mut utf8_buffer = String::with_capacity(512);

        let start_ptr = tape.as_mut_ptr();
        let end_ptr = unsafe { start_ptr.add(tape_len) };

        let mut state = State::new(
            stdin,
            stdout,
            &mut utf8_buffer,
            start_ptr.cast(),
            end_ptr.cast(),
        );

        let jit_start = Instant::now();
        let jit_return = panic::catch_unwind(AssertUnwindSafe(|| unsafe {
            self.call(&mut state, start_ptr, end_ptr)
        }))
        .map_err(|err| anyhow::anyhow!("jitted function panicked: {:?}", err));
        let elapsed = jit_start.elapsed();

        tracing::debug!(
            "jitted function returned {:?} in {:#?}",
            jit_return,
            elapsed,
        );

        jit_return
    }
}

impl Drop for Executable {
    fn drop(&mut self) {
        if let Some(module) = self.module.take() {
            unsafe { module.free_memory() };
        }
    }
}
