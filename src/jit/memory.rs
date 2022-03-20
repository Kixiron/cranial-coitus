use crate::jit::{ffi::State, JitFunction};
use anyhow::{Context, Result};
use cranelift_jit::JITModule;
use std::{
    io::{self, Write},
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
        let mut state = State::new(stdin.lock(), stdout.lock(), &mut utf8_buffer);
        state
            .stdout
            .flush()
            .context("failed to flush stdout prior to jitted function execution")?;

        let jit_start = Instant::now();
        let jit_return = panic::catch_unwind(AssertUnwindSafe(|| unsafe {
            let start = tape.as_mut_ptr();
            let end = start.add(tape_len);

            self.call(&mut state, start, end)
        }))
        .map_err(|err| anyhow::anyhow!("jitted function panicked: {:?}", err));
        let elapsed = jit_start.elapsed();

        writeln!(state.stdout).context("failed to write newline to stdout")?;
        state
            .stdout
            .flush()
            .context("failed to flush stdout after jitted function execution")?;
        drop(state);

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
