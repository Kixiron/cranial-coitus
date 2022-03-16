use std::{
    ascii,
    io::{Read, StdinLock, StdoutLock, Write},
    panic::{self, AssertUnwindSafe},
    slice,
};

const IO_FAILURE_MESSAGE: &[u8] = b"encountered an io failure during execution";

pub struct State<'a> {
    stdin: StdinLock<'a>,
    pub(super) stdout: StdoutLock<'a>,
}

impl<'a> State<'a> {
    pub fn new(stdin: StdinLock<'a>, stdout: StdoutLock<'a>) -> Self {
        Self { stdin, stdout }
    }
}

#[allow(unused_macros)]
macro_rules! log_registers {
    () => {
        #[allow(unused_assignments, unused_variables)]
        {
            let (
                mut rcx_val,
                mut rdx_val,
                mut r8_val,
                mut r9_val,
                mut rax_val,
                mut rsp_val,
            ): (u64, u64, u64, u64, u64, u64);

            ::std::arch::asm!(
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

            ::std::println!(
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
        }
    };
}

/// Returns a bool with the input call's status, writing to `input` if successful
///
/// If the function returns `true`, the contents of `input` are undefined.
/// If the function returns `false`, the location pointed to by `input` will
/// contain the input value
pub(super) unsafe extern "fastcall" fn input(state: *mut State, input: *mut u8) -> bool {
    debug_assert!(!state.is_null());
    debug_assert!(!input.is_null());

    let (state, input) = unsafe { (&mut *state, &mut *input) };
    panic::catch_unwind(AssertUnwindSafe(|| {
        // Flush stdout
        if let Err(err) = state.stdout.flush() {
            tracing::error!("failed to flush stdout while getting byte: {:?}", err);
            return true;
        }

        // Read one byte from stdin
        if let Err(err) = state.stdin.read_exact(slice::from_mut(input)) {
            tracing::error!("getting byte from stdin failed: {:?}", err);
            return true;
        }

        false
    }))
    .unwrap_or_else(|err| {
        tracing::error!("getting byte from stdin panicked: {:?}", err);
        true
    })
}

pub(super) unsafe extern "fastcall" fn output(state: *mut State, byte: u8) -> bool {
    debug_assert!(!state.is_null());

    let state = unsafe { &mut *state };
    panic::catch_unwind(AssertUnwindSafe(|| {
        let write_result = if byte.is_ascii() {
            state.stdout.write_all(&[byte])
        } else {
            let escape = ascii::escape_default(byte);
            write!(state.stdout, "{}", escape)
        };

        match write_result {
            Ok(()) => false,
            Err(err) => {
                tracing::error!("writing byte to stdout failed: {:?}", err);
                true
            }
        }
    }))
    .unwrap_or_else(|err| {
        tracing::error!("writing byte to stdout panicked: {:?}", err);
        true
    })
}

pub(super) unsafe extern "fastcall" fn io_error(state: *mut State) -> bool {
    debug_assert!(!state.is_null());

    let state = unsafe { &mut *state };
    let io_failure_panicked = panic::catch_unwind(AssertUnwindSafe(|| {
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
