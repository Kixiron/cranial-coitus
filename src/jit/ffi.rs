use std::{
    ascii,
    io::{Read, StdinLock, StdoutLock, Write},
    panic::{self, AssertUnwindSafe},
    slice,
};

const IO_FAILURE_MESSAGE: &[u8] = b"encountered an io failure during execution";

pub(super) struct State<'a> {
    stdin: StdinLock<'a>,
    stdout: StdoutLock<'a>,
}

impl<'a> State<'a> {
    pub(super) fn new(stdin: StdinLock<'a>, stdout: StdoutLock<'a>) -> Self {
        Self { stdin, stdout }
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

/// Returns a `u16` where the first byte is the input value and the second
/// byte is a 1 upon IO failure and a 0 upon success
pub(super) unsafe extern "win64" fn input(state: *mut State) -> u16 {
    log_registers!();
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

pub(super) unsafe extern "win64" fn output(state: *mut State, byte: u64) -> bool {
    log_registers!();
    // println!("state = {}, byte = {}", state as usize, byte);

    let byte = byte as u8;

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

pub(super) unsafe extern "win64" fn io_error_encountered(state: *mut State) -> bool {
    log_registers!();
    println!("state = {}", state as usize);

    let state = &mut *state;

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
