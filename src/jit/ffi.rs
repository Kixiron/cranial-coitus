use core::str::lossy::{Utf8Lossy, Utf8LossyChunk};
use std::{
    io::{Read, StdinLock, StdoutLock, Write},
    panic::{self, AssertUnwindSafe},
    slice,
};

const IO_FAILURE_MESSAGE: &[u8] = b"encountered an io failure during execution";

pub struct State<'a> {
    stdin: StdinLock<'a>,
    pub(super) stdout: StdoutLock<'a>,
    utf8_buffer: &'a mut String,
}

impl<'a> State<'a> {
    pub fn new(stdin: StdinLock<'a>, stdout: StdoutLock<'a>, utf8_buffer: &'a mut String) -> Self {
        Self {
            stdin,
            stdout,
            utf8_buffer,
        }
    }
}

/// Returns a bool with the input call's status, writing to `input` if successful
///
/// If the function returns `true`, the contents of `input` are undefined.
/// If the function returns `false`, the location pointed to by `input` will
/// contain the input value
///
/// # Safety
///
/// - `state` must be a valid pointer to a valid [`State`] instance
/// - `input` must be a valid pointer to a single (possibly uninitialized) byte
pub(super) unsafe extern "fastcall" fn input(state_ptr: *mut State, input_ptr: *mut u8) -> bool {
    panic::catch_unwind(AssertUnwindSafe(|| {
        if state_ptr.is_null() || input_ptr.is_null() {
            tracing::error!(?state_ptr, ?input_ptr, "input call got null pointer");
            return true;
        }
        let (state, input) = unsafe { (&mut *state_ptr, &mut *input_ptr) };

        // Flush stdout
        if let Err(error) = state.stdout.flush() {
            tracing::error!("failed to flush stdout during input call: {:?}", error);
            return true;
        }

        // Read one byte from stdin
        if let Err(error) = state.stdin.read_exact(slice::from_mut(input)) {
            tracing::error!("reading from stdin failed during input call: {:?}", error);
            return true;
        }

        false
    }))
    .unwrap_or_else(|error| {
        tracing::error!("input call panicked: {:?}", error);
        true
    })
}

pub(super) unsafe extern "fastcall" fn output(
    state_ptr: *mut State,
    bytes_ptr: *const u8,
    length: usize,
) -> bool {
    panic::catch_unwind(AssertUnwindSafe(|| {
        if state_ptr.is_null() || bytes_ptr.is_null() || length == 0 {
            tracing::error!(
                ?state_ptr,
                ?bytes_ptr,
                length,
                "input call got null pointer or empty string"
            );
            return true;
        }

        let state = unsafe { &mut *state_ptr };
        let bytes = unsafe { slice::from_raw_parts(bytes_ptr, length) };

        let utf8 = from_utf8_lossy_buffered(state.utf8_buffer, bytes);

        match state.stdout.write_all(utf8.as_bytes()) {
            Ok(()) => false,
            Err(err) => {
                tracing::error!("writing to stdout during output call failed: {:?}", err);
                true
            }
        }
    }))
    .unwrap_or_else(|err| {
        tracing::error!("writing byte to stdout panicked: {:?}", err);
        true
    })
}

fn from_utf8_lossy_buffered<'a>(buffer: &'a mut String, bytes: &'a [u8]) -> &'a str {
    let mut iter = Utf8Lossy::from_bytes(bytes).chunks();

    let first_valid = if let Some(chunk) = iter.next() {
        let Utf8LossyChunk { valid, broken } = chunk;
        if broken.is_empty() {
            debug_assert_eq!(valid.len(), bytes.len());
            return valid;
        }
        valid
    } else {
        return "";
    };

    const REPLACEMENT: &str = "\u{FFFD}";

    buffer.clear();
    buffer.reserve(bytes.len());
    buffer.push_str(first_valid);
    buffer.push_str(REPLACEMENT);

    for Utf8LossyChunk { valid, broken } in iter {
        buffer.push_str(valid);
        if !broken.is_empty() {
            buffer.push_str(REPLACEMENT);
        }
    }

    buffer
}

pub(super) unsafe extern "fastcall" fn io_error(state_ptr: *mut State) -> bool {
    panic::catch_unwind(AssertUnwindSafe(|| {
        if state_ptr.is_null() {
            tracing::error!(?state_ptr, "io_error call got null state pointer");
            return true;
        }
        let state = unsafe { &mut *state_ptr };

        // Write the io error message to stdout
        if let Err(error) = state.stdout.write_all(IO_FAILURE_MESSAGE) {
            tracing::error!(
                "failed to write to stdout during io_error call: {:?}",
                error,
            );
            return true;
        }

        // Flush stdout
        if let Err(error) = state.stdout.flush() {
            tracing::error!("failed to flush stdout during io_error call: {:?}", error);
            return true;
        }

        false
    }))
    .unwrap_or_else(|error| {
        tracing::error!("io_error panicked: {:?}", error);
        true
    })
}
