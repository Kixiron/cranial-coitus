use core::str::lossy::{Utf8Lossy, Utf8LossyChunk};
use std::{
    io::{Read, Write},
    panic::{self, AssertUnwindSafe},
    slice,
};

const IO_FAILURE_MESSAGE: &[u8] = b"encountered an io failure during execution\n";

pub struct State<'a> {
    stdin: &'a mut dyn Read,
    stdout: &'a mut dyn Write,
    utf8_buffer: &'a mut String,
    start_ptr: *const u8,
    end_ptr: *const u8,
    stdout_flushed: bool,
}

impl<'a> State<'a> {
    pub fn new(
        stdin: &'a mut dyn Read,
        stdout: &'a mut dyn Write,
        utf8_buffer: &'a mut String,
        start_ptr: *const u8,
        end_ptr: *const u8,
    ) -> Self {
        Self {
            stdin,
            stdout,
            utf8_buffer,
            start_ptr,
            end_ptr,
            stdout_flushed: true,
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
        if !state.stdout_flushed {
            state.stdout_flushed = true;

            if let Err(error) = state.stdout.flush() {
                tracing::error!("failed to flush stdout during input call: {:?}", error);
                return true;
            }
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
        let result = match state.stdout.write_all(utf8.as_bytes()) {
            Ok(()) => false,
            Err(err) => {
                tracing::error!("writing to stdout during output call failed: {:?}", err);
                true
            }
        };
        state.utf8_buffer.clear();
        state.stdout_flushed = false;

        result
    }))
    .unwrap_or_else(|err| {
        tracing::error!("writing byte to stdout panicked: {:?}", err);
        true
    })
}

/// Performs a right scan over the program tape for the given `needle` value
///
/// The scan starts at `tape_offset` and progresses to the end of the tape before
/// wrapping around to the beginning of the tape. The `step` value decides
/// which cells are checked, a step of 1 will cause every cell to be checked while
/// a step of 2 will cause every *other* cell to be checked.
///
/// The function returns [`u32::MAX`] as an error value. An error will be returned
/// if `step` is zero or another unforeseen error occurs. Additionally, if the current
/// tape doesn't contain the `needle` value or the given `step` value doesn't allow it
/// to ever reach it, the function will loop infinitely.
///
/// # Safety
///
/// `state_ptr` must be a valid pointer to a [`State`] instance and `tape_offset`
/// must be inbounds of the current program tape.
///
pub(super) unsafe extern "fastcall" fn scanr_wrapping(
    state_ptr: *const State,
    tape_offset: u16,
    step: u16,
    needle: u8,
) -> u32 {
    panic::catch_unwind(AssertUnwindSafe(|| {
        if state_ptr.is_null() {
            tracing::error!(
                ?state_ptr,
                ?tape_offset,
                ?step,
                ?needle,
                "scanr_wrapping call got null state pointer"
            );
            return u32::MAX;
        } else if step == 0 {
            tracing::error!(
                ?state_ptr,
                ?tape_offset,
                ?step,
                ?needle,
                "scanr_wrapping called with a step of zero"
            );
            return u32::MAX;
        }

        unsafe {
            let (tape_start, tape_end) = ((*state_ptr).start_ptr, (*state_ptr).end_ptr);

            if step == 1 {
                // Initial scan over the tape's suffix
                let mut ptr = tape_start.add(tape_offset as usize);
                while ptr < tape_end {
                    if *ptr == needle {
                        return (ptr as usize - tape_start as usize) as u32;
                    }

                    ptr = ptr.add(1);
                }

                // If we couldn't find it within the suffix, scan the entire tape until
                // we find it (or loop infinitely)
                loop {
                    let mut ptr = tape_start;
                    while ptr < tape_end {
                        if *ptr == needle {
                            return (ptr as usize - tape_start as usize) as u32;
                        }

                        ptr = ptr.add(1);
                    }
                }

            // For non-one steps, fall back to our crappy impl
            } else {
                let mut ptr = tape_start.add(tape_offset as usize);
                loop {
                    if ptr < tape_end {
                        if *ptr == needle {
                            return (ptr as usize - tape_start as usize) as u32;
                        }

                        ptr = ptr.add(step as usize);
                    } else {
                        ptr = tape_start.add(tape_end as usize - ptr as usize);
                    }
                }
            }
        }

        // unsafe {
        //     let (base_ptr, tape_len) = ((*state_ptr).start_ptr, (*state_ptr).tape_len);
        //     let mut ptr = Ptr::new(tape_offset, tape_len as u16);
        //
        //     loop {
        //         let current = base_ptr.add(ptr.value() as usize);
        //         if *current == needle {
        //             break ptr.value() as u32;
        //         } else {
        //             ptr = ptr.wrapping_add_u16(step);
        //         }
        //     }
        // }
    }))
    .unwrap_or_else(|error| {
        tracing::error!(
            ?state_ptr,
            ?tape_offset,
            ?step,
            ?needle,
            "scanr_wrapping panicked: {:?}",
            error,
        );
        u32::MAX
    })
}

/// Performs a right scan over the program tape for the given `needle` value
///
/// The scan starts at `tape_offset` and progresses to the end of the tape.
/// The `step` value decides which cells are checked, a step of 1 will cause
/// every cell to be checked while a step of 2 will cause every *other* cell
/// to be checked.
///
/// The function returns [`u32::MAX`] as an error value. An error will be returned
/// if `step` is zero or another unforeseen error occurs. Additionally, if the current
/// tape doesn't contain the `needle` value or the given `step` value doesn't allow it
/// to ever reach it, the function will loop infinitely.
///
/// # Safety
///
/// `state_ptr` must be a valid pointer to a [`State`] instance and `tape_offset`
/// must be inbounds of the current program tape.
///
/// If the value is not contained within the scanned portion (`tape[tape_offset..]`)
/// this function causes UB.
///
pub(super) unsafe extern "fastcall" fn scanr_non_wrapping(
    state_ptr: *const State,
    tape_offset: u16,
    step: u16,
    needle: u8,
) -> u32 {
    panic::catch_unwind(AssertUnwindSafe(|| {
        if state_ptr.is_null() {
            tracing::error!(
                ?state_ptr,
                ?tape_offset,
                ?step,
                ?needle,
                "scanr_non_wrapping call got null state pointer"
            );
            return u32::MAX;
        } else if step == 0 {
            tracing::error!(
                ?state_ptr,
                ?tape_offset,
                ?step,
                ?needle,
                "scanr_non_wrapping called with a step of zero"
            );
            return u32::MAX;
        }

        unsafe {
            let tape_start = (*state_ptr).start_ptr;
            let mut ptr = tape_start.add(tape_offset as usize);

            while ptr >= tape_start {
                if *ptr == needle {
                    return (ptr as usize - tape_start as usize) as u32;
                }

                ptr = ptr.add(step as usize);
            }

            // TODO: We could also just use `unreachable_unchecked()` here
            u32::MAX
        }
    }))
    .unwrap_or_else(|error| {
        tracing::error!(
            ?state_ptr,
            ?tape_offset,
            ?step,
            ?needle,
            "scanr_non_wrapping panicked: {:?}",
            error,
        );
        u32::MAX
    })
}

/// Performs a left scan over the program tape for the given `needle` value
///
/// The scan starts at `tape_offset` and progresses to the beginning of the tape before
/// wrapping around to the end of the tape. The `step` value decides
/// which cells are checked, a step of 1 will cause every cell to be checked while
/// a step of 2 will cause every *other* cell to be checked.
///
/// The function returns [`u32::MAX`] as an error value. An error will be returned
/// if `step` is zero or another unforeseen error occurs. Additionally, if the current
/// tape doesn't contain the `needle` value or the given `step` value doesn't allow it
/// to ever reach it, the function will loop infinitely.
///
/// # Safety
///
/// `state_ptr` must be a valid pointer to a [`State`] instance and `tape_offset`
/// must be inbounds of the current program tape.
///
pub(super) unsafe extern "fastcall" fn scanl_wrapping(
    state_ptr: *const State,
    tape_offset: u16,
    step: u16,
    needle: u8,
) -> u32 {
    panic::catch_unwind(AssertUnwindSafe(|| {
        if state_ptr.is_null() {
            tracing::error!(
                ?state_ptr,
                ?tape_offset,
                ?step,
                ?needle,
                "scanl_wrapping call got null state pointer"
            );
            return u32::MAX;
        } else if step == 0 {
            tracing::error!(
                ?state_ptr,
                ?tape_offset,
                ?step,
                ?needle,
                "scanl_wrapping called with a step of zero"
            );
            return u32::MAX;
        }

        unsafe {
            let (tape_start, tape_end) = ((*state_ptr).start_ptr, (*state_ptr).end_ptr);

            if step == 1 {
                // Initial scan over the tape's prefix
                let mut ptr = tape_start.add(tape_offset as usize);
                while ptr >= tape_start {
                    if *ptr == needle {
                        return (ptr as usize - tape_start as usize) as u32;
                    }

                    ptr = ptr.sub(1);
                }

                // If we couldn't find it within the prefix, scan the entire tape until
                // we find it (or loop infinitely)
                loop {
                    let mut ptr = tape_end;
                    while ptr >= tape_start {
                        if *ptr == needle {
                            return (ptr as usize - tape_start as usize) as u32;
                        }

                        ptr = ptr.sub(1);
                    }
                }

            // For non-one steps, fall back to our crappy impl
            } else {
                let mut ptr = tape_start.add(tape_offset as usize);
                loop {
                    if ptr >= tape_start {
                        if *ptr == needle {
                            return (ptr as usize - tape_start as usize) as u32;
                        }

                        ptr = ptr.sub(step as usize);
                    } else {
                        ptr = tape_end.sub(tape_start as usize - ptr as usize);
                    }
                }
            }
        }
    }))
    .unwrap_or_else(|error| {
        tracing::error!(
            ?state_ptr,
            ?tape_offset,
            ?step,
            ?needle,
            "scanl_wrapping panicked: {:?}",
            error,
        );
        u32::MAX
    })
}

/// Performs a left scan over the program tape for the given `needle` value
///
/// The scan starts at `tape_offset` and progresses to the beginning of the tape.
/// The `step` value decides which cells are checked, a step of 1 will cause every
/// cell to be checked while a step of 2 will cause every *other* cell to be checked.
///
/// The function returns [`u32::MAX`] as an error value. An error will be returned
/// if `step` is zero or another unforeseen error occurs. Additionally, if the current
/// tape doesn't contain the `needle` value or the given `step` value doesn't allow it
/// to ever reach it, the function will loop infinitely.
///
/// # Safety
///
/// `state_ptr` must be a valid pointer to a [`State`] instance and `tape_offset`
/// must be inbounds of the current program tape.
///
/// If the value is not contained within the scanned portion (`tape[..tape_offset]`)
/// this function causes UB.
///
pub(super) unsafe extern "fastcall" fn scanl_non_wrapping(
    state_ptr: *const State,
    tape_offset: u16,
    step: u16,
    needle: u8,
) -> u32 {
    panic::catch_unwind(AssertUnwindSafe(|| {
        if state_ptr.is_null() {
            tracing::error!(
                ?state_ptr,
                ?tape_offset,
                ?step,
                ?needle,
                "scanl_non_wrapping call got null state pointer"
            );
            return u32::MAX;
        } else if step == 0 {
            tracing::error!(
                ?state_ptr,
                ?tape_offset,
                ?step,
                ?needle,
                "scanl_non_wrapping called with a step of zero"
            );
            return u32::MAX;
        }

        unsafe {
            let tape_start = (*state_ptr).start_ptr;
            let mut ptr = tape_start.add(tape_offset as usize);

            while ptr >= tape_start {
                if *ptr == needle {
                    return (ptr as usize - tape_start as usize) as u32;
                }

                ptr = ptr.sub(step as usize);
            }

            // TODO: We could also just use `unreachable_unchecked()` here
            u32::MAX
        }
    }))
    .unwrap_or_else(|error| {
        tracing::error!(
            ?state_ptr,
            ?tape_offset,
            ?step,
            ?needle,
            "scanl_non_wrapping panicked: {:?}",
            error,
        );
        u32::MAX
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
