use crate::jit::{ffi::State, JitFunction};
use anyhow::{Context, Result};
use std::{
    cmp::max,
    io::{self, Error, Write},
    mem::transmute,
    panic::{self, AssertUnwindSafe},
    ptr::{self, NonNull},
    slice,
    time::Instant,
};
use winapi::um::{
    handleapi::CloseHandle,
    memoryapi::{VirtualAlloc, VirtualFree, VirtualProtect},
    processthreadsapi::{FlushInstructionCache, GetCurrentProcess},
    winnt::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE, PAGE_READWRITE},
};

/// The OS's memory page size, currently set to a blanket of 4kb
// https://devblogs.microsoft.com/oldnewthing/20210510-00/?p=105200
const PAGE_SIZE: usize = 1024 * 4;

// TODO: Full-on commit-on-write semantics for this
pub struct CodeBuffer {
    /// A pointer to the allocated memory, `buffer..buffer + capacity` is *virtually allocated*
    /// memory and `buffer..buffer + length` is *physically allocated* memory. Only the latter
    /// may be read from or written to, the former must first be committed in order to be
    /// readable/writable
    buffer: NonNull<u8>,
    /// The length of the buffer's valid and executable area
    length: usize,
    /// The total capacity of the buffer, aligned to memory pages
    capacity: usize,
}

impl CodeBuffer {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            tracing::error!("tried to allocate code buffer of zero bytes");
            anyhow::bail!("tried to allocate a code buffer of 0 bytes");
        }

        // FIXME: This isn't aligning things properly?
        let padding = capacity % PAGE_SIZE;
        let padded_capacity = capacity + padding;
        tracing::debug!(
            "allocating a code buffer with a capacity of {} bytes \
            (added {} bytes of padding to {} in order to align it to {} bytes)",
            padded_capacity,
            padding,
            capacity,
            PAGE_SIZE,
        );

        // Safety: VirtualAlloc allocates the requested memory initialized with zeroes
        let ptr = unsafe {
            VirtualAlloc(
                ptr::null_mut(),
                padded_capacity,
                MEM_RESERVE,
                PAGE_READWRITE,
            )
        };

        let buffer = NonNull::new(ptr.cast()).ok_or_else(|| {
            let error = Error::last_os_error();
            tracing::error!(
                "call to VirtualAlloc() for {} bytes failed: {}",
                padded_capacity,
                error,
            );

            anyhow::anyhow!(error).context("failed to allocate code buffer's backing memory")
        })?;

        tracing::debug!(
            "allocated code buffer of {} bytes at {:p}",
            padded_capacity,
            buffer.as_ptr(),
        );

        Ok(Self {
            buffer,
            length: 0,
            capacity: padded_capacity,
        })
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn copy_from_slice(&mut self, source: &[u8]) -> Result<()> {
        self.length = max(self.length, source.len());

        // Commit the memory for the length of the source array
        let ret = unsafe {
            VirtualAlloc(
                self.buffer.as_ptr().cast(),
                self.len(),
                MEM_COMMIT,
                PAGE_READWRITE,
            )
        };

        if ret.is_null() {
            let error = Error::last_os_error();
            tracing::error!(
                "committing {} bytes in `.copy_from_slice()` via `VirtualAlloc()` failed: {}",
                self.len(),
                error,
            );

            let error =
                anyhow::anyhow!(error).context("failed to allocate code buffer's backing memory");
            return Err(error);
        }
        assert_eq!(self.buffer.as_ptr().cast(), ret);

        // Copy the data over
        self.as_mut_slice()[..source.len()].copy_from_slice(source);

        Ok(())
    }

    #[allow(dead_code)]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.buffer.as_ptr(), self.len()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.buffer.as_ptr(), self.len()) }
    }

    pub const fn as_ptr(&self) -> *const u8 {
        self.buffer.as_ptr() as *const u8
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.buffer.as_ptr()
    }

    pub fn executable(mut self) -> Result<Executable> {
        tracing::debug!("making code buffer at {:p} executable", self.as_ptr());

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
                let error = Error::last_os_error();
                tracing::error!(
                    "call to VirtualProtect() for buffer {:p} ({} bytes) and PAGE_EXECUTE failed: {}",
                    self.as_ptr(),
                    self.len(),
                    error,
                );

                let error = anyhow::anyhow!(error)
                    .context("failed to mark code buffer's backing memory as executable");
                return Err(error);
            }

            // Get a pseudo handle to the current process
            let handle = GetCurrentProcess();

            // Flush the instruction cache
            let flush_result = FlushInstructionCache(handle, self.as_mut_ptr().cast(), self.len());

            // Closing the handle of the current process is a noop, but we do it anyways for correctness
            let close_handle_result = CloseHandle(handle);

            // Handle errors in FlushInstructionCache()
            if flush_result == 0 {
                let error = Error::last_os_error();
                tracing::error!(
                    "call to FlushInstructionCache() for buffer {:p} ({} bytes) failed: {}",
                    self.as_ptr(),
                    self.len(),
                    error,
                );

                let error = anyhow::anyhow!(error).context("failed to flush instruction cache");
                return Err(error);

            // Handle errors in CloseHandle()
            } else if close_handle_result == 0 {
                let error = Error::last_os_error();
                tracing::error!(
                    "call to CloseHandle() for buffer {:p} ({} bytes) failed: {}",
                    self.as_ptr(),
                    self.len(),
                    error,
                );

                let error = anyhow::anyhow!(error).context("failed to close process handle");
                return Err(error);
            }
        }

        todo!()
        // Ok(Executable::new(self))
    }
}

// impl Deref for CodeBuffer {
//     type Target = [u8];
//
//     fn deref(&self) -> &Self::Target {
//         self.as_slice()
//     }
// }

// impl DerefMut for CodeBuffer {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         self.as_mut_slice()
//     }
// }

impl Drop for CodeBuffer {
    fn drop(&mut self) {
        let free_result = unsafe { VirtualFree(self.as_mut_ptr().cast(), 0, MEM_RELEASE) };

        if free_result == 0 {
            tracing::error!(
                "failed to deallocate {} bytes at {:p} from jit",
                self.capacity(),
                self.as_ptr(),
            );
        }
    }
}

#[repr(transparent)]
pub struct Executable(JitFunction);

impl Executable {
    pub(super) fn new(function: JitFunction) -> Self {
        Self(function)
    }

    pub unsafe fn call(&self, state: *mut State, start: *mut u8, end: *mut u8) -> u8 {
        unsafe { (self.0)(state, start, end) }
    }

    pub unsafe fn execute(&self, tape: &mut [u8]) -> Result<u8> {
        let tape_len = tape.len();

        let (stdin, stdout) = (io::stdin(), io::stdout());
        let mut state = State::new(stdin.lock(), stdout.lock());
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

        writeln!(&mut state.stdout).context("failed to write newline to stdout")?;
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
