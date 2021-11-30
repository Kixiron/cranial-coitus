use crate::jit::ffi::State;
use anyhow::Result;
use std::{
    io::Error,
    mem::{align_of, transmute},
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
};
use winapi::um::{
    handleapi::CloseHandle,
    memoryapi::{VirtualAlloc, VirtualFree, VirtualProtect},
    processthreadsapi::{FlushInstructionCache, GetCurrentProcess},
    winnt::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE, PAGE_READWRITE},
};

#[repr(transparent)]
pub(super) struct CodeBuffer {
    buffer: NonNull<[u8]>,
}

impl CodeBuffer {
    pub fn new(length: usize) -> Result<Self> {
        if length == 0 {
            tracing::error!("tried to allocate code buffer of zero bytes");
            anyhow::bail!("tried to allocate a code buffer of 0 bytes");
        }

        let length = length + (length % align_of::<u64>());
        tracing::debug!("allocating a code buffer of {} bytes", length);

        // Safety: VirtualAlloc allocates the requested memory initialized with zeroes
        let ptr = unsafe {
            VirtualAlloc(
                ptr::null_mut(),
                length,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };

        let ptr = if let Some(ptr) = NonNull::new(ptr.cast()) {
            ptr
        } else {
            let error = Error::last_os_error();
            tracing::error!(
                "call to VirtualAlloc() for {} bytes failed: {}",
                length,
                error,
            );

            let error =
                anyhow::anyhow!(error).context("failed to allocate code buffer's backing memory");
            return Err(error);
        };

        let buffer = NonNull::slice_from_raw_parts(ptr, length);
        tracing::debug!(
            "allocated code buffer of {} bytes at {:p}",
            length,
            buffer.as_ptr(),
        );

        Ok(Self { buffer })
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    #[allow(dead_code)]
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

    pub fn executable(mut self) -> Result<Executable<Self>> {
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

        Ok(Executable::new(self))
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
pub(super) struct Executable<T>(T);

impl<T> Executable<T> {
    fn new(inner: T) -> Self {
        Self(inner)
    }
}

impl Executable<CodeBuffer> {
    pub(super) unsafe fn call(&self, state: *mut State, start: *mut u8, end: *mut u8) -> u8 {
        let func: unsafe extern "win64" fn(*mut State, *mut u8, *const u8) -> u8 =
            transmute(self.0.as_ptr());

        func(state, start, end)
    }
}
