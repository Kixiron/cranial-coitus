#[macro_use]
mod ffi;
mod basic_block;
mod block_builder;
mod block_visitor;
// mod cir_jit;
mod cir_to_bb;
mod codegen;
mod disassemble;
mod memory;

pub use codegen::JitReturnCode;
pub use ffi::State;
pub use memory::Executable;

use crate::{
    args::Settings,
    ir::{Block, Pretty, PrettyConfig},
    jit::codegen::Codegen,
};
use anyhow::{Context as _, Result};
use cranelift::{
    codegen::Context,
    frontend::FunctionBuilderContext,
    prelude::{
        isa::TargetIsa,
        settings::{self, Flags},
        Configurable,
    },
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataContext, Linkage, Module};
use std::{fs, mem::transmute, path::Path, slice};

/// The function produced by jit code
pub type JitFunction = unsafe extern "fastcall" fn(*mut State, *mut u8, *const u8) -> u8;

const SCAN_ERROR_MESSAGE: &str = "A scanr or scanl call failed";

pub struct Jit<'a> {
    /// The function builder context, which is reused across multiple
    /// `FunctionBuilder` instances
    builder_ctx: FunctionBuilderContext,

    /// The main Cranelift context, which holds the state for codegen. Cranelift
    /// separates this from `Module` to allow for parallel compilation, with a
    /// context per thread, though this isn't in the simple demo here
    ctx: Context,

    data_ctx: DataContext,

    /// The module, with the jit backend, which manages the JIT'd
    /// functions
    module: JITModule,

    /// The length of the (program/turing) tape we're targeting
    tape_len: u16,

    dump_dir: Option<&'a Path>,
    file_name: Option<&'a str>,
}

impl<'a> Jit<'a> {
    /// Create a new jit
    pub fn new(
        settings: &Settings,
        dump_dir: Option<&'a Path>,
        file_name: Option<&'a str>,
    ) -> Result<Self> {
        let isa = build_isa()?;
        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Add external functions to the module so they're accessible within
        // generated code
        builder.symbols([
            ("input", ffi::input as *const u8),
            ("output", ffi::output as *const u8),
            ("io_error", ffi::io_error as *const u8),
        ]);

        // If tape wrapping is UB, we can use our functions that take advantage of it
        if settings.tape_wrapping_ub {
            builder.symbols([
                ("scanr", ffi::scanr_non_wrapping as *const u8),
                ("scanl", ffi::scanl_non_wrapping as *const u8),
            ]);
        } else {
            builder.symbols([
                ("scanr", ffi::scanr_wrapping as *const u8),
                ("scanl", ffi::scanl_wrapping as *const u8),
            ]);
        }

        let mut module = JITModule::new(builder);
        let ctx = module.make_context();
        let builder_ctx = FunctionBuilderContext::new();

        let mut data_ctx = DataContext::new();
        data_ctx.define(Box::from(SCAN_ERROR_MESSAGE.as_bytes()));
        let message_id =
            module.declare_data("scan_error_message", Linkage::Export, false, false)?;
        module.define_data(message_id, &data_ctx)?;

        Ok(Self {
            builder_ctx,
            ctx,
            data_ctx,
            module,
            tape_len: settings.tape_len.get(),
            dump_dir,
            file_name,
        })
    }

    /// Compile a block of CIR into an executable buffer
    pub fn compile(mut self, block: &Block) -> Result<Executable> {
        // Translate CIR into SSA form
        let blocks = cir_to_bb::translate(block);

        // Create a formatted version of the ssa ir we just generated
        if let (Some(dump_dir), Some(file_name)) = (self.dump_dir, self.file_name) {
            let ssa_ir = blocks.pretty_print(PrettyConfig::minimal());
            fs::write(dump_dir.join(file_name).with_extension("ssa"), ssa_ir)
                .context("failed to write ssa ir file")?;
        }

        let mut codegen = Codegen::new(
            &blocks,
            &mut self.ctx,
            &mut self.module,
            &mut self.builder_ctx,
            &mut self.data_ctx,
            self.tape_len,
        )?;
        let function = codegen.run()?;

        if let (Some(dump_dir), Some(file_name)) = (self.dump_dir, self.file_name) {
            fs::write(
                dump_dir.join(file_name).with_extension("clif"),
                &function.to_string(),
            )
            .context("failed to write clif ir file")?;
        }

        codegen.finish();

        // Next, declare the function to jit. Functions must be declared
        // before they can be called, or defined
        let function_id = self.module.declare_function(
            "coitus_jit",
            Linkage::Export,
            &self.ctx.func.signature,
        )?;

        // Define the function within the jit
        let code_len = self.module.define_function(function_id, &mut self.ctx)?;

        // Now that compilation is finished, we can clear out the context state.
        self.module.clear_context(&mut self.ctx);

        // Finalize the functions which we just defined, which resolves any
        // outstanding relocations (patching in addresses, now that they're
        // available).
        self.module.finalize_definitions();

        // We can now retrieve a pointer to the machine code.
        let code = self.module.get_finalized_function(function_id);

        // Disassemble the generated instructions
        if let (Some(dump_dir), Some(file_name)) = (self.dump_dir, self.file_name) {
            let code_bytes = unsafe { slice::from_raw_parts(code, code_len.size as usize) };
            let disassembly = disassemble::disassemble(code_bytes)?;

            fs::write(dump_dir.join(file_name).with_extension("asm"), disassembly)
                .context("failed to write asm file")?;
        }

        // TODO: Assembly statistics

        let function = unsafe { transmute::<*const u8, JitFunction>(code) };
        let executable = Executable::new(function, self.module);

        Ok(executable)
    }
}

/// Produces the isa description for the current machine so that we can jit for it
fn build_isa() -> Result<Box<dyn TargetIsa>> {
    let mut flag_builder = settings::builder();

    // On at least AArch64, "colocated" calls use shorter-range relocations,
    // which might not reach all definitions; we can't handle that here, so
    // we require long-range relocation types.
    flag_builder
        .set("use_colocated_libcalls", "false")
        .context("failed to use colocated library calls")?;

    // Position independent code screws with our disassembly
    // flag_builder
    //     .set("is_pic", "true")
    //     .context("failed to set position independent code")?;

    cranelift_native::builder()
        .map_err(|msg| anyhow::anyhow!("host machine is not supported: {}", msg))?
        .finish(Flags::new(flag_builder))
        .context("failed to build target isa")
}
