mod instruction;
mod terminator;

use crate::{
    jit::{
        basic_block::{BlockId, Blocks, ValId},
        ffi, State, SCAN_ERROR_MESSAGE,
    },
    utils::{HashMap, HashSet},
};
use anyhow::{Context as _, Result};
use cranelift::{
    codegen::{
        ir::{FuncRef, Function},
        Context,
    },
    frontend::{FunctionBuilder, FunctionBuilderContext},
    prelude::{
        isa::CallConv,
        types::{B1, I16, I32, I8},
        AbiParam, Block, InstBuilder, Signature, Type, Value,
    },
};
use cranelift_jit::JITModule;
use cranelift_module::{DataContext, Linkage, Module};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum JitReturnCode {
    Success = 0,
    IoFailure = 101,
    ScanFailure = 102,
}

pub struct Codegen<'a> {
    ssa: &'a Blocks,
    builder: FunctionBuilder<'a>,
    module: &'a mut JITModule,
    _data_ctx: &'a mut DataContext,
    intrinsics: Intrinsics,
    handlers: Handlers,
    params: JitParams,
    values: HashMap<ValId, (Value, Type)>,
    blocks: HashMap<BlockId, Block>,
    visited_blocks: HashSet<BlockId>,
    ptr_type: Type,
    param_buffer: Vec<Value>,
    tape_len: u16,
}

impl<'a> Codegen<'a> {
    pub fn new(
        ssa: &'a Blocks,
        ctx: &'a mut Context,
        module: &'a mut JITModule,
        builder: &'a mut FunctionBuilderContext,
        data_ctx: &'a mut DataContext,
        tape_len: u16,
    ) -> Result<Self> {
        let mut builder = FunctionBuilder::new(&mut ctx.func, builder);
        let ptr_type = module.target_config().pointer_type();

        // Create all intrinsic functions
        let intrinsics = Intrinsics::new(&mut builder, module, ptr_type)?;

        // Create all basic blocks
        let blocks: HashMap<_, _> = ssa
            .iter()
            .map(|block| (block.id(), builder.create_block()))
            .collect();

        // Add parameters to the generated function
        create_function_signature(&mut builder, ptr_type);

        // Create handlers for our generated code
        let handlers = Handlers::new(&mut builder);

        // Get the parameters added to the entry block
        let params = {
            // Get the entry block
            let entry = blocks[&ssa.entry().id()];
            builder.switch_to_block(entry);

            // Add the function parameters to the entry block
            builder.append_block_params_for_function_params(entry);

            // Retrieve the values of the function parameters
            let params = builder.block_params(entry);
            JitParams::new(params[0], params[1], params[2])
        };

        Ok(Self {
            ssa,
            builder,
            module,
            _data_ctx: data_ctx,
            intrinsics,
            handlers,
            params,
            values: HashMap::default(),
            blocks,
            visited_blocks: HashSet::default(),
            ptr_type,
            param_buffer: Vec::new(),
            tape_len,
        })
    }

    pub fn run(&mut self) -> Result<&Function> {
        // TODO: Seal blocks as soon as there's no more branches to it
        for block in self.ssa {
            // Switch to the basic block for the current block
            let clif_block = self.blocks[&block.id()];
            self.builder.switch_to_block(clif_block);

            // Generate code for each instruction
            for inst in block.instructions() {
                self.instruction(inst);
            }

            // Generate code for the block's terminator
            self.terminator(block, block.terminator());

            // After we've done code generation for the current block,
            // mark it as visited
            self.visited_blocks.insert(block.id());

            // If all incoming branches have been visited, we can seal this block
            let all_incoming_branches_finished = self
                .ssa
                .incoming_jumps(block.id())
                .all(|source| self.visited_blocks.contains(&source));
            if all_incoming_branches_finished {
                self.builder.seal_block(clif_block);
            }
        }

        // If the io handler block was ever used, construct it
        if self.handlers.io_error_handler_used() {
            assert_type::<unsafe extern "fastcall" fn(*mut State) -> bool>(ffi::io_error);

            let io_error_handler = self.io_error_handler();
            self.builder.switch_to_block(io_error_handler);

            // Call the io error reporting function
            let state_ptr = self.state_ptr();
            self.builder
                .ins()
                .call(self.intrinsics.io_error, &[state_ptr]);

            // Return with an error code
            let return_code = self
                .builder
                .ins()
                .iconst(I8, JitReturnCode::IoFailure as i64);
            self.builder.ins().return_(&[return_code]);

            // Mark this block as cold since it's only for error handling
            self.builder.set_cold_block(io_error_handler);
            self.builder.seal_block(io_error_handler);
        }

        // If the scan handler block was ever used, construct it
        if self.handlers.scan_error_handler_used() {
            assert_type::<unsafe extern "fastcall" fn(*mut State, *const u8, usize) -> bool>(
                ffi::output,
            );

            let scan_error_handler = self.scan_error_handler();
            self.builder.switch_to_block(scan_error_handler);

            // Get the pointer to the scan error message
            let error_message =
                self.module
                    .declare_data("scan_error_message", Linkage::Export, false, false)?;
            let error_message = self
                .module
                .declare_data_in_func(error_message, self.builder.func);
            let error_message = self
                .builder
                .ins()
                .symbol_value(self.ptr_type, error_message);

            // Get the length of the message
            let message_len = self
                .builder
                .ins()
                .iconst(self.ptr_type, SCAN_ERROR_MESSAGE.len() as i64);

            // Call the output function to report the error
            let state_ptr = self.state_ptr();
            self.builder.ins().call(
                self.intrinsics.output,
                &[state_ptr, error_message, message_len],
            );

            // Return with an error code
            let return_code = self
                .builder
                .ins()
                .iconst(I8, JitReturnCode::ScanFailure as i64);
            self.builder.ins().return_(&[return_code]);

            // Mark this block as cold since it's only for error handling
            self.builder.set_cold_block(scan_error_handler);
            self.builder.seal_block(scan_error_handler);
        }

        // We're now finished so we can seal every block
        self.builder.seal_all_blocks();

        Ok(&*self.builder.func)
    }

    #[track_caller]
    pub fn finish(mut self) {
        // Finalize the generated code
        self.builder.finalize();
    }

    fn state_ptr(&self) -> Value {
        self.params.state_ptr
    }

    fn tape_start(&self) -> Value {
        self.params.tape_start
    }

    fn input_function(&self) -> FuncRef {
        self.intrinsics.input
    }

    fn output_function(&self) -> FuncRef {
        self.intrinsics.output
    }

    fn scanr_function(&self) -> FuncRef {
        self.intrinsics.scanr
    }

    fn scanl_function(&self) -> FuncRef {
        self.intrinsics.scanl
    }

    fn io_error_handler(&mut self) -> Block {
        self.handlers.io_error_handler()
    }

    fn scan_error_handler(&mut self) -> Block {
        self.handlers.scan_error_handler()
    }
}

/// Intrinsic functions for the brainfuck runtime
struct Intrinsics {
    input: FuncRef,
    output: FuncRef,
    io_error: FuncRef,
    scanr: FuncRef,
    scanl: FuncRef,
}

impl Intrinsics {
    fn new(
        builder: &mut FunctionBuilder<'_>,
        module: &mut JITModule,
        ptr_type: Type,
    ) -> Result<Self> {
        assert_type::<unsafe extern "fastcall" fn(*mut State, *mut u8) -> bool>(ffi::input);
        let input = create_imported_function(builder, module, "input", &[ptr_type, ptr_type], B1)?;

        assert_type::<unsafe extern "fastcall" fn(*mut State, *const u8, usize) -> bool>(
            ffi::output,
        );
        let output = create_imported_function(
            builder,
            module,
            "output",
            &[ptr_type, ptr_type, ptr_type],
            B1,
        )?;

        assert_type::<unsafe extern "fastcall" fn(*mut State) -> bool>(ffi::io_error);
        let io_error = create_imported_function(builder, module, "io_error", &[ptr_type], B1)?;

        assert_type::<unsafe extern "fastcall" fn(*const State, u16, u16, u8) -> u32>(
            ffi::scanr_wrapping,
        );
        let scanr =
            create_imported_function(builder, module, "scanr", &[ptr_type, I16, I16, I8], I32)?;

        assert_type::<unsafe extern "fastcall" fn(*const State, u16, u16, u8) -> u32>(
            ffi::scanl_wrapping,
        );
        let scanl =
            create_imported_function(builder, module, "scanl", &[ptr_type, I16, I16, I8], I32)?;

        Ok(Self {
            input,
            output,
            io_error,
            scanr,
            scanl,
        })
    }
}

struct Handlers {
    io_error_handler: (Block, bool),
    scan_error_handler: (Block, bool),
}

impl Handlers {
    fn new(builder: &mut FunctionBuilder<'_>) -> Self {
        Self {
            io_error_handler: (builder.create_block(), false),
            scan_error_handler: (builder.create_block(), false),
        }
    }

    fn io_error_handler(&mut self) -> Block {
        self.io_error_handler.1 = true;
        self.io_error_handler.0
    }

    fn io_error_handler_used(&self) -> bool {
        self.io_error_handler.1
    }

    fn scan_error_handler(&mut self) -> Block {
        self.scan_error_handler.1 = true;
        self.scan_error_handler.0
    }

    fn scan_error_handler_used(&self) -> bool {
        self.scan_error_handler.1
    }
}

struct JitParams {
    state_ptr: Value,
    tape_start: Value,
    _tape_end: Value,
}

impl JitParams {
    fn new(state_ptr: Value, tape_start: Value, tape_end: Value) -> Self {
        Self {
            state_ptr,
            tape_start,
            _tape_end: tape_end,
        }
    }
}

/// Creates the signature of the current function,
/// `extern "fastcall" fn(*mut State, *mut u8, *const u8) -> u8`
fn create_function_signature(builder: &mut FunctionBuilder, ptr_type: Type) {
    let func = &mut *builder.func;
    let ptr_param = AbiParam::new(ptr_type);

    // We use fastcall here since it's one of the only calling conventions
    // that both Rust and Cranelift support (Rust functions must use the
    // `fastcall` convention to match this)
    //
    // TODO: I'd really rather use the C calling convention or something,
    //       but as of now I don't think cranelift supports it
    func.signature.call_conv = CallConv::WindowsFastcall;

    // State pointer
    func.signature.params.push(ptr_param);

    // Data start pointer
    func.signature.params.push(ptr_param);

    // Data end pointer
    func.signature.params.push(ptr_param);

    // Return a byte from the function
    func.signature.returns.push(AbiParam::new(I8));
}

fn create_imported_function(
    builder: &mut FunctionBuilder<'_>,
    module: &mut JITModule,
    name: &str,
    params: &[Type],
    ret: Type,
) -> Result<FuncRef> {
    let sig = create_signature(module, params, ret);
    let callee = module
        .declare_function(name, Linkage::Import, &sig)
        .with_context(|| format!("failed to create '{}' intrinsic", name))?;

    Ok(module.declare_func_in_func(callee, builder.func))
}

fn create_signature(module: &mut JITModule, params: &[Type], ret: Type) -> Signature {
    let mut sig = module.make_signature();
    sig.params.extend(params.iter().copied().map(AbiParam::new));
    sig.params.shrink_to_fit();
    sig.returns.push(AbiParam::new(ret));
    sig.call_conv = CallConv::WindowsFastcall;

    sig
}

fn assert_type<T>(_: T) {}
