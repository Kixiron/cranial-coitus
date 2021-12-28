#[macro_use]
mod ffi;
mod basic_block;
mod block_builder;
mod block_visitor;
mod cir_jit;
mod cir_to_bb;
mod coloring;
mod disassemble;
mod liveliness;
mod memory;
mod regalloc;

pub use memory::{CodeBuffer, Executable};

use crate::{
    ir::{Block, Pretty, PrettyConfig},
    jit::{
        basic_block::{
            Assign, BasicBlock, BlockId, Blocks, Instruction, Output, RValue, Store, Terminator,
            ValId, Value,
        },
        ffi::State,
        liveliness::Liveliness,
        regalloc::{Regalloc, StackSlot},
    },
    utils::AssertNone,
};
use iced_x86::code_asm::{CodeAssembler, *};
use std::{borrow::Cow, collections::BTreeMap};

type AsmResult<T> = Result<T, IcedError>;

const BITNESS: u32 = 64;

const RETURN_SUCCESS: i64 = 0;

const RETURN_IO_FAILURE: i64 = 101;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Operand {
    Byte(u8),
    Uint(u32),
    Bool(bool),
    Stack(StackSlot),
    Register(AsmRegister64),
}

impl Operand {
    pub fn is_zero(&self) -> bool {
        match *self {
            Operand::Byte(byte) => byte == 0,
            Operand::Uint(uint) => uint == 0,
            Operand::Bool(_) => todo!(),
            Operand::Stack(_) | Operand::Register(_) => false,
        }
    }
}

pub struct Jit {
    asm: CodeAssembler,
    io_failure: CodeLabel,
    has_io_functions: bool,
    values: BTreeMap<ValId, Operand>,
    regalloc: Regalloc,
    comments: BTreeMap<usize, Vec<String>>,
    named_labels: BTreeMap<usize, Cow<'static, str>>,
    block_labels: BTreeMap<BlockId, CodeLabel>,
    // TODO: Control flow graph + liveliness gives us the ability to
    //       get the current set of live variables at any given point
    liveliness: Liveliness,
    current_block: BlockId,
    current_inst: Option<usize>,
}

impl Jit {
    pub fn new(tape_len: usize) -> AsmResult<Self> {
        let mut asm = CodeAssembler::new(BITNESS)?;
        let io_failure = asm.create_label();

        Ok(Self {
            asm,
            io_failure,
            has_io_functions: false,
            values: BTreeMap::new(),
            regalloc: Regalloc::new(),
            comments: BTreeMap::new(),
            named_labels: BTreeMap::new(),
            block_labels: BTreeMap::new(),
            liveliness: Liveliness::new(),
            current_block: BlockId::new(u32::MAX),
            current_inst: None,
        })
    }

    #[tracing::instrument(skip_all)]
    pub fn assemble(&mut self, block: &Block) -> AsmResult<(Executable<CodeBuffer>, String)> {
        let blocks = cir_to_bb::translate(block);
        println!(
            "SSA form:\n{}",
            blocks.pretty_print(PrettyConfig::minimal()),
        );

        self.liveliness.run(&blocks);
        println!("liveliness: {:#?}", self.liveliness);

        self.named_label(".PROLOGUE");
        self.prologue()?;

        self.named_label(".BODY");

        self.create_block_labels(&blocks);
        for (idx, block) in blocks.iter().enumerate() {
            self.current_block = block.id();

            let next_block = blocks.get(idx + 1).map(BasicBlock::id);
            self.assemble_block(block, next_block)?;
        }

        // Only build the IO handler if there's IO functions
        if self.has_io_functions {
            self.build_io_failure()?;
        }

        let code_buffer = {
            // The maximum size of an instruction is 15 bytes, so we (virtually) allocate the most memory we could possibly use
            let maximum_possible_size = self.asm.instructions().len() * 15;
            let mut code_buffer = CodeBuffer::new(maximum_possible_size).unwrap();

            let code = self.asm.assemble(code_buffer.as_ptr() as u64)?;

            debug_assert!(code_buffer.len() <= code.capacity());
            code_buffer.copy_from_slice(&code).unwrap();

            code_buffer.executable().unwrap()
        };
        let pretty = self.disassemble();

        Ok((code_buffer, pretty))
    }

    fn assemble_block(&mut self, block: &BasicBlock, next_block: Option<BlockId>) -> AsmResult<()> {
        let label = *self.block_labels.get(&block.id()).unwrap();
        self.set_label(label);

        self.add_comment(format!("start {}", block.id()));
        // self.asm.nop()?;

        for (idx, inst) in block.iter().enumerate() {
            self.current_inst = Some(idx);
            self.assemble_inst(inst)?;
        }

        self.current_inst = None;
        self.assemble_terminator(block.terminator(), next_block)?;

        Ok(())
    }

    fn assemble_inst(&mut self, inst: &Instruction) -> AsmResult<()> {
        self.add_comment(inst.pretty_print(PrettyConfig::minimal()));

        match inst {
            Instruction::Store(store) => self.assemble_store(store),
            Instruction::Assign(assign) => self.assemble_assign(assign),
            Instruction::Output(output) => self.assemble_output(output),
        }
    }

    fn add(&mut self, lhs: AsmRegister64, rhs: i32) -> AsmResult<()> {
        if rhs == 0 {
            Ok(())
        } else if rhs == 1 {
            self.asm.inc(lhs)
        } else {
            self.asm.add(lhs, rhs)
        }
    }

    fn assemble_store(&mut self, store: &Store) -> AsmResult<()> {
        let ptr = match self.get_value(store.ptr()) {
            Operand::Byte(byte) => Operand::Uint(byte as u32),
            Operand::Bool(bool) => Operand::Uint(bool as u32),
            operand => operand,
        };
        let value = match self.get_value(store.value()) {
            Operand::Byte(byte) => Operand::Uint(byte as u32),
            Operand::Bool(bool) => Operand::Uint(bool as u32),
            operand => operand,
        };

        match (ptr, value) {
            (Operand::Uint(ptr), Operand::Uint(value)) => {
                let temp = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp, tape_ptr)?;

                self.asm.mov(byte_ptr(temp + ptr), value)?;

                self.deallocate_register(temp);
            }

            (Operand::Uint(ptr), Operand::Stack(value)) => {
                let temp1 = self.allocate_register()?;
                let temp2 = self.allocate_register()?;

                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp1, tape_ptr)?;

                let value = self.stack_offset(value);
                self.asm.mov(temp2, value)?;

                self.asm.mov(byte_ptr(temp1 + ptr), temp2)?;

                self.deallocate_register(temp2);
                self.deallocate_register(temp1);
            }

            (Operand::Uint(ptr), Operand::Register(value)) => {
                let temp = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp, tape_ptr)?;

                self.asm.mov(byte_ptr(temp + ptr), value)?;

                self.deallocate_register(temp);
            }

            (Operand::Stack(ptr), Operand::Uint(value)) => {
                let temp = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp, tape_ptr)?;

                let ptr = self.stack_offset(ptr);
                self.asm.add(temp, ptr)?;
                self.asm.mov(byte_ptr(temp), value)?;

                self.deallocate_register(temp);
            }
            (Operand::Stack(ptr), Operand::Stack(value)) => {
                let temp1 = self.allocate_register()?;
                let temp2 = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp1, tape_ptr)?;

                let (ptr, value) = (self.stack_offset(ptr), self.stack_offset(value));
                self.asm.add(temp1, ptr)?;
                self.asm.mov(temp2, value)?;
                self.asm.mov(byte_ptr(temp1), temp2)?;

                self.deallocate_register(temp1);
                self.deallocate_register(temp2);
            }
            (Operand::Stack(ptr), Operand::Register(value)) => {
                let temp = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp, tape_ptr)?;

                let ptr = self.stack_offset(ptr);
                self.asm.add(temp, ptr)?;
                self.asm.mov(byte_ptr(temp), value)?;

                self.deallocate_register(temp);
            }

            (Operand::Register(ptr), Operand::Uint(value)) => {
                let temp = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp, tape_ptr)?;

                self.asm.add(temp, ptr)?;
                self.asm.mov(byte_ptr(temp), value)?;

                self.deallocate_register(temp);
            }
            (Operand::Register(ptr), Operand::Stack(value)) => {
                let temp1 = self.allocate_register()?;
                let temp2 = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp1, tape_ptr)?;

                self.asm.add(temp1, ptr)?;
                let value = self.stack_offset(value);
                self.asm.mov(temp2, value)?;
                self.asm.mov(byte_ptr(temp1), temp2)?;

                self.deallocate_register(temp1);
                self.deallocate_register(temp2);
            }
            (Operand::Register(ptr), Operand::Register(value)) => {
                let temp = self.allocate_register()?;
                let tape_ptr = self.tape_start_ptr();
                self.asm.mov(temp, tape_ptr)?;

                self.asm.add(temp, ptr)?;
                self.asm.mov(byte_ptr(temp), value)?;

                self.deallocate_register(temp);
            }

            // We removed Byte and Bool from the possible operand types
            (Operand::Byte(_) | Operand::Bool(_), _) | (_, Operand::Byte(_) | Operand::Bool(_)) => {
                unreachable!()
            }
        }

        Ok(())
    }

    fn assemble_assign(&mut self, assign: &Assign) -> AsmResult<()> {
        match assign.rval() {
            RValue::Eq(eq) => {
                let dest = self.allocate_register()?;
                let (lhs, rhs) = (self.get_value(eq.lhs()), self.get_value(eq.rhs()));

                match (lhs, rhs) {
                    (Operand::Byte(lhs), Operand::Byte(rhs)) => {
                        self.asm.mov(dest, lhs as i64)?;
                        self.asm.cmp(dest, rhs as i32)?;
                    }
                    (Operand::Byte(lhs), Operand::Uint(rhs)) => {
                        self.asm.mov(dest, lhs as i64)?;
                        self.asm.cmp(dest, rhs as i32)?;
                    }
                    (Operand::Byte(lhs), Operand::Bool(rhs)) => {
                        self.asm.mov(dest, lhs as i64)?;
                        self.asm.cmp(dest, rhs as i32)?;
                    }
                    (Operand::Byte(lhs), Operand::Stack(rhs)) => {
                        let rhs = self.stack_offset(rhs);
                        self.asm.mov(dest, lhs as i64)?;
                        self.asm.cmp(dest, rhs)?;
                    }
                    (Operand::Byte(lhs), Operand::Register(rhs)) => {
                        self.asm.mov(dest, lhs as i64)?;
                        self.asm.cmp(dest, rhs)?;
                    }

                    (Operand::Uint(_), Operand::Byte(_)) => todo!(),
                    (Operand::Uint(_), Operand::Uint(_)) => todo!(),
                    (Operand::Uint(_), Operand::Bool(_)) => todo!(),
                    (Operand::Uint(_), Operand::Stack(_)) => todo!(),
                    (Operand::Uint(_), Operand::Register(_)) => todo!(),

                    (Operand::Bool(_), Operand::Byte(_)) => todo!(),
                    (Operand::Bool(_), Operand::Uint(_)) => todo!(),
                    (Operand::Bool(_), Operand::Bool(_)) => todo!(),
                    (Operand::Bool(_), Operand::Stack(_)) => todo!(),
                    (Operand::Bool(_), Operand::Register(_)) => todo!(),

                    (Operand::Stack(_), Operand::Byte(_)) => todo!(),
                    (Operand::Stack(_), Operand::Uint(_)) => todo!(),
                    (Operand::Stack(_), Operand::Bool(_)) => todo!(),
                    (Operand::Stack(_), Operand::Stack(_)) => todo!(),
                    (Operand::Stack(_), Operand::Register(_)) => todo!(),

                    (Operand::Register(lhs), Operand::Byte(rhs)) => {
                        self.asm.cmp(lhs, rhs as i32)?;
                    }
                    (Operand::Register(lhs), Operand::Uint(rhs)) => {
                        self.asm.cmp(lhs, rhs as i32)?;
                    }
                    (Operand::Register(lhs), Operand::Bool(rhs)) => {
                        self.asm.cmp(lhs, rhs as i32)?;
                    }
                    (Operand::Register(lhs), Operand::Stack(rhs)) => {
                        let rhs = self.stack_offset(rhs);
                        self.asm.cmp(lhs, rhs)?;
                    }
                    (Operand::Register(lhs), Operand::Register(rhs)) => self.asm.cmp(lhs, rhs)?,
                }

                // Set al to 1 if the operands are equal
                self.asm.setne(al)?;

                // Move the comparison result from al into the allocated
                // register with a zero sign extension
                self.asm.movzx(dest, al)?;

                self.values
                    .insert(assign.value(), Operand::Register(dest))
                    .debug_unwrap_none();
            }

            // RValue::Phi(_) => todo!(),
            // RValue::Neg(_) => todo!(),
            RValue::Not(not) => {
                let value = self.get_value(not.value());
                let dest = if let Value::Val(val) = not.value() {
                    match value {
                        Operand::Register(register)
                            if self.liveliness.is_last_usage(
                                val,
                                self.current_block,
                                self.current_inst,
                            ) =>
                        {
                            self.regalloc.allocate_overwrite(register);
                            self.values.remove(&val);
                            register
                        }

                        _ => {
                            let dest = self.allocate_register()?;
                            self.move_to_reg(dest, value)?;
                            dest
                        }
                    }
                } else {
                    let dest = self.allocate_register()?;
                    self.move_to_reg(dest, value)?;
                    dest
                };

                // Perform a *logical* not on the destination value
                self.asm.xor(dest, 1)?;

                self.values
                    .insert(assign.value(), Operand::Register(dest))
                    .debug_unwrap_none();
            }

            RValue::Add(add) => {
                let (lhs, rhs) = (self.get_value(add.lhs()), self.get_value(add.rhs()));

                let mut dest = None;
                if let Operand::Register(reg) = lhs {
                    if let Value::Val(val) = add.lhs() {
                        if self
                            .liveliness
                            .is_last_usage(val, self.current_block, self.current_inst)
                        {
                            self.regalloc.allocate_overwrite(reg);
                            self.values.remove(&val);
                            dest = Some(reg);
                        }
                    }
                }

                // TODO: We can reuse the right register as well, order of addition doesn't matter
                // if dest.is_none() {
                //     if let Operand::Register(reg) = rhs {
                //         if let Value::Val(val) = add.rhs() {
                //             if self.liveliness.is_last_usage(
                //                 val,
                //                 self.current_block,
                //                 self.current_inst,
                //             ) {
                //                 self.regalloc.allocate_overwrite(reg);
                //                 self.values.remove(&val);
                //                 dest = Some(reg);
                //             }
                //         }
                //     }
                // }

                // Currently `dest` is assumed to contain the lhs
                let dest = match dest {
                    Some(dest) => dest,
                    None => {
                        let dest = self.allocate_register()?;
                        self.move_to_reg(dest, lhs)?;
                        dest
                    }
                };

                match rhs {
                    Operand::Byte(0) | Operand::Uint(0) => {}
                    Operand::Byte(1) | Operand::Uint(1) => self.asm.inc(dest)?,

                    Operand::Byte(byte) => self.asm.add(dest, byte as i32)?,
                    Operand::Uint(uint) => self.asm.add(dest, uint as i32)?,

                    Operand::Bool(_) => panic!("cannot add a boolean"),

                    Operand::Stack(slot) => {
                        let addr = self.stack_offset(slot);
                        self.asm.add(dest, addr)?;
                    }

                    Operand::Register(reg) => self.asm.add(dest, reg)?,
                }

                self.values
                    .insert(assign.value(), Operand::Register(dest))
                    .debug_unwrap_none();
            }

            RValue::Sub(sub) => {
                let dest = self.allocate_register()?;
                let (lhs, rhs) = (self.get_value(sub.lhs()), self.get_value(sub.rhs()));

                self.move_to_reg(dest, lhs)?;
                match rhs {
                    Operand::Byte(byte) => self.asm.sub(dest, byte as i32)?,
                    Operand::Uint(uint) => self.asm.sub(dest, uint as i32)?,
                    Operand::Bool(bool) => self.asm.sub(dest, bool as i32)?,
                    Operand::Stack(slot) => {
                        let addr = self.stack_offset(slot);
                        self.asm.sub(dest, addr)?;
                    }
                    Operand::Register(reg) => self.asm.sub(dest, reg)?,
                }

                self.values
                    .insert(assign.value(), Operand::Register(dest))
                    .debug_unwrap_none();
            }

            // RValue::Mul(_) => todo!(),
            RValue::Load(load) => {
                let ptr = self.get_value(load.ptr());

                let dest = match ptr {
                    Operand::Byte(ptr) => {
                        let dest = self.allocate_register()?;
                        let tape_ptr = self.tape_start_ptr();

                        self.asm.mov(dest, tape_ptr)?;
                        self.asm.mov(dest, byte_ptr(dest + ptr as u32))?;

                        dest
                    }

                    Operand::Uint(ptr) => {
                        let dest = self.allocate_register()?;
                        let tape_ptr = self.tape_start_ptr();

                        self.asm.mov(dest, tape_ptr)?;
                        self.asm.mov(dest, byte_ptr(dest + ptr as u32))?;

                        dest
                    }

                    Operand::Bool(_) => panic!("cannot offset a pointer by a boolean"),

                    Operand::Stack(slot) => {
                        let dest = self.allocate_register()?;
                        let tape_ptr = self.tape_start_ptr();
                        let addr = self.stack_offset(slot);

                        self.asm.mov(dest, tape_ptr)?;
                        self.asm.add(dest, addr)?;
                        self.asm.mov(dest, byte_ptr(dest))?;

                        dest
                    }

                    Operand::Register(reg) => {
                        let mut dest = None;
                        if let Value::Val(val) = load.ptr() {
                            if self.liveliness.is_last_usage(
                                val,
                                self.current_block,
                                self.current_inst,
                            ) {
                                self.regalloc.allocate_overwrite(reg);
                                self.values.remove(&val);

                                dest = Some(reg);
                            }
                        }

                        let dest = match dest {
                            Some(dest) => dest,
                            None => self.allocate_register()?,
                        };
                        let tape_ptr = self.tape_start_ptr();

                        self.asm.mov(dest, tape_ptr)?;
                        self.asm.mov(dest, byte_ptr(dest + reg))?;

                        dest
                    }
                };

                self.values
                    .insert(assign.value(), Operand::Register(dest))
                    .debug_unwrap_none();
            }

            RValue::Input(_input) => {
                self.has_io_functions = true;

                // Move the state pointer into rcx
                self.asm.mov(rcx, self.state_ptr())?;

                // Setup the stack and save used registers
                let (stack_padding, slots) = self.before_win64_call()?;

                // Call the input function
                type_eq::<unsafe extern "win64" fn(state: *mut State) -> u16>(ffi::input);
                #[allow(clippy::fn_to_numeric_cast)]
                self.asm.mov(rax, ffi::input as u64)?;
                self.asm.call(rax)?;

                // Restore saved registers and deallocate stack space
                self.after_win64_call(stack_padding, slots)?;

                // `input()` returns a u16 within the rax register where the top six bytes are garbage, the
                // 7th byte is the input value (if the function succeeded) and the 8th byte is the status
                // code, 1 for error and 0 for success

                // Allocate the register that'll hold the value gotten from the input
                let input_reg = self.allocate_register()?;

                // Check if an IO error occurred
                self.asm.cmp(low_byte_register(rax), 0)?;

                // If so, jump to the io error label and exit
                self.asm.jnz(self.io_failure)?;

                // Save rax to the input register
                self.asm.mov(input_reg, rax)?;
                // Shift right by one byte in order to keep only the input value
                self.asm.shr(input_reg, 8)?;

                self.values
                    .insert(assign.value(), Operand::Register(input_reg))
                    .debug_unwrap_none();
            }

            // RValue::BitNot(_) => todo!(),
            rvalue => {
                todo!("{:?}", rvalue);
                self.values
                    .insert(assign.value(), Operand::Byte(0))
                    .debug_unwrap_none();
            }
        }

        Ok(())
    }

    fn assemble_output(&mut self, output: &Output) -> AsmResult<()> {
        self.has_io_functions = true;

        // Move the state pointer into rcx
        self.asm.mov(rcx, self.state_ptr())?;

        // Move the given byte into rdx
        let value = self.get_value(output.value());
        self.move_to_reg(rdx, value)?;

        // Setup the stack and save used registers
        let (stack_padding, slots) = self.before_win64_call()?;

        // Call the output function
        type_eq::<unsafe extern "win64" fn(state: *mut State, byte: u64) -> bool>(ffi::output);
        #[allow(clippy::fn_to_numeric_cast)]
        self.asm.mov(rax, ffi::output as u64)?;
        self.asm.call(rax)?;

        // Restore saved registers and deallocate stack space
        self.after_win64_call(stack_padding, slots)?;

        // Check if an IO error occurred
        self.asm.cmp(low_byte_register(rax), 0)?;
        // If so, jump to the io error label and exit
        self.asm.jnz(self.io_failure)?;

        Ok(())
    }

    fn assemble_terminator(
        &mut self,
        terminator: &Terminator,
        next_block: Option<BlockId>,
    ) -> AsmResult<()> {
        self.add_comment(terminator.pretty_print(PrettyConfig::minimal()));

        match terminator {
            Terminator::Error => panic!("encountered an error terminator"),
            Terminator::Unreachable => self.asm.ud2(),

            // If the next block is the target block, we can fall into it
            &Terminator::Jump(block) if Some(block) == next_block => Ok(()),
            Terminator::Jump(block) => {
                let jump_label = *self.block_labels.get(block).unwrap();
                self.asm.jmp(jump_label)
            }

            // TODO: Deallocate stack space and whatnot
            Terminator::Return(value) => {
                let value = self.get_value(*value);
                self.move_to_reg(rax, value)?;

                // Deallocate the stack space for the operand, we know that this is
                // the last usage for the value since the next instruction is a return
                if let Operand::Stack(slot) = value {
                    self.regalloc.free(&mut self.asm, slot)?;
                }

                self.asm.ret()
            }

            Terminator::Branch(branch) => {
                let cond = self.get_value(branch.condition());

                // Get both block's labels
                let (true_label, false_label) = (
                    *self.block_labels.get(&branch.true_jump()).unwrap(),
                    *self.block_labels.get(&branch.false_jump()).unwrap(),
                );

                // See <https://stackoverflow.com/a/54499552/9885253> for the logic behind the const-known
                // polyfills. Since our comparisons are currently pretty dumb, everything is an `x ≡ 0` so we
                // can just polyfill with things that non-destructively (in regards to register & stack state)
                // set the ZF to one or zero based on the result of the comparison, ZF=1 for equal and ZF=0 for
                // not equal
                match cond {
                    Operand::Byte(byte) => {
                        self.add_comment(format!("{} ≡ 0 = {}", byte, byte == 0));

                        if byte == 0 {
                            self.asm.jmp(true_label)?;
                        } else {
                            self.asm.jmp(false_label)?;
                        }

                        return Ok(());
                    }

                    Operand::Uint(uint) => {
                        self.add_comment(format!("{} ≡ 0 = {}", uint, uint == 0));

                        if uint == 0 {
                            self.asm.jmp(true_label)?;
                        } else {
                            self.asm.jmp(false_label)?;
                        }

                        return Ok(());
                    }

                    Operand::Bool(bool) => {
                        self.add_comment(format!("{} = {}", bool, bool));

                        if bool {
                            self.asm.jmp(true_label)?;
                        } else {
                            self.asm.jmp(false_label)?;
                        }

                        return Ok(());
                    }

                    Operand::Stack(slot) => {
                        let addr = self.stack_offset(slot);
                        self.asm.cmp(addr, 0)?;
                    }

                    Operand::Register(reg) => self.asm.cmp(reg, 0)?,
                }

                // If the next block is the true branch's block we can fall into it
                if Some(branch.true_jump()) == next_block {
                    self.asm.jnz(false_label)?;

                // If the next block is the false branch's block we can fall into it
                } else if Some(branch.false_jump()) == next_block {
                    self.asm.jz(true_label)?;

                // Otherwise we have to make a jump in both the true and false cases
                } else {
                    // If the condition is true, jump to the true label
                    self.asm.jz(true_label)?;

                    // Otherwise unconditionally jump to the false label
                    self.asm.jmp(false_label)?;
                }

                Ok(())
            }
        }
    }

    // See <https://stackoverflow.com/a/54499552/9885253>
    fn set_zf_one(&mut self) -> AsmResult<()> {
        // If we have a free register available we can xor zero it in order to set ZF=1
        if let Some(free) = self.regalloc.peek_free_register().map(low_byte_register) {
            self.asm.xor(free, free)?;

        // Otherwise we'll compare a register against itself, which is always
        // true and results in ZF=1
        } else {
            let reg = low_byte_register(self.regalloc.least_recently_used_register());
            self.asm.cmp(reg, reg)?;
        }

        Ok(())
    }

    // See <https://stackoverflow.com/a/54499552/9885253>
    fn set_zf_zero(&mut self) -> AsmResult<()> {
        // If we have a free register available we can use `or %reg, -1` to set ZF=0
        if let Some(free) = self.regalloc.peek_free_register().map(low_byte_register) {
            self.asm.or(free, -1)?;

        // Otherwise we'll test rsp against itself, rsp is known to be non-zero
        } else {
            self.asm.test(rsp, rsp)?;
        }

        Ok(())
    }

    /// Create labels for each basic block
    fn create_block_labels(&mut self, blocks: &Blocks) {
        for block in blocks {
            let label = self.create_label();
            self.block_labels
                .insert(block.id(), label)
                .debug_unwrap_none();
        }
    }

    fn move_to_reg(&mut self, dest: AsmRegister64, value: Operand) -> AsmResult<()> {
        match value {
            Operand::Byte(byte) => self.asm.mov(dest, byte as i64),
            Operand::Uint(uint) => self.asm.mov(dest, uint as i64),
            Operand::Bool(bool) => self.asm.mov(dest, bool as i64),
            Operand::Stack(slot) => {
                let ptr = self.stack_offset(slot);
                self.asm.mov(dest, ptr)
            }
            // If the source and destination registers are the same, do nothing
            Operand::Register(src) if src == dest => Ok(()),
            Operand::Register(src) => self.asm.mov(dest, src),
        }
    }

    #[track_caller]
    fn get_value(&mut self, value: Value) -> Operand {
        match value {
            Value::Byte(byte) => Operand::Byte(byte),
            Value::Uint(uint) => Operand::Uint(uint),
            Value::Bool(bool) => Operand::Bool(bool),
            Value::Val(var) => self.values.get(&var).copied().unwrap_or_else(|| {
                panic!(
                    "attempted to get the value of {:?}, but {} couldn't be found",
                    value, var,
                )
            }),
        }
    }

    #[track_caller]
    fn get_var(&self, var: ValId) -> Operand {
        match self.values.get(&var) {
            Some(&operand) => operand,
            None => panic!(
                "attempted to get the value of {}, but it couldn't be found",
                var,
            ),
        }
    }

    /// Create the IO failure block
    // FIXME: Needs to deallocate all of the stack space from any given failure point
    //        as well as restore callee sided registers
    #[allow(clippy::fn_to_numeric_cast)]
    fn build_io_failure(&mut self) -> AsmResult<()> {
        self.set_label(self.io_failure);

        self.add_comment("handle IO errors");

        // Move the state pointer into the rcx register
        self.asm.mov(rcx, self.state_ptr())?;

        // Setup the stack and save used registers
        let (stack_padding, slots) = self.before_win64_call()?;

        // Call the io error function
        type_eq::<unsafe extern "win64" fn(*mut State) -> bool>(ffi::io_error_encountered);
        self.asm.mov(rax, ffi::io_error_encountered as u64)?;
        self.asm.call(rax)?;

        // Restore saved registers and deallocate stack space
        self.after_win64_call(stack_padding, slots)?;

        // Set the return code
        self.asm.mov(rax, RETURN_IO_FAILURE)?;

        // Return from the function
        self.asm.ret()?;

        Ok(())
    }

    /// Set up the function's prologue
    // TODO: Save & restore callee sided registers
    fn prologue(&mut self) -> AsmResult<()> {
        // rcx contains the state pointer
        self.asm.mov(self.state_ptr(), rcx)?;

        // rdx contains the tape's start pointer
        self.asm.mov(self.tape_start_ptr(), rdx)?;

        // r8 contains the tape's end pointer
        self.asm.mov(self.tape_end_ptr(), r8)?;

        Ok(())
    }

    /// Get a pointer to the `*mut State` stored on the stack
    fn state_ptr(&self) -> AsmMemoryOperand {
        qword_ptr(rsp + 8 + self.regalloc.virtual_rsp())
    }

    /// Get a pointer to the `*mut u8` stored on the stack
    fn tape_start_ptr(&self) -> AsmMemoryOperand {
        qword_ptr(rsp + 16 + self.regalloc.virtual_rsp())
    }

    /// Get a pointer to the `*const u8` stored on the stack
    fn tape_end_ptr(&self) -> AsmMemoryOperand {
        qword_ptr(rsp + 24 + self.regalloc.virtual_rsp())
    }

    fn stack_offset(&self, slot: StackSlot) -> AsmMemoryOperand {
        qword_ptr(rsp + self.regalloc.slot_offset(slot))
    }

    // /// Push a register's value to the stack
    // fn push(&mut self, register: AsmRegister64) -> AsmResult<StackSlot> {
    //     let slot = StackSlot::new(self.regalloc.virtual_rsp(), 8);
    //
    //     self.asm.push(register)?;
    //     self.regalloc.stack.virtual_rsp += 8;
    //
    //     Ok(slot)
    // }

    fn allocate_register(&mut self) -> AsmResult<AsmRegister64> {
        let (register, spilled) = self.regalloc.allocate(&mut self.asm, true)?;

        // Spill the spilled register to the stack
        if let Some(spilled) = spilled {
            self.spill_register(register, spilled);
        }

        Ok(register)
    }

    fn allocate_specific(&mut self, register: AsmRegister64) -> AsmResult<AsmRegister64> {
        let (register, spilled) = self
            .regalloc
            .allocate_specific(&mut self.asm, register, true)?;

        // Spill the spilled register to the stack
        if let Some(spilled) = spilled {
            self.spill_register(register, spilled);
        }

        Ok(register)
    }

    fn allocate_specific_clobbered(&mut self, register: AsmRegister64) -> AsmResult<AsmRegister64> {
        let (register, spilled) =
            self.regalloc
                .allocate_specific(&mut self.asm, register, false)?;

        // Spill the spilled register to the stack
        if let Some(spilled) = spilled {
            self.spill_register(register, spilled);
        }

        Ok(register)
    }

    fn spill_register(&mut self, register: AsmRegister64, spilled: StackSlot) {
        let mut replaced_old = false;
        for value in self.values.values_mut() {
            if *value == Operand::Register(register) {
                *value = Operand::Stack(spilled);
                replaced_old = true;

                break;
            }
        }

        debug_assert!(replaced_old);
    }

    fn deallocate_register(&mut self, register: AsmRegister64) {
        self.regalloc.deallocate(register);
    }

    fn infinite_loop(&mut self) -> AsmResult<()> {
        // Add a nop to avoid having multiple labels for an instruction
        self.asm.nop()?;

        let label = self.asm.create_label();
        self.set_label(label);
        self.asm.jmp(label)?;

        Ok(())
    }

    /// Saves all currently used registers and allocates 32 bytes of stack for the called function
    /// as well as aligning the stack to 16 bytes. Returns the total extra stack space that was allocated
    /// for both the parameters and alignment padding
    fn before_win64_call(&mut self) -> AsmResult<(i32, Vec<(AsmRegister64, StackSlot)>)> {
        let mut slots = Vec::new();

        let registers: Vec<_> = self.regalloc.used_volatile_registers().collect();
        for register in registers {
            let slot = self.regalloc.push(&mut self.asm, register)?;
            slots.push((register, slot));

            tracing::debug!(
                "pushed register {:?} to slot {:?} before win64 call",
                register,
                slot,
            );
        }

        let stack_padding = self.regalloc.align_for_call() as i32;

        // Allocate the required space by subtracting from rsp
        self.asm.sub(rsp, stack_padding)?;

        Ok((stack_padding, slots))
    }

    /// Restores all saved registers and deallocates the space that was allocated for the called function
    fn after_win64_call(
        &mut self,
        stack_padding: i32,
        slots: Vec<(AsmRegister64, StackSlot)>,
    ) -> AsmResult<()> {
        // Deallocate the stack space we allocated for both the function arguments
        // and the padding required to align the stack to 16 bytes
        self.asm.add(rsp, stack_padding)?;

        // Pop all used registers from the stack
        for (register, slot) in slots.into_iter().rev() {
            self.regalloc.pop(&mut self.asm, slot, register)?;
            tracing::debug!(
                "popped register {:?} from slot {:?} after win64 call",
                register,
                slot,
            );
        }

        // Remove all clobbered registers
        let registers: Vec<_> = self.regalloc.clobbered_registers().collect();
        for clobbered in registers {
            self.regalloc.deallocate(clobbered);
            tracing::debug!("deallocated clobbered register {:?}", clobbered);
        }

        Ok(())
    }

    fn add_comment<C>(&mut self, comment: C)
    where
        C: ToString,
    {
        let idx = self.asm.instructions().len();
        self.comments
            .entry(idx)
            .or_insert_with(|| Vec::with_capacity(1))
            .push(comment.to_string());
    }

    fn create_label(&mut self) -> CodeLabel {
        self.asm.create_label()
    }

    #[track_caller]
    fn named_label<N>(&mut self, name: N)
    where
        N: Into<Cow<'static, str>>,
    {
        self.named_labels
            .insert(self.asm.instructions().len(), name.into())
            .debug_unwrap_none();
    }

    #[track_caller]
    fn set_label(&mut self, mut label: CodeLabel) {
        self.asm.set_label(&mut label).unwrap();
    }
}

fn type_eq<T>(_: T) {}

#[allow(non_upper_case_globals)]
fn low_byte_register(register: AsmRegister64) -> AsmRegister8 {
    match register {
        rax => al,
        rbx => bl,
        rcx => cl,
        rdx => dl,
        rsp => spl,
        rbp => bpl,
        rsi => sil,
        rdi => dil,
        r8 => r8b,
        r9 => r9b,
        r10 => r10b,
        r11 => r11b,
        r12 => r12b,
        r13 => r13b,
        r14 => r14b,
        r15 => r15b,
        _ => unreachable!(),
    }
}
