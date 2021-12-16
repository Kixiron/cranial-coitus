use crate::{jit::AsmResult, utils::AssertNone};
use iced_x86::code_asm::{registers::*, AsmRegister64, CodeAssembler};
use std::{cmp::Reverse, collections::VecDeque, fmt::Debug, num::NonZeroUsize};

pub(super) const NONVOLATILE_REGISTERS: &[AsmRegister64] =
    &[rbx, rbp, rdi, rsi, rsp, r12, r13, r14, r15];

const VOLATILE_REGISTERS: &[AsmRegister64] = &[rax, rcx, rdx, r8, r9, r10, r11];

const STACK_ALIGNMENT: usize = 16;

#[derive(Debug, Clone)]
pub struct Regalloc {
    registers: Registers,
    pub(super) stack: Stack,
}

impl Regalloc {
    /// Creates a new register allocator
    pub fn new() -> Self {
        Self {
            registers: Registers::new(),
            stack: Stack::new(),
        }
    }

    pub fn max_stack_size(&self) -> usize {
        self.stack.max_size
    }

    /// Allocate a specific register, returning it and the stack slot of the value it spilled
    /// (if applicable)
    pub fn allocate_specific(
        &mut self,
        asm: &mut CodeAssembler,
        register: AsmRegister64,
        clobber_sensitive: bool,
    ) -> AsmResult<(AsmRegister64, Option<StackSlot>)> {
        let (register, spilled) = self
            .registers
            .allocate_specific_r64(register, clobber_sensitive);

        let spilled = if let Some(spilled) = spilled {
            let (slot, can_push) = self.stack.reserve();

            // If we can push to the stack slot, use a push
            if can_push {
                asm.push(spilled)?;
                tracing::debug!(
                    "pushed value in register {:?} to the slot {:?}",
                    spilled,
                    slot,
                );

            // Otherwise move into the slot
            } else {
                asm.mov(rsp + self.stack.slot_offset(slot), spilled)?;
                tracing::debug!(
                    "moved value in register {:?} to the slot {:?}",
                    spilled,
                    slot,
                );
            }

            Some(slot)
        } else {
            None
        };

        Ok((register, spilled))
    }

    /// Allocate a register, returning it and the stack slot of the value it spilled (if applicable)
    pub fn allocate(
        &mut self,
        asm: &mut CodeAssembler,
        clobber_sensitive: bool,
    ) -> AsmResult<(AsmRegister64, Option<StackSlot>)> {
        let (register, spilled) = self.registers.allocate_r64(clobber_sensitive);

        let spilled = if let Some(spilled) = spilled {
            let (slot, can_push) = self.stack.reserve();

            // If we can push to the stack slot, use a push
            if can_push {
                asm.push(spilled)?;
                tracing::debug!(
                    "pushed value in register {:?} to the slot {:?}",
                    spilled,
                    slot,
                );

            // Otherwise move into the slot
            } else {
                asm.mov(rsp + self.stack.slot_offset(slot), spilled)?;
                tracing::debug!(
                    "moved value in register {:?} to the slot {:?}",
                    spilled,
                    slot,
                );
            }

            Some(slot)
        } else {
            None
        };

        Ok((register, spilled))
    }

    /// Deallocates a register
    pub fn deallocate(&mut self, register: AsmRegister64) {
        self.registers.deallocate_r64(register);
    }

    pub fn free_stack(&mut self) -> usize {
        let stack_size = self.stack.virtual_rsp;

        self.stack.dirty = false;
        self.stack.virtual_rsp = 0;
        self.stack.vacant_slots.clear();

        tracing::debug!(
            "freed entire stack with size of {}, vrsp: {} → 0",
            stack_size,
            stack_size,
        );

        stack_size
    }

    pub fn free(&mut self, asm: &mut CodeAssembler, slot: StackSlot) -> AsmResult<()> {
        let can_pop = self.stack.free(slot);

        if can_pop {
            self.stack.pop(slot);
            asm.add(rsp, slot.size as i32)?;

            tracing::debug!(
                "freed stack slot memory for slot {:?}, vrsp: {} → {}",
                slot,
                self.stack.virtual_rsp + slot.size,
                self.stack.virtual_rsp,
            );
        }

        Ok(())
    }

    pub fn push(
        &mut self,
        asm: &mut CodeAssembler,
        register: AsmRegister64,
    ) -> AsmResult<StackSlot> {
        self.registers.deallocate_r64(register);
        let (slot, can_push) = self.stack.reserve();

        // If we can push to the stack slot, use a push
        if can_push {
            asm.push(register)?;
            tracing::debug!("pushed register {:?} to slot {:?}", register, slot);

        // Otherwise move into the slot
        } else {
            asm.mov(rsp + self.stack.slot_offset(slot), register)?;
            tracing::debug!("moved register {:?} to slot {:?}", register, slot);
        }

        Ok(slot)
    }

    pub fn pop(
        &mut self,
        asm: &mut CodeAssembler,
        slot: StackSlot,
        register: AsmRegister64,
    ) -> AsmResult<AsmRegister64> {
        let can_pop = self.stack.free(slot);

        // We're popping a stack value into a register, so we need to allocate said register
        self.registers
            .allocate_specific_r64(register, true)
            .1
            .debug_unwrap_none();

        // If we can pop the slot, do it
        if can_pop {
            self.stack.pop(slot);
            asm.pop(register)?;

            tracing::debug!(
                "popped slot {:?} to register {:?}, vrsp: {} → {}",
                slot,
                register,
                self.virtual_rsp() + slot.size,
                self.virtual_rsp(),
            );

        // Otherwise move from the slot to the register
        } else {
            asm.mov(register, rsp + self.stack.slot_offset(slot))?;
            tracing::debug!("moved slot {:?} to register {:?}", slot, register);
        }

        Ok(register)
    }

    pub fn align_for_call(&self) -> usize {
        self.stack.align_for_call(&self.registers)
    }

    pub fn used_volatile_registers(&self) -> impl Iterator<Item = AsmRegister64> + '_ {
        self.registers.used_volatile_registers()
    }

    pub fn clobbered_registers(&self) -> impl Iterator<Item = AsmRegister64> + '_ {
        self.registers.clobbered_registers()
    }

    pub fn virtual_rsp(&self) -> usize {
        self.stack.virtual_rsp
    }

    pub fn slot_offset(&self, slot: StackSlot) -> usize {
        self.stack.slot_offset(slot)
    }
}

#[derive(Debug, Clone)]
pub(super) struct Stack {
    /// The "virtual" stack pointer
    pub(super) virtual_rsp: usize,
    /// Vacant stack slots
    pub(super) vacant_slots: Vec<StackSlot>,
    /// Is `true` if the vacant stack slots aren't sorted
    dirty: bool,
    max_size: usize,
}

impl Stack {
    /// Create a new stack manager
    fn new() -> Self {
        Self {
            virtual_rsp: 0,
            vacant_slots: Vec::new(),
            dirty: false,
            max_size: 0,
        }
    }

    fn slot_offset(&self, slot: StackSlot) -> usize {
        self.virtual_rsp - slot.offset
    }

    /// Reserve a stack slot of 8 bytes, returns the slot and whether or not the
    /// `push` instruction can be used to place a value into it
    // TODO: Allow customizable sizes
    pub(super) fn reserve(&mut self) -> (StackSlot, bool) {
        // If the vacant slots aren't sorted, sort them
        self.sort_vacant_slots();

        match self.vacant_slots.pop() {
            // If we can reuse an old stack slot, do it
            Some(slot) => {
                tracing::debug!(
                    is_top = slot.offset + slot.size == self.virtual_rsp,
                    "reused vacant stack slot: {:?}",
                    slot,
                );

                (slot, slot.offset + slot.size == self.virtual_rsp)
            }

            // Otherwise allocate an entirely new one
            None => {
                let slot = StackSlot {
                    offset: self.virtual_rsp,
                    size: 8,
                };
                self.virtual_rsp += 8;

                if self.virtual_rsp > self.max_size {
                    self.max_size = self.virtual_rsp;
                }

                tracing::debug!(
                    is_top = true,
                    "allocated new stack slot: {:?}, vrsp: {} → {}",
                    slot,
                    slot.offset,
                    self.virtual_rsp,
                );

                (slot, true)
            }
        }
    }

    /// Free a stack slot, returns whether or not `pop` can be used to deallocate the space
    pub(super) fn free(&mut self, slot: StackSlot) -> bool {
        tracing::debug!(
            is_top = slot.offset + slot.size == self.virtual_rsp,
            "freed slot from stack: {:?}",
            slot,
        );

        self.vacant_slots.push(slot);
        self.dirty = true;

        slot.offset + slot.size == self.virtual_rsp
    }

    /// Pops the slot from the top of the stack, it must be located at the top
    fn pop(&mut self, slot: StackSlot) {
        tracing::debug!(
            "popped slot from top of stack: {:?}, vrsp: {} → {}",
            slot,
            self.virtual_rsp,
            self.virtual_rsp - slot.size,
        );

        // Remove the slot from the set of vacant slots
        self.vacant_slots
            .drain_filter(|&mut stack| stack == slot)
            .for_each(|_| {});

        assert_eq!(slot.offset + slot.size, self.virtual_rsp);
        self.virtual_rsp -= slot.size;
    }

    /// Get the padding required to align the stack to 16 bytes for a function call
    // TODO: Check if there's 32 bytes of free space on the top of the stack
    fn align_for_call(&self, registers: &Registers) -> usize {
        // We start with an offset of 8 since `call` pushes the return address to the stack
        // https://www.felixcloutier.com/x86/call#operation
        let mut stack_offset = 8 + self.virtual_rsp;

        // Push all currently used volatile registers to the stack
        stack_offset += 8 * registers.used_volatile_registers().count();

        // Get the required padding to align the stack to 16 bytes
        // https://docs.microsoft.com/en-us/cpp/build/stack-usage?view=msvc-170#stack-allocation
        let stack_padding = (stack_offset + 32) % STACK_ALIGNMENT;

        // Reserve stack space for 4 arguments (must be allocated unconditionally for win64)
        // as well as the required padding to align the stack to 16 bytes
        // https://docs.microsoft.com/en-us/cpp/build/stack-usage?view=msvc-170#stack-allocation
        stack_padding + 32
    }

    /// Frees all contiguous vacant slots at the top of the stack
    pub(super) fn free_vacant_slots(&mut self) -> Option<NonZeroUsize> {
        self.sort_vacant_slots();

        let mut total_deallocated = 0;
        self.vacant_slots
            .drain_filter(|slot| {
                if slot.offset + slot.size == self.virtual_rsp {
                    self.virtual_rsp -= total_deallocated;
                    total_deallocated += slot.size;

                    true
                } else {
                    false
                }
            })
            .for_each(|_| {});

        NonZeroUsize::new(total_deallocated)
    }

    /// Sort all vacant stack slots in increasing order by their offset
    fn sort_vacant_slots(&mut self) {
        if self.dirty {
            self.dirty = false;

            // Reuse lower (higher offsets) slots first so that we have a better chance
            // of being able to deallocate higher stack slots
            let virtual_rsp = self.virtual_rsp;
            self.vacant_slots
                .sort_unstable_by_key(|slot| Reverse(virtual_rsp - slot.offset));

            if cfg!(debug_assertions) {
                // Ensure all slots are valid
                for slot in &self.vacant_slots {
                    assert!(slot.offset + slot.size <= self.virtual_rsp);
                }

                // Ensure no slots overlap
                for &[slot1, slot2] in self.vacant_slots.array_windows::<2>() {
                    let slot1_ptr = self.slot_offset(slot1);
                    let slot2_ptr = self.slot_offset(slot2);

                    assert!(
                        slot1_ptr <= slot2_ptr && slot1_ptr + slot1.size <= slot2_ptr + slot2.size,
                    );
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StackSlot {
    /// The offset from the virtual rsp
    offset: usize,
    /// The size of the stack slot
    size: usize,
}

impl StackSlot {
    pub fn new(offset: usize, size: usize) -> Self {
        Self { offset, size }
    }
}

#[derive(Debug, Clone)]
struct Registers {
    /// 64-bit registers
    r64: Vec<(AsmRegister64, Option<bool>)>,
    r64_lru: RegisterLru<AsmRegister64>,
    // TODO: Use xmm registers as two distinct 64-bit (or 4 32-bit integers)?
    // /// XMM registers
    // xmm: Vec<(AsmRegisterXmm, Option<()>)>,
    // xmm_lru: RegisterLru<AsmRegisterXmm>,
    //
    // TODO: 8-bit registers for storing booleans?
    // /// 8-bit registers
    // r8: Vec<(AsmRegister8, Option<()>)>,
    // r8_lru: RegisterLru<AsmRegister8>,
}

impl Registers {
    /// Create a new set of registers
    fn new() -> Self {
        // TODO: We could use rax and rcx here as well
        let r64 = vec![
            (r8, None),
            (r9, None),
            (r10, None),
            (r11, None),
            (r12, None),
            (r13, None),
            (r14, None),
            (r15, None),
            (rbp, None),
            (rbx, None),
        ];
        let r64_lru = RegisterLru::new(r64.len());

        Self { r64, r64_lru }
    }

    /// Allocates a specific 64-bit register and the register it evicted, if there is one
    #[track_caller]
    fn allocate_specific_r64(
        &mut self,
        register: AsmRegister64,
        clobber_sensitive: bool,
    ) -> (AsmRegister64, Option<AsmRegister64>) {
        tracing::debug!(
            "allocating clobber {} specific register: {:?}",
            if clobber_sensitive {
                "sensitive"
            } else {
                "insensitive"
            },
            register,
        );

        debug_assert!(self.r64.iter().any(|&(reg, _)| reg == register));

        for (reg, occupied) in &mut self.r64 {
            if *reg == register {
                let spilled = occupied.replace(clobber_sensitive).map(|_| register);
                self.r64_lru.add(register);

                return (register, spilled);
            }
        }

        panic!("attempted to allocate unmanaged register {:?}", register);
    }

    /// Allocates a 64-bit register and the register it evicted, if there is one
    fn allocate_r64(&mut self, clobber_sensitive: bool) -> (AsmRegister64, Option<AsmRegister64>) {
        tracing::debug!(
            "allocating clobber {} register",
            if clobber_sensitive {
                "sensitive"
            } else {
                "insensitive"
            },
        );

        // If there's an empty register we want to occupy it
        let empty_register = self.r64.iter_mut().find_map(|(register, occupied)| {
            if occupied.is_none() {
                *occupied = Some(clobber_sensitive);
                self.r64_lru.add(*register);

                Some(*register)
            } else {
                None
            }
        });

        match empty_register {
            Some(register) => {
                tracing::debug!(
                    "allocated clobber {} register in unoccupied register {:?}",
                    if clobber_sensitive {
                        "sensitive"
                    } else {
                        "insensitive"
                    },
                    register,
                );

                (register, None)
            }
            None => {
                let register = self
                    .r64_lru
                    .evict()
                    .expect("failed to evict register from lru cache");
                self.r64_lru.add(register);

                tracing::debug!(
                    "allocated clobber {} register in occupied register {:?}",
                    if clobber_sensitive {
                        "sensitive"
                    } else {
                        "insensitive"
                    },
                    register,
                );

                // Set the register with the new clobber_sensitive value
                let mut did_set_clobber = false;
                for (reg, occupied) in &mut self.r64 {
                    if *reg == register {
                        debug_assert!(occupied.is_some());
                        *occupied = Some(clobber_sensitive);

                        did_set_clobber = true;
                        break;
                    }
                }
                debug_assert!(did_set_clobber);

                // In the case that we spilled a register to the stack, the registers are the same;
                // The register currently holds the spilled value, so the caller must spill it to the
                // stack and then store the new value into the given register
                (register, Some(register))
            }
        }
    }

    /// Deallocates a 64bit register
    fn deallocate_r64(&mut self, register: AsmRegister64) {
        tracing::debug!("deallocated {:?}", register);

        self.r64_lru.remove(register);
        for (reg, occupied) in &mut self.r64 {
            if *reg == register {
                // debug_assert!(occupied.is_some());
                *occupied = None;

                return;
            }
        }

        if cfg!(debug_assertions) {
            panic!("tried to deallocate a register that wasn't allocated");
        }
    }

    /// Collects all registers that are currently in use, are volatile and can't be clobbered
    fn used_volatile_registers(&self) -> impl Iterator<Item = AsmRegister64> + '_ {
        self.r64
            .iter()
            // Filter out any unused registers or registers who don't care if they're clobbered
            .filter_map(|&(register, occupied)| {
                occupied
                    .filter(|&clobber_sensitive| clobber_sensitive)
                    .map(|_| register)
            })
            // We only want volatile registers, non-volatile registers must be preserved by the callee
            .filter(|register| VOLATILE_REGISTERS.contains(register))
    }

    /// Collects all registers that are currently in use, are volatile and can be clobbered.
    /// Should be used after a function call to determine which registers have been clobbered
    /// by said function call
    fn clobbered_registers(&self) -> impl Iterator<Item = AsmRegister64> + '_ {
        self.r64
            .iter()
            // Filter out any unused registers or registers who do care if they're clobbered
            .filter_map(|&(register, occupied)| {
                occupied
                    .filter(|&clobber_sensitive| !clobber_sensitive)
                    .map(|_| register)
            })
            // We only want volatile registers, non-volatile registers must be preserved by the callee
            .filter(|register| VOLATILE_REGISTERS.contains(register))
    }
}

#[derive(Debug, Clone)]
struct RegisterLru<T> {
    cache: VecDeque<T>,
    slots: usize,
}

impl<T> RegisterLru<T> {
    /// Create a new register LRU cache with `slot` slots
    fn new(slots: usize) -> Self {
        debug_assert_ne!(slots, 0);

        Self {
            cache: VecDeque::with_capacity(slots),
            slots,
        }
    }

    /// Adds or promotes a register to the front of the cache, evicting another if the cache is currently full
    fn add(&mut self, register: T) -> Option<T>
    where
        T: Copy + PartialEq + Debug,
    {
        // If the register is already in the cache, move it to the front
        if let Some(idx) = self.cache.iter().position(|&reg| reg == register) {
            if idx != 0 {
                self.cache.remove(idx).debug_unwrap();
                self.cache.push_front(register);
            }

            None

        // If the cache has space, push it to the front
        } else if self.cache.len() < self.slots {
            self.cache.push_front(register);
            None

        // Otherwise evict the least recently used register and push the given
        // register to the front of the cache
        } else {
            let evicted = self.cache.pop_back().unwrap();
            self.cache.push_front(register);

            Some(evicted)
        }
    }

    /// Removes a register from the cache
    fn remove(&mut self, register: T)
    where
        T: Copy + PartialEq + Debug,
    {
        // If the register exists within the cache, evict it
        if let Some(idx) = self.cache.iter().position(|&reg| reg == register) {
            self.cache.remove(idx).debug_unwrap();
        }
    }

    /// Evicts the least recently used register from the cache
    fn evict(&mut self) -> Option<T> {
        self.cache.pop_back()
    }
}
