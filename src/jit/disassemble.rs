use crate::jit::{ffi, Jit};
use iced_x86::{
    FlowControl, Formatter, FormatterOptions, Instruction, MasmFormatter, SymbolResolver,
    SymbolResult,
};
use std::{borrow::Cow, collections::BTreeMap};

impl Jit {
    /// Disassembles the jit's current code
    pub fn disassemble(&self) -> String {
        let mut output = String::with_capacity(1024);
        self.disassemble_into(&mut output);
        output.shrink_to_fit();

        output
    }

    /// Writes the jit's disassembled code to the given string
    pub fn disassemble_into(&self, output: &mut String) {
        let mut formatter = MasmFormatter::with_options(Some(Box::new(Resolver)), None);
        set_formatter_options(formatter.options_mut());

        // Make sure all of the named labels are valid label names
        if cfg!(debug_assertions) {
            for label in self.named_labels.values() {
                assert!(is_valid_label_name(label));
            }
        }

        // Collect all jump targets and label them
        let mut labels = self.collect_jump_labels();

        // Collect the addresses of all named labels
        for (idx, inst) in self.asm.instructions().iter().enumerate() {
            let address = inst.ip();

            if let Some(label) = labels.get_mut(&address) {
                if let Some(named_label) = self.named_labels.get(&idx) {
                    label.clone_from(named_label);
                }
            }
        }

        // Format all instructions
        let mut is_indented = false;
        for (idx, inst) in self.asm.instructions().iter().enumerate() {
            let address = inst.ip();

            // Create pseudo labels for points of interest
            let mut has_named_label = false;
            if let Some(label) = self.named_labels.get(&idx) {
                is_indented = true;
                has_named_label = true;

                if idx != 0 {
                    output.push('\n');
                }

                debug_assert!(is_valid_label_name(label));
                output.push_str(label);
                output.push_str(":\n");
            }

            // If the current address is jumped to, add a label to the output text
            if let Some(label) = labels.get(&address).filter(|_| !has_named_label) {
                is_indented = true;

                if idx != 0 {
                    output.push('\n');
                }

                debug_assert!(is_valid_label_name(label));
                output.push_str(label);
                output.push_str(":\n");
            }

            // Display any comments
            if let Some(comments) = self.comments.get(&idx) {
                for comment in comments {
                    // Print each line of the comment
                    for line in comment.split('\n') {
                        if is_indented {
                            output.push_str("  ");
                        }

                        output.push_str("; ");
                        output.push_str(line);
                        output.push('\n');
                    }
                }
            }

            // Indent the line if needed
            if is_indented {
                output.push_str("  ");
            }

            // If this is a branch instruction we want to replace the branch address with
            // our human readable label
            if matches!(
                inst.flow_control(),
                FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch,
            ) {
                // Use the label name if we can find it
                if let Some(label) = labels.get(&inst.near_branch_target()) {
                    let mnemonic = format!("{:?}", inst.mnemonic()).to_lowercase();
                    output.push_str(&mnemonic);
                    output.push(' ');

                    debug_assert!(is_valid_label_name(label));
                    output.push_str(label);

                // Otherwise fall back to normal formatting
                } else {
                    tracing::warn!(
                        "failed to get branch label for {} (address: {:#x})",
                        inst,
                        inst.near_branch_target(),
                    );

                    formatter.format(inst, output);
                }

            // Otherwise format the instruction into the output buffer
            } else {
                formatter.format(inst, output);
            }

            // Add a newline between each instruction (and a trailing one)
            output.push('\n');
        }
    }

    fn collect_jump_labels(&self) -> BTreeMap<u64, Cow<'static, str>> {
        // Collect all jump targets
        let mut jump_targets = Vec::new();
        for inst in self.asm.instructions() {
            if matches!(
                inst.flow_control(),
                FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch,
            ) {
                jump_targets.push(inst.near_branch_target());
            }
        }

        // Sort and deduplicate the jump targets
        jump_targets.sort_unstable();
        jump_targets.dedup();

        // Name each jump target in increasing order
        // FIXME: Allow jumps to the epilogue to use the
        //        epilogue label & generalized named labels
        jump_targets
            .into_iter()
            .enumerate()
            .map(|(idx, address)| (address, Cow::Owned(format!(".LBL_{}", idx))))
            .collect()
    }
}

struct Resolver;

impl SymbolResolver for Resolver {
    #[allow(clippy::fn_to_numeric_cast)]
    fn symbol(
        &mut self,
        _instruction: &Instruction,
        _operand: u32,
        _instruction_operand: Option<u32>,
        address: u64,
        _address_size: u32,
    ) -> Option<SymbolResult<'_>> {
        if address == ffi::io_error_encountered as u64 {
            Some(SymbolResult::with_str(address, "io_error_encountered"))
        } else if address == ffi::input as u64 {
            Some(SymbolResult::with_str(address, "input"))
        } else if address == ffi::output as u64 {
            Some(SymbolResult::with_str(address, "output"))
        } else {
            None
        }
    }
}

/// Set up the formatter's options
fn set_formatter_options(options: &mut FormatterOptions) {
    // Set up hex formatting
    options.set_uppercase_hex(true);
    options.set_hex_prefix("0x");
    options.set_hex_suffix("");

    // Make operand formatting pretty
    options.set_space_after_operand_separator(true);
    options.set_space_between_memory_add_operators(true);
    options.set_space_between_memory_mul_operators(true);
    options.set_scale_before_index(false);
}

/// Returns `true` if the label matches `\.[a-zA-Z0-9_]+
fn is_valid_label_name(label: &str) -> bool {
    label.starts_with('.')
        && label.len() >= 2
        && label
            .chars()
            .skip(1)
            .all(|char| char.is_alphanumeric() || char == '_')
}
