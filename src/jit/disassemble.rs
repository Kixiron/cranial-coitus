use crate::jit::ffi;
use iced_x86::{
    Decoder, DecoderOptions, FlowControl, Formatter, FormatterOptions, Instruction, MasmFormatter,
    SymbolResolver, SymbolResult,
};
use std::collections::BTreeMap;

pub fn disassemble(code: &[u8]) -> String {
    let mut formatter = MasmFormatter::with_options(Some(Box::new(Resolver)), None);
    set_formatter_options(formatter.options_mut());

    // Decode all instructions
    let instructions: Vec<_> = Decoder::new(64, code, DecoderOptions::NONE)
        .into_iter()
        .collect();

    // Collect all jump targets and label them
    let labels = collect_jump_labels(&instructions);

    // Format all instructions
    let mut output = String::with_capacity(1024);
    let (mut is_indented, mut in_jump_block) = (false, false);

    for (idx, inst) in instructions.iter().enumerate() {
        let address = inst.ip();
        let is_control_flow = matches!(
            inst.flow_control(),
            FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch,
        );

        if !is_control_flow && in_jump_block {
            in_jump_block = false;
            output.push('\n');
        }

        // If the current address is jumped to, add a label to the output text
        if let Some(label) = labels.get(&address) {
            is_indented = true;

            if idx != 0 {
                output.push('\n');
            }

            debug_assert!(is_valid_label_name(label));
            output.push_str(label);
            output.push_str(":\n");
        }

        // Indent the line if needed
        if is_indented {
            output.push_str("  ");
        }

        // If this is a branch instruction we want to replace the branch address with
        // our human readable label
        if is_control_flow {
            in_jump_block = true;

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

                formatter.format(inst, &mut output);
            }

        // Otherwise format the instruction into the output buffer
        } else {
            formatter.format(inst, &mut output);
        }

        // Add a newline between each instruction (and a trailing one)
        output.push('\n');
    }

    output
}

fn collect_jump_labels(instructions: &[Instruction]) -> BTreeMap<u64, String> {
    // Collect all jump targets
    let mut jump_targets = Vec::new();
    for inst in instructions {
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
    jump_targets
        .into_iter()
        .enumerate()
        .map(|(idx, address)| (address, format!(".LBL_{}", idx)))
        .collect()
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
