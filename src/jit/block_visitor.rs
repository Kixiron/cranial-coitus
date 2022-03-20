use crate::jit::basic_block::{
    Add, Assign, BasicBlock, BitNot, BlockId, Blocks, Branch, Cmp, Input, Instruction, Load, Mul,
    Neg, Not, Output, Phi, RValue, Store, Sub, Terminator, ValId, Value,
};

pub trait BasicBlockVisitor {
    fn visit_blocks(&mut self, blocks: &Blocks) {
        for block in blocks {
            self.before_visit_block(block);
            self.visit_block(block);
        }
    }

    /// Called before a block is visited
    fn before_visit_block(&mut self, _block: &BasicBlock) {}

    fn visit_block(&mut self, block: &BasicBlock) {
        for inst in block {
            self.before_visit_inst(inst);
            self.visit_inst(inst);
        }

        self.before_visit_term(block.terminator());
        self.visit_term(block.terminator());
    }

    /// Called before an instruction is visited
    fn before_visit_inst(&mut self, _inst: &Instruction) {}

    fn visit_inst(&mut self, inst: &Instruction) {
        match inst {
            Instruction::Store(store) => self.visit_store(store),
            Instruction::Assign(assign) => self.visit_assign(assign),
            Instruction::Output(output) => self.visit_output(output),
        }
    }

    fn visit_store(&mut self, store: &Store) {
        self.visit_value(store.ptr());
        self.visit_value(store.value());
    }

    fn visit_assign(&mut self, assign: &Assign) {
        self.visit_rval(assign.rval(), assign.value());
    }

    fn visit_rval(&mut self, rval: &RValue, assigned_to: ValId) {
        match rval {
            RValue::Cmp(eq) => self.visit_eq(eq, assigned_to),
            RValue::Phi(phi) => self.visit_phi(phi, assigned_to),
            RValue::Neg(neg) => self.visit_neg(neg, assigned_to),
            RValue::Not(not) => self.visit_not(not, assigned_to),
            RValue::Add(add) => self.visit_add(add, assigned_to),
            RValue::Sub(sub) => self.visit_sub(sub, assigned_to),
            RValue::Mul(mul) => self.visit_mul(mul, assigned_to),
            RValue::Load(load) => self.visit_load(load, assigned_to),
            RValue::Input(input) => self.visit_input(input, assigned_to),
            RValue::BitNot(bit_not) => self.visit_bit_not(bit_not, assigned_to),
        }
    }

    fn visit_eq(&mut self, eq: &Cmp, _assigned_to: ValId) {
        self.visit_value(eq.lhs());
        self.visit_value(eq.rhs());
    }

    fn visit_phi(&mut self, phi: &Phi, _assigned_to: ValId) {
        self.visit_value(phi.lhs());
        self.visit_value(phi.rhs());
    }

    fn visit_neg(&mut self, neg: &Neg, _assigned_to: ValId) {
        self.visit_value(neg.value());
    }

    fn visit_not(&mut self, not: &Not, _assigned_to: ValId) {
        self.visit_value(not.value());
    }

    fn visit_add(&mut self, add: &Add, _assigned_to: ValId) {
        self.visit_value(add.lhs());
        self.visit_value(add.rhs());
    }

    fn visit_sub(&mut self, sub: &Sub, _assigned_to: ValId) {
        self.visit_value(sub.lhs());
        self.visit_value(sub.rhs());
    }

    fn visit_mul(&mut self, mul: &Mul, _assigned_to: ValId) {
        self.visit_value(mul.lhs());
        self.visit_value(mul.rhs());
    }

    fn visit_load(&mut self, load: &Load, _assigned_to: ValId) {
        self.visit_value(load.ptr());
    }

    fn visit_input(&mut self, _input: &Input, _assigned_to: ValId) {}

    fn visit_bit_not(&mut self, bit_not: &BitNot, _assigned_to: ValId) {
        self.visit_value(bit_not.value());
    }

    fn visit_output(&mut self, output: &Output) {
        for &value in output.values() {
            self.visit_value(value);
        }
    }

    /// Called before a terminator is visited
    fn before_visit_term(&mut self, _term: &Terminator) {}

    fn visit_term(&mut self, term: &Terminator) {
        match term {
            Terminator::Unreachable => self.visit_unreachable(),
            Terminator::Jump(target) => self.visit_jump(target),
            Terminator::Return(value) => self.visit_return(value),
            Terminator::Branch(branch) => self.visit_branch(branch),
            Terminator::Error => self.visit_error(),
        }
    }

    fn visit_unreachable(&mut self) {}

    fn visit_jump(&mut self, _target: &BlockId) {}

    fn visit_return(&mut self, &value: &Value) {
        self.visit_value(value);
    }

    fn visit_branch(&mut self, branch: &Branch) {
        self.visit_value(branch.condition());
    }

    fn visit_error(&mut self) {
        panic!("visited an error terminator");
    }

    fn visit_value(&mut self, _value: Value) {}
}
