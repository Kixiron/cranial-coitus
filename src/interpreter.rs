use crate::{
    ir::{
        Add, Assign, AssignTag, Block, Call, Cmp, CmpKind, Const, Expr, Gamma, Instruction, Load,
        Mul, Neg, Not, Store, Sub, Theta, Value, VarId, Variance,
    },
    utils::{self, AssertNone},
    values::{Cell, Ptr},
};
use std::{
    collections::BTreeMap,
    fmt::{self, Display},
    ops::{Neg as _, Not as _},
};

type Result<T> = std::result::Result<T, EvaluationError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationError {
    StepLimitReached,
    UnknownCellRead,
}

pub struct Machine<I, O> {
    pub tape: Vec<Option<Cell>>,
    pub values: Vec<BTreeMap<VarId, Const>>,
    pub values_idx: usize,
    input: I,
    output: O,
    step_limit: usize,
    steps: usize,
    // TODO: Track and display per-instruction stats
    pub stats: ExecutionStats,
}

impl<I, O> Machine<I, O>
where
    I: FnMut() -> u8,
    O: FnMut(u8),
{
    pub fn new(step_limit: usize, tape_len: u16, input: I, output: O) -> Self {
        let mut values = Vec::with_capacity(256);
        values.push(BTreeMap::new());

        Self {
            tape: vec![Some(Cell::zero()); tape_len as usize],
            values,
            values_idx: 0,
            input,
            output,
            step_limit,
            steps: 0,
            stats: ExecutionStats::new(),
        }
    }

    #[tracing::instrument(target = "cranial_coitus::interpreter", skip_all)]
    pub fn execute(&mut self, block: &mut Block, should_profile: bool) -> Result<&[Option<Cell>]> {
        for inst in block.iter_mut() {
            self.handle(inst, should_profile)?;
        }

        Ok(&self.tape)
    }

    fn handle(&mut self, inst: &mut Instruction, should_profile: bool) -> Result<()> {
        let is_subgraph_param = matches!(
            inst,
            Instruction::Assign(Assign {
                tag: AssignTag::InputParam(_) | AssignTag::OutputParam,
                ..
            })
        );

        if !is_subgraph_param && !inst.is_lifetime_end() {
            self.stats.instructions += 1;
        }

        match inst {
            Instruction::Call(call) => self.call(call, should_profile)?.debug_unwrap_none(),
            Instruction::Assign(assign) => self.assign(assign, should_profile)?,
            Instruction::Theta(theta) => self.theta(theta, should_profile)?,
            Instruction::Gamma(gamma) => self.gamma(gamma, should_profile)?,
            Instruction::Store(store) => self.store(store, should_profile)?,
            Instruction::LifetimeEnd(_) => {}
        }

        if self.steps >= self.step_limit {
            match inst {
                Instruction::Call(call) => {
                    tracing::info!("hit step limit within call {}", call.effect);
                }
                Instruction::Assign(assign) => {
                    tracing::info!("hit step limit within assign {}", assign.var);
                }
                Instruction::Theta(theta) => {
                    tracing::info!(
                        "hit step limit within theta {}",
                        theta.output_effect.unwrap(),
                    );
                }
                Instruction::Gamma(gamma) => {
                    tracing::info!("hit step limit within gamma {}", gamma.effect);
                }
                Instruction::Store(store) => {
                    tracing::info!("hit step limit within store {}", store.effect);
                }

                // We don't want to error out on a lifetime end
                Instruction::LifetimeEnd(_) => return Ok(()),
            }

            Err(EvaluationError::StepLimitReached)
        } else {
            let is_var_assign = matches!(
                inst,
                Instruction::Assign(Assign {
                    value: Expr::Value(_),
                    ..
                })
            );

            // Don't penalize constant assignments or subgraph params
            if !is_subgraph_param && !is_var_assign && !inst.is_lifetime_end() {
                self.steps += 1;
            }

            Ok(())
        }
    }

    fn eval(&mut self, expr: &mut Expr, should_profile: bool) -> Result<Const> {
        match expr {
            Expr::Cmp(cmp) => self.cmp(cmp),
            Expr::Add(add) => self.add(add),
            Expr::Sub(sub) => self.sub(sub),
            Expr::Mul(mul) => self.mul(mul),
            Expr::Not(not) => self.not(not),
            Expr::Neg(neg) => self.neg(neg),
            Expr::Value(value) => self.get_const(value),
            Expr::Load(load) => self.load(load),
            Expr::Call(call) => Ok(self
                .call(call, should_profile)?
                .expect("expected a call that produces output")),
        }
    }

    fn call(&mut self, call: &mut Call, should_profile: bool) -> Result<Option<Const>> {
        if should_profile {
            call.invocations += 1;
        }

        match &*call.function {
            "input" => {
                self.stats.input_calls += 1;

                if !call.args.is_empty() {
                    panic!("expected zero args for input call")
                }

                Ok(Some(Const::Cell(Cell::new((self.input)()))))
            }

            "output" => {
                self.stats.output_calls += 1;

                for value in &call.args {
                    let byte = self.get_byte(value)?.into_inner();
                    (self.output)(byte);
                }

                Ok(None)
            }

            other => panic!("unrecognized function {:?}", other),
        }
    }

    fn assign(&mut self, assign: &mut Assign, should_profile: bool) -> Result<()> {
        if should_profile {
            assign.invocations += 1;
        }

        if assign.tag.is_input_param() {
            self.values_idx -= 1;
        }

        tracing::trace!(
            ?assign,
            values_idx = self.values_idx,
            "getting assign value for input param",
        );
        let value = self.eval(&mut assign.value, should_profile)?;

        if assign.tag.is_input_param() {
            self.values_idx += 1;
        }

        self.values[self.values_idx]
            .insert(assign.var, value)
            .debug_unwrap_none();

        if assign.tag.is_output_param() {
            self.values[self.values_idx - 1].insert(assign.var, value);
        }

        tracing::trace!(
            values_idx = self.values_idx,
            ?assign,
            "assigned {:?} to {}",
            value,
            assign.var,
        );

        Ok(())
    }

    fn theta(&mut self, theta: &mut Theta, should_profile: bool) -> Result<()> {
        self.values.push(BTreeMap::new());
        self.values_idx += 1;

        let mut iter = 0;
        loop {
            self.stats.loop_iterations += 1;
            if should_profile {
                theta.loops += 1;
            }

            let current_steps = self.steps;
            for inst in &mut theta.body {
                // In the first loop iteration we want to execute every single assign instruction,
                // including variant and invariant ones. The invariant ones will pull the invariant
                // values into the loop and the variant ones will get the initial values of variant
                // variables. After the first iteration we don't execute variant or invariant assigns,
                // invariant assigns already have their proper values and variant ones are set within
                // the theta's code
                if iter != 0
                    && matches!(
                        inst,
                        Instruction::Assign(Assign {
                            tag: AssignTag::InputParam(Variance::Variant { .. }),
                            ..
                        })
                    )
                {
                    continue;
                }

                self.handle(inst, should_profile)?;

                let is_subgraph_param = matches!(
                    inst,
                    Instruction::Assign(Assign {
                        tag: AssignTag::InputParam(_) | AssignTag::OutputParam,
                        ..
                    })
                );
                if should_profile && !is_subgraph_param {
                    theta.body_inst_count += 1;
                }
            }

            // If the body doesn't change our step limit at all, we want to still take steps
            // to make sure we don't infinitely loop
            if self.steps == current_steps {
                self.steps += 1;
            }

            let cond = theta.cond.as_ref().expect("expected a theta condition");
            let should_break = self.get_bool(cond)?.not();
            tracing::trace!(
                "theta condition: {}, iteration: {}, value: {:?}",
                should_break,
                iter,
                cond,
            );

            if should_break {
                self.values.pop().debug_unwrap();
                self.values_idx -= 1;

                tracing::trace!(
                    values_idx = self.values_idx,
                    values = ?theta
                        .outputs
                        .iter()
                        .map(|(output, _value)| (output, &self.values[self.values_idx][output]))
                        .collect::<BTreeMap<_, _>>(),
                    "theta output values",
                );

                break;
            } else {
                let mut values_buffer = BTreeMap::new();
                for (output, value) in &theta.outputs {
                    let input_var = theta.output_feedback[output];
                    let feedback_value = self.get_const(value)?;

                    tracing::trace!(
                        values_idx = self.values_idx,
                        "feeding back {:?} from {:?} to {:?}",
                        feedback_value,
                        output,
                        input_var,
                    );

                    values_buffer
                        .insert(input_var, feedback_value)
                        .debug_unwrap_none();
                }
                self.values[self.values_idx] = values_buffer;

                tracing::trace!(
                    values_idx = self.values_idx,
                    values = ?self.values[self.values_idx],
                    "theta feedback values",
                );

                iter += 1;
            }
        }

        Ok(())
    }

    fn gamma(&mut self, gamma: &mut Gamma, should_profile: bool) -> Result<()> {
        self.stats.branches += 1;

        let cond = self.get_bool(&gamma.cond)?;

        self.values.push(BTreeMap::new());
        self.values_idx += 1;

        let branch = if cond {
            if should_profile {
                gamma.true_branches += 1;
            }

            &mut gamma.true_branch
        } else {
            if should_profile {
                gamma.false_branches += 1;
            }

            &mut gamma.false_branch
        };

        for inst in branch {
            self.handle(inst, should_profile)?;
        }

        self.values.pop().debug_unwrap();
        self.values_idx -= 1;

        Ok(())
    }

    fn store(&mut self, store: &mut Store, should_profile: bool) -> Result<()> {
        self.stats.stores += 1;
        if should_profile {
            store.stores += 1;
        }

        let ptr = self.get_ptr(&store.ptr)?;
        let value = self.get_byte(&store.value)?;
        tracing::trace!(
            "stored {} to {} (previous value: {:?})",
            ptr,
            value,
            self.tape[ptr],
        );

        self.tape[ptr] = Some(value);

        Ok(())
    }

    fn load(&mut self, load: &Load) -> Result<Const> {
        self.stats.loads += 1;

        let ptr = self.get_ptr(&load.ptr)?;
        tracing::trace!("loaded {:?} from {}", self.tape[ptr], ptr);

        self.tape[ptr]
            .ok_or(EvaluationError::UnknownCellRead)
            .map(Const::Cell)
    }

    fn cmp(&self, cmp: &Cmp) -> Result<Const> {
        let (lhs, rhs) = (self.get_byte(&cmp.lhs)?, self.get_byte(&cmp.rhs)?);
        tracing::trace!(
            lhs = ?self.get_const(&cmp.lhs),
            rhs = ?self.get_const(&cmp.rhs),
            "({:?} {} {:?}) is {}",
            lhs,
            cmp.op.operator(),
            rhs,
            lhs == rhs,
        );

        let result = match cmp.op {
            CmpKind::Eq => lhs == rhs,
            CmpKind::Neq => lhs != rhs,
            CmpKind::Less => lhs < rhs,
            CmpKind::Greater => lhs > rhs,
            CmpKind::LessEq => lhs <= rhs,
            CmpKind::GreaterEq => lhs >= rhs,
        };

        Ok(Const::Bool(result))
    }

    fn add(&self, add: &Add) -> Result<Const> {
        let (lhs, rhs) = (self.get_const(&add.lhs)?, self.get_const(&add.rhs)?);
        Ok(lhs + rhs)
    }

    fn sub(&self, sub: &Sub) -> Result<Const> {
        let (lhs, rhs) = (self.get_const(&sub.lhs)?, self.get_const(&sub.rhs)?);
        Ok(lhs - rhs)
    }

    fn mul(&self, mul: &Mul) -> Result<Const> {
        let (lhs, rhs) = (self.get_const(&mul.lhs)?, self.get_const(&mul.rhs)?);
        Ok(lhs * rhs)
    }

    fn not(&self, not: &Not) -> Result<Const> {
        Ok(self.get_const(&not.value)?.not())
    }

    fn neg(&self, neg: &Neg) -> Result<Const> {
        Ok(self.get_const(&neg.value)?.neg())
    }

    fn get_const(&self, value: &Value) -> Result<Const> {
        match value {
            Value::Var(var) => Ok(self.values[self.values_idx]
                .get(var)
                .copied()
                .unwrap_or_else(|| panic!("expected a value for the given variable {:?}", var))),
            &Value::Const(constant) => Ok(constant),
            Value::Missing => Err(EvaluationError::UnknownCellRead),
        }
    }

    fn get_byte(&self, value: &Value) -> Result<Cell> {
        Ok(self.get_const(value)?.into_cell())
    }

    fn get_ptr(&self, value: &Value) -> Result<Ptr> {
        Ok(self.get_const(value)?.into_ptr(self.tape.len() as u16))
    }

    fn get_bool(&self, value: &Value) -> Result<bool> {
        Ok(self
            .get_const(value)?
            .as_bool()
            .expect("expected a boolean constant"))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ExecutionStats {
    pub loads: usize,
    pub stores: usize,
    pub loop_iterations: usize,
    pub branches: usize,
    pub input_calls: usize,
    pub output_calls: usize,
    pub instructions: usize,
    pub in_branch_instructions: usize,
    pub in_loop_instructions: usize,
}

impl ExecutionStats {
    pub fn new() -> Self {
        Self {
            loads: 0,
            stores: 0,
            loop_iterations: 0,
            branches: 0,
            input_calls: 0,
            output_calls: 0,
            instructions: 0,
            in_branch_instructions: 0,
            in_loop_instructions: 0,
        }
    }

    pub fn display(&self) -> DisplayStats<'_> {
        DisplayStats::new(self)
    }

    fn fields(&self) -> [(&'static str, usize); 7] {
        [
            ("instructions", self.instructions),
            ("loads", self.loads),
            ("stores", self.stores),
            ("loop iters", self.loop_iterations),
            ("branches", self.branches),
            ("input calls", self.input_calls),
            ("output calls", self.output_calls),
        ]
    }
}

impl Default for ExecutionStats {
    fn default() -> Self {
        Self::new()
    }
}

pub struct DisplayStats<'a> {
    stats: &'a ExecutionStats,
}

impl<'a> DisplayStats<'a> {
    pub const fn new(stats: &'a ExecutionStats) -> Self {
        Self { stats }
    }
}

impl Display for DisplayStats<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fields = self.stats.fields();
        let longest_field = fields
            .iter()
            .map(|(field, _)| field.len())
            .max()
            .unwrap_or(0);

        for (name, value) in fields {
            write!(
                f,
                "  {:<padding$} : {}",
                name,
                value,
                padding = longest_field,
            )?;

            if !matches!(name, "instructions" | "loop iters" | "branches") {
                writeln!(
                    f,
                    ", {:.02}%",
                    utils::percent_total(self.stats.instructions, value),
                )?;
            } else {
                writeln!(f)?;
            }
        }

        Ok(())
    }
}

test_opts! {
    addition_loop,
    input = [50, 24],
    output = [74, 0],
    |graph, effect, tape_len| {
        let ptr = graph.int(Ptr::zero(tape_len)).value();
        compile_brainfuck_into(",>,[-<+>]<.>.", graph, ptr, effect, tape_len).1
    },
}

test_opts! {
    branching,
    input = [50, 24, 10, 0],
    output = [50, 24, 10, 0],
    |graph, effect, tape_len| {
        let ptr = graph.int(Ptr::zero(tape_len)).value();
        compile_brainfuck_into("+[>,.]", graph, ptr, effect, tape_len).1
    },
}
