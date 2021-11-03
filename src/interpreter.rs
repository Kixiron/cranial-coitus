use crate::{
    ir::{
        Add, Assign, AssignTag, Block, Call, Const, Eq, Expr, Gamma, Instruction, Load, Neg, Not,
        Store, Theta, Value, VarId, Variance,
    },
    utils::{AssertNone, HashMap},
};

type Result<T> = std::result::Result<T, StepLimitReached>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepLimitReached;

pub struct Machine<'a> {
    tape: Vec<u8>,
    values: HashMap<VarId, Const>,
    input: Box<dyn FnMut() -> u8 + 'a>,
    output: Box<dyn FnMut(u8) + 'a>,
    step_limit: usize,
    steps: usize,
    // TODO: Track and display per-instruction stats
    pub stats: ExecutionStats,
}

impl<'a> Machine<'a> {
    pub fn new<I, O>(step_limit: usize, length: usize, input: I, output: O) -> Self
    where
        I: FnMut() -> u8 + 'a,
        O: FnMut(u8) + 'a,
    {
        Self {
            tape: vec![0; length],
            values: HashMap::with_capacity_and_hasher(2048, Default::default()),
            input: Box::new(input),
            output: Box::new(output),
            step_limit,
            steps: 0,
            stats: ExecutionStats::new(),
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn execute(&mut self, block: &mut Block) -> Result<&[u8]> {
        for inst in block.iter_mut() {
            self.handle(inst)?;
        }

        Ok(&self.tape)
    }

    fn handle(&mut self, inst: &mut Instruction) -> Result<()> {
        let is_subgraph_param = matches!(
            inst,
            Instruction::Assign(Assign {
                tag: AssignTag::InputParam(_) | AssignTag::OutputParam,
                ..
            })
        );

        if !is_subgraph_param {
            self.stats.instructions += 1;
        }

        match inst {
            Instruction::Call(call) => self.call(call).debug_unwrap_none(),
            Instruction::Assign(assign) => self.assign(assign),
            Instruction::Theta(theta) => self.theta(theta)?,
            Instruction::Gamma(gamma) => self.gamma(gamma)?,
            Instruction::Store(store) => self.store(store),
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
            }

            Err(StepLimitReached)
        } else {
            let is_var_assign = matches!(
                inst,
                Instruction::Assign(Assign {
                    value: Expr::Value(_),
                    ..
                })
            );

            // Don't penalize constant assignments or subgraph params
            if !is_subgraph_param && !is_var_assign {
                self.steps += 1;
            }

            Ok(())
        }
    }

    fn eval(&mut self, expr: &mut Expr) -> Const {
        match expr {
            Expr::Eq(eq) => self.eq(eq),
            Expr::Add(add) => self.add(add),
            Expr::Not(not) => self.not(not),
            Expr::Neg(neg) => self.neg(neg),
            Expr::Value(value) => self.get_const(value).clone(),
            Expr::Load(load) => self.load(load),
            Expr::Call(call) => self
                .call(call)
                .expect("expected a call that produces output"),
        }
    }

    fn call(&mut self, call: &mut Call) -> Option<Const> {
        call.invocations += 1;

        match &*call.function {
            "input" => {
                self.stats.input_calls += 1;

                if !call.args.is_empty() {
                    panic!("expected zero args for input call")
                }

                Some(Const::Byte((self.input)()))
            }

            "output" => {
                self.stats.output_calls += 1;

                let [value]: &[Value; 1] = (&*call.args)
                    .try_into()
                    .expect("expected one arg for output call");
                let byte = self.get_byte(value);

                (self.output)(byte);

                None
            }

            other => panic!("unrecognized function {:?}", other),
        }
    }

    fn assign(&mut self, assign: &mut Assign) {
        assign.invocations += 1;

        let value = self.eval(&mut assign.value);
        tracing::trace!("assigned {:?} to {}", value, assign.var);

        // Note: Double inserts are allowed here because of loops
        self.values.insert(assign.var, value);
    }

    fn theta(&mut self, theta: &mut Theta) -> Result<()> {
        let mut iter = 0;
        loop {
            self.stats.loop_iterations += 1;
            theta.loops += 1;

            let current_steps = self.steps;
            for inst in &mut theta.body {
                theta.body_inst_count += 1;

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
                            tag: AssignTag::InputParam(Variance::Variant { .. })
                                | AssignTag::InputParam(Variance::Invariant),
                            ..
                        })
                    )
                {
                    continue;
                }

                self.handle(inst)?;
            }

            // If the body doesn't change our step limit at all, we want to still take steps
            // to make sure we don't infinitely loop
            if self.steps == current_steps {
                self.steps += 1;
            }

            let cond = theta.cond.as_ref().expect("expected a theta condition");
            let should_break = !self.get_bool(cond);
            tracing::trace!(
                "theta condition: {}, iteration: {}, value: {:?}",
                should_break,
                iter,
                cond,
            );

            if should_break {
                for (&output, value) in &theta.outputs {
                    self.values.insert(output, self.get_const(value).clone());
                }

                tracing::trace!(
                    values = ?theta
                        .outputs
                        .iter()
                        .map(|(output, value)| (output, self.get_const(value)))
                        .collect::<HashMap<_, _>>(),
                    "theta output values",
                );

                break;
            }

            for (output, value) in &theta.outputs {
                let input_var = theta.output_feedback[output];
                let feedback_value = self.get_const(value).clone();

                let old_value = self.values.insert(input_var, feedback_value.clone());
                tracing::trace!(
                    ?old_value,
                    "feeding back {:?} from {:?} to {:?}",
                    feedback_value,
                    output,
                    input_var,
                );
            }

            tracing::trace!(
                values = ?theta
                    .outputs
                    .iter()
                    .map(|(output, value)| (theta.output_feedback[output], (output, self.get_const(value))))
                    .collect::<HashMap<_, _>>(),
                "theta feedback values",
            );

            iter += 1;
        }

        Ok(())
    }

    fn gamma(&mut self, gamma: &mut Gamma) -> Result<()> {
        self.stats.branches += 1;

        let cond = self.get_bool(&gamma.cond);
        let branch = if cond {
            gamma.true_branches += 1;
            &mut gamma.truthy
        } else {
            gamma.false_branches += 1;
            &mut gamma.falsy
        };

        for inst in branch {
            self.handle(inst)?;
        }

        Ok(())
    }

    fn store(&mut self, store: &mut Store) {
        self.stats.stores += 1;
        store.stores += 1;

        let ptr = self.get_ptr(&store.ptr);
        let value = self.get_byte(&store.value);
        tracing::trace!(
            "stored {} to {} (previous value: {})",
            ptr,
            value,
            self.tape[ptr],
        );

        self.tape[ptr] = value;
    }

    fn load(&mut self, load: &Load) -> Const {
        self.stats.loads += 1;

        let ptr = self.get_ptr(&load.ptr);
        tracing::trace!("loaded {} from {}", self.tape[ptr], ptr);

        Const::Byte(self.tape[ptr])
    }

    fn eq(&self, eq: &Eq) -> Const {
        let (lhs, rhs) = (self.get_byte(&eq.lhs), self.get_byte(&eq.rhs));
        tracing::trace!(
            lhs = ?self.get_const(&eq.lhs),
            rhs = ?self.get_const(&eq.rhs),
            "({:?} == {:?}) is {}",
            lhs,
            rhs,
            lhs == rhs,
        );

        Const::Bool(lhs == rhs)
    }

    fn add(&self, add: &Add) -> Const {
        let (lhs, rhs) = (self.get_const(&add.lhs), self.get_const(&add.rhs));
        lhs + rhs
    }

    fn not(&self, not: &Not) -> Const {
        !self.get_const(&not.value)
    }

    fn neg(&self, neg: &Neg) -> Const {
        -self.get_const(&neg.value)
    }

    fn get_const<'b>(&'a self, value: &'b Value) -> &'b Const
    where
        'a: 'b,
    {
        match value {
            Value::Var(var) => self
                .values
                .get(var)
                .unwrap_or_else(|| panic!("expected a value for the given variable {:?}", var)),
            Value::Const(constant) => constant,
            Value::Missing => panic!("expected a value, got missing"),
        }
    }

    fn get_byte(&self, value: &Value) -> u8 {
        self.get_const(value)
            .convert_to_u8()
            .expect("expected a u8-convertible constant")
    }

    fn get_ptr(&self, value: &Value) -> usize {
        let ptr = self
            .get_const(value)
            .convert_to_i32()
            .expect("expected an i32-convertible constant") as usize;

        // We have to wrap the pointer into the tape's address space
        ptr.rem_euclid(self.tape.len()) as usize
    }

    fn get_bool(&self, value: &Value) -> bool {
        self.get_const(value)
            .as_bool()
            .expect("expected a boolean constant")
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
}

impl Default for ExecutionStats {
    fn default() -> Self {
        Self::new()
    }
}

test_opts! {
    addition_loop,
    passes = [],
    input = [50, 24],
    output = [74, 0],
    |graph, effect| {
        let ptr = graph.int(0).value();
        compile_brainfuck_into(",>,[-<+>]<.>.", graph, ptr, effect).1
    },
}

test_opts! {
    branching,
    passes = [],
    input = [50, 24, 10, 0],
    output = [50, 24, 10, 0],
    |graph, effect| {
        let ptr = graph.int(0).value();
        compile_brainfuck_into("+[>,.]", graph, ptr, effect).1
    },
}
