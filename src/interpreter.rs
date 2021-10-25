use crate::{
    ir::{
        Add, Assign, Block, Call, Const, Eq, Expr, Gamma, Instruction, Load, Neg, Not, Store,
        Theta, Value, VarId,
    },
    utils::AssertNone,
};
use std::collections::HashMap;

pub struct Machine<'a> {
    tape: Vec<u8>,
    values: HashMap<VarId, Const>,
    input: Box<dyn FnMut() -> u8 + 'a>,
    output: Box<dyn FnMut(u8) + 'a>,
}

impl<'a> Machine<'a> {
    pub fn new<I, O>(length: usize, input: I, output: O) -> Self
    where
        I: FnMut() -> u8 + 'a,
        O: FnMut(u8) + 'a,
    {
        Self {
            tape: vec![0; length],
            values: HashMap::new(),
            input: Box::new(input),
            output: Box::new(output),
        }
    }

    #[tracing::instrument(skip(self, block))]
    pub fn execute(&mut self, block: &Block) -> &[u8] {
        for inst in block.iter() {
            self.handle(inst);
        }

        &self.tape
    }

    fn handle(&mut self, inst: &Instruction) {
        match inst {
            Instruction::Call(call) => {
                self.call(call).debug_unwrap_none();
            }
            Instruction::Assign(assign) => self.assign(assign),
            Instruction::Theta(theta) => self.theta(theta),
            Instruction::Gamma(gamma) => self.gamma(gamma),
            Instruction::Store(store) => self.store(store),
        }
    }

    fn eval(&mut self, expr: &Expr) -> Const {
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

    fn call(&mut self, call: &Call) -> Option<Const> {
        match &*call.function {
            "input" => {
                if !call.args.is_empty() {
                    panic!("expected zero args for input call")
                }

                Some(Const::Byte((self.input)()))
            }

            "output" => {
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

    fn assign(&mut self, assign: &Assign) {
        let value = self.eval(&assign.value);
        tracing::trace!("assigned {:?} to {}", value, assign.var);

        // Note: Double inserts are allowed here because of loops
        self.values.insert(assign.var, value);
    }

    fn theta(&mut self, theta: &Theta) {
        let mut iter = 0;
        loop {
            for inst in &theta.body {
                self.handle(inst);
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
                break;
            }
            iter += 1;

            // FIXME: Set input ports to the proper output port's values to
            //        allow loop values to propagate within the loop
        }
    }

    fn gamma(&mut self, gamma: &Gamma) {
        let cond = self.get_bool(&gamma.cond);
        let branch = if cond { &gamma.truthy } else { &gamma.falsy };

        for inst in branch {
            self.handle(inst);
        }
    }

    fn store(&mut self, store: &Store) {
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

    fn load(&self, load: &Load) -> Const {
        let ptr = self.get_ptr(&load.ptr);
        tracing::trace!("loaded {} from {}", self.tape[ptr], ptr);

        Const::Byte(self.tape[ptr])
    }

    fn eq(&self, eq: &Eq) -> Const {
        let (lhs, rhs) = (self.get_const(&eq.lhs), self.get_const(&eq.rhs));
        tracing::trace!("({:?} == {:?}) is {}", lhs, rhs, lhs.equal_values(rhs));

        Const::Bool(lhs.equal_values(rhs))
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
