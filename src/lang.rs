use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use string_interner::{DefaultSymbol as Symbol, StringInterner};

thread_local! {
  pub static INTERNER: RefCell<StringInterner> = RefCell::new(StringInterner::default());
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Var(Symbol);

impl Var {
  pub fn new(t: impl Into<String>) -> Self {
    INTERNER.with(|interner| {
      let symbol = interner.borrow_mut().get_or_intern(t.into());
      Var(symbol)
    })
  }
}

pub fn v(t: impl Into<String>) -> Var {
  Var::new(t)
}

impl fmt::Display for Var {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    INTERNER.with(|interner| {
      let interner = interner.borrow();
      let s = interner.resolve(self.0).unwrap();
      write!(f, "{}", s)
    })
  }
}

impl fmt::Debug for Var {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "Var({})", self)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinOp {
  Add,
  Mul,
  Div,
  Sub,
}

impl BinOp {
  pub fn inverse(&self) -> Self {
    match self {
      BinOp::Add => BinOp::Sub,
      BinOp::Sub => BinOp::Add,
      BinOp::Mul => BinOp::Div,
      BinOp::Div => BinOp::Mul,
    }
  }
}

impl fmt::Display for BinOp {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    match self {
      BinOp::Add => write!(f, "+"),
      BinOp::Mul => write!(f, "*"),
      BinOp::Div => write!(f, "/"),
      BinOp::Sub => write!(f, "-"),
    }
  }
}

#[derive(Debug)]
pub enum Expr {
  Int(isize),
  EVar(Var),
  Bin(Box<Expr>, Box<Expr>, BinOp),
  App(Box<Expr>, Box<Expr>),
  Tuple(Vec<Expr>),
}

pub enum Stmt {
  Assign(Var, Box<Expr>),
  AssignEx(Box<Expr>, Box<Expr>),
  If(Box<Expr>, Vec<Stmt>, Vec<Stmt>),
}

pub type Prog = Vec<Stmt>;

#[derive(Default, Clone)]
pub struct BoundVars(HashMap<Var, usize>);
impl BoundVars {
  pub fn bind(&mut self, x: Var) {
    *self.0.entry(x).or_insert(0) += 1;
  }

  pub fn unbind(&mut self, x: Var) {
    *self.0.entry(x).or_insert(0) -= 1;
  }

  pub fn bound_vars(&self) -> HashSet<Var> {
    self
      .0
      .iter()
      .filter_map(|(k, v)| (*v > 0).then(|| *k))
      .collect()
  }
}
