use string_interner::{StringInterner, DefaultSymbol as Symbol};
use std::cell::RefCell;
use std::fmt;

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

impl fmt::Debug for Var {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    INTERNER.with(|interner| {
      let interner = interner.borrow();
      let s = interner.resolve(self.0).unwrap();
      write!(f, "{}", s)      
    })
  }
}


#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
  Add, Mul
}

impl fmt::Debug for BinOp {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    match self {
      BinOp::Add => write!(f, "+"),
      BinOp::Mul => write!(f, "*"),
    }
  }
}

#[derive(Debug)]
pub enum Expr {
  Int(isize),
  EVar(Var),
  Bin(Box<Expr>, Box<Expr>, BinOp),  
  App(Box<Expr>, Box<Expr>),
  Tuple(Vec<Expr>)
}

pub enum Stmt {
  Assign(Var, Box<Expr>),
  AssignEx(Box<Expr>, Box<Expr>),
  If(Box<Expr>, Vec<Stmt>, Vec<Stmt>)
}

pub type Prog = Vec<Stmt>;