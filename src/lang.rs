use num::BigRational;
use pretty::RcDoc;
use regex::Regex;
use std::cell::RefCell;
use std::collections::HashMap;
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

  pub fn to_string(&self) -> String {
    INTERNER.with(|interner| {
      let interner = interner.borrow();
      let s = interner.resolve(self.0).unwrap();
      s.to_string()
    })
  }

  pub fn fresh(&self) -> Var {
    let s = INTERNER.with(|interner| {
      let interner = interner.borrow();
      let re = Regex::new(r"^(.+)(\d*)$").unwrap();
      let caps = re.captures(interner.resolve(self.0).unwrap()).unwrap();
      let (s, mut i) = (
        caps.get(1).unwrap().as_str(),
        caps
          .get(2)
          .unwrap()
          .as_str()
          .parse::<usize>()
          .map(|i| i + 1)
          .unwrap_or(1),
      );

      loop {
        let s2 = format!("{}{}", s, i);
        if interner.get(&s2).is_none() {
          return s2;
        }
        i += 1;
      }
    });
    Var::new(s)
  }

  pub fn to_doc(&self) -> RcDoc<()> {
    RcDoc::as_string(format!("{}", self))
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

#[derive(Default, Clone)]
pub struct BoundVars(HashMap<Var, usize>);
impl BoundVars {
  pub fn bind(&mut self, x: Var) {
    *self.0.entry(x).or_insert(0) += 1;
  }

  pub fn unbind(&mut self, x: Var) {
    *self.0.entry(x).or_insert(0) -= 1;
  }

  pub fn is_bound(&self, x: Var) -> bool {
    self.0.contains_key(&x)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredOp {
  Eq,
  Leq,
  Neq,
  Le,
}

impl fmt::Display for PredOp {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    use PredOp::*;
    write!(
      f,
      "{}",
      match self {
        Eq => "=",
        Leq => "≤",
        Neq => "≠",
        Le => "<",
      }
    )
  }
}

#[derive(Debug)]
pub enum Expr {
  Rat(BigRational),
  EVar(Var),
  Bin(Box<Expr>, Box<Expr>, BinOp),
  Pred(Box<Expr>, Box<Expr>, PredOp),
  App(Box<Expr>, Box<Expr>),
  Tuple(Vec<Expr>),
  Proj(Box<Expr>, Var),
}

#[derive(Debug)]
pub enum Stmt {
  Init(Var, Expr),
  Assign(Expr, Expr),
  If(Expr, Box<Stmt>, Box<Stmt>),
  Observe(Expr),
  Seq(Box<Stmt>, Box<Stmt>),
}

pub type Prog = Vec<Stmt>;
