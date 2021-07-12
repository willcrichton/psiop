use crate::lang::*;
use num::BigRational;
use std::collections::{HashMap, HashSet};
use std::fmt;

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

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Dist {
  DVar(Var),
  Rat(BigRational),
  E,
  Pi,
  Neg(Box<Dist>),
  Bin(Box<Dist>, Box<Dist>, BinOp),
  Exp(Box<Dist>, Box<Dist>),
  Log(Box<Dist>),
  Pred(Box<Dist>, Box<Dist>, PredOp),
  Integral(Var, Box<Dist>),
  Func(Var, Box<Dist>),
  Distr(Var, Box<Dist>),
  Delta(Box<Dist>, Var),
  Lebesgue(Var),
  Pdf(Box<Dist>, Var),
  App(Box<Dist>, Box<Dist>),
  Record(HashMap<Var, Dist>),
  Proj(Box<Dist>, Var),
  Tuple(Vec<Dist>),
}

pub trait Visitor {
  fn visit_var(&mut self, v: Var) {
    self.super_visit_var(v)
  }

  fn super_visit_var(&mut self, _v: Var) {}

  fn visit_dvar(&mut self, v: Var) {
    self.super_visit_dvar(v);
  }

  fn super_visit_dvar(&mut self, v: Var) {
    self.visit_var(v);
  }

  fn visit_rat(&mut self, rat: &BigRational) {
    self.super_visit_rat(rat)
  }

  fn super_visit_rat(&mut self, _rat: &BigRational) {}

  fn visit_bin(&mut self, d1: &Dist, d2: &Dist, op: BinOp) {
    self.super_visit_bin(d1, d2, op)
  }

  fn super_visit_bin(&mut self, d1: &Dist, d2: &Dist, _op: BinOp) {
    self.visit(d1);
    self.visit(d2);
  }

  fn visit_integral(&mut self, x: Var, d: &Dist) {
    self.super_visit_integral(x, d)
  }

  fn super_visit_integral(&mut self, x: Var, d: &Dist) {
    self.visit_var(x);
    self.visit(d);
  }

  fn visit_func(&mut self, x: Var, d: &Dist) {
    self.super_visit_func(x, d)
  }

  fn super_visit_func(&mut self, x: Var, d: &Dist) {
    self.visit_var(x);
    self.visit(d);
  }

  fn visit_distr(&mut self, x: Var, d: &Dist) {
    self.super_visit_distr(x, d)
  }

  fn super_visit_distr(&mut self, x: Var, d: &Dist) {
    self.visit_var(x);
    self.visit(d);
  }

  fn visit_delta(&mut self, d: &Dist, x: Var) {
    self.super_visit_delta(d, x)
  }

  fn super_visit_delta(&mut self, d: &Dist, x: Var) {
    self.visit(d);
    self.visit_var(x);
  }

  fn visit_pdf(&mut self, d: &Dist, x: Var) {
    self.super_visit_pdf(d, x)
  }

  fn super_visit_pdf(&mut self, d: &Dist, x: Var) {
    self.visit(d);
    self.visit_var(x);
  }

  fn visit_app(&mut self, e1: &Dist, e2: &Dist) {
    self.super_visit_app(e1, e2)
  }

  fn super_visit_app(&mut self, e1: &Dist, e2: &Dist) {
    self.visit(e1);
    self.visit(e2);
  }

  fn visit_tuple(&mut self, ds: &Vec<Dist>) {
    self.super_visit_tuple(ds)
  }

  fn super_visit_tuple(&mut self, ds: &Vec<Dist>) {
    for d in ds {
      self.visit(d);
    }
  }

  fn visit_proj(&mut self, d: &Dist, x: Var) {
    self.super_visit_proj(d, x);
  }

  fn super_visit_proj(&mut self, d: &Dist, x: Var) {
    self.visit(d);
    self.visit_var(x);
  }

  fn visit_record(&mut self, h: &HashMap<Var, Dist>) {
    self.super_visit_record(h)
  }

  fn super_visit_record(&mut self, h: &HashMap<Var, Dist>) {
    for v in h.values() {
      self.visit(v);
    }
  }

  fn visit_lebesgue(&mut self, x: Var) {
    self.super_visit_lebesgue(x)
  }

  fn super_visit_lebesgue(&mut self, x: Var) {}

  fn visit_pred(&mut self, d1: &Dist, d2: &Dist, op: PredOp) {
    self.super_visit_pred(d1, d2, op);
  }

  fn super_visit_pred(&mut self, d1: &Dist, d2: &Dist, op: PredOp) {
    self.visit(d1);
    self.visit(d2);
  }

  fn visit(&mut self, d: &Dist) {
    match d {
      Dist::DVar(v) => self.visit_dvar(*v),
      Dist::Rat(rat) => self.visit_rat(rat),
      Dist::Bin(box d1, box d2, op) => self.visit_bin(d1, d2, *op),
      Dist::Integral(x, box d) => self.visit_integral(*x, d),
      Dist::Func(x, box d) => self.visit_func(*x, d),
      Dist::Distr(x, box d) => self.visit_distr(*x, d),
      Dist::Delta(box d, x) => self.visit_delta(d, *x),
      Dist::Pdf(box d, x) => self.visit_pdf(d, *x),
      Dist::App(box e1, box e2) => self.visit_app(e1, e2),
      Dist::Tuple(es) => self.visit_tuple(es),
      Dist::Proj(d, x) => self.visit_proj(d, *x),
      Dist::Record(h) => self.visit_record(h),
      Dist::Lebesgue(x) => self.visit_lebesgue(*x),
      Dist::Pred(box d1, box d2, op) => self.visit_pred(d1, d2, *op),
      _ => todo!("{}", d),
    }
  }
}

pub trait Folder {
  fn fold_var(&mut self, v: Var) -> Var {
    self.super_fold_var(v)
  }

  fn super_fold_var(&mut self, v: Var) -> Var {
    v
  }

  fn fold_dvar(&mut self, v: Var) -> Dist {
    self.super_fold_dvar(v)
  }

  fn super_fold_dvar(&mut self, v: Var) -> Dist {
    Dist::DVar(self.fold_var(v))
  }

  fn fold_rat(&mut self, rat: &BigRational) -> Dist {
    self.super_fold_rat(rat)
  }

  fn super_fold_rat(&mut self, rat: &BigRational) -> Dist {
    Dist::Rat(rat.clone())
  }

  fn fold_bin(&mut self, d1: &Dist, d2: &Dist, op: BinOp) -> Dist {
    self.super_fold_bin(d1, d2, op)
  }

  fn super_fold_bin(&mut self, d1: &Dist, d2: &Dist, op: BinOp) -> Dist {
    Dist::Bin(box self.fold(d1), box self.fold(d2), op)
  }

  fn fold_integral(&mut self, x: Var, d: &Dist) -> Dist {
    self.super_fold_integral(x, d)
  }

  fn super_fold_integral(&mut self, x: Var, d: &Dist) -> Dist {
    Dist::Integral(self.fold_var(x), box self.fold(d))
  }

  fn fold_func(&mut self, x: Var, d: &Dist) -> Dist {
    self.super_fold_func(x, d)
  }

  fn super_fold_func(&mut self, x: Var, d: &Dist) -> Dist {
    Dist::Func(self.fold_var(x), box self.fold(d))
  }

  fn fold_distr(&mut self, x: Var, d: &Dist) -> Dist {
    self.super_fold_distr(x, d)
  }

  fn super_fold_distr(&mut self, x: Var, d: &Dist) -> Dist {
    Dist::Distr(self.fold_var(x), box self.fold(d))
  }

  fn fold_delta(&mut self, d: &Dist, x: Var) -> Dist {
    self.super_fold_delta(d, x)
  }

  fn super_fold_delta(&mut self, d: &Dist, x: Var) -> Dist {
    Dist::Delta(box self.fold(d), self.fold_var(x))
  }

  fn fold_pdf(&mut self, d: &Dist, x: Var) -> Dist {
    self.super_fold_pdf(d, x)
  }

  fn super_fold_pdf(&mut self, d: &Dist, x: Var) -> Dist {
    Dist::Pdf(box self.fold(d), self.fold_var(x))
  }

  fn fold_app(&mut self, e1: &Dist, e2: &Dist) -> Dist {
    self.super_fold_app(e1, e2)
  }

  fn super_fold_app(&mut self, e1: &Dist, e2: &Dist) -> Dist {
    Dist::App(box self.fold(e1), box self.fold(e2))
  }

  fn fold_tuple(&mut self, es: &Vec<Dist>) -> Dist {
    self.super_fold_tuple(es)
  }

  fn super_fold_tuple(&mut self, es: &Vec<Dist>) -> Dist {
    Dist::Tuple(es.iter().map(|e| self.fold(e)).collect())
  }

  fn fold_proj(&mut self, d: &Dist, x: Var) -> Dist {
    self.super_fold_proj(d, x)
  }

  fn super_fold_proj(&mut self, d: &Dist, x: Var) -> Dist {
    Dist::Proj(box self.fold(d), self.fold_var(x))
  }

  fn fold_record(&mut self, h: &HashMap<Var, Dist>) -> Dist {
    self.super_fold_record(h)
  }

  fn super_fold_record(&mut self, h: &HashMap<Var, Dist>) -> Dist {
    Dist::Record(
      h.iter()
        .map(|(k, v)| (self.fold_var(*k), self.fold(v)))
        .collect(),
    )
  }

  fn fold_lebesgue(&mut self, x: Var) -> Dist {
    self.super_fold_lebesgue(x)
  }

  fn super_fold_lebesgue(&mut self, x: Var) -> Dist {
    Dist::Lebesgue(self.fold_var(x))
  }

  fn fold_pred(&mut self, d1: &Dist, d2: &Dist, op: PredOp) -> Dist {
    self.super_fold_pred(d1, d2, op)
  }

  fn super_fold_pred(&mut self, d1: &Dist, d2: &Dist, op: PredOp) -> Dist {
    Dist::Pred(box self.fold(d1), box self.fold(d2), op)
  }

  fn fold(&mut self, d: &Dist) -> Dist {
    match d {
      Dist::DVar(v) => self.fold_dvar(*v),
      Dist::Rat(rat) => self.fold_rat(rat),
      Dist::Bin(box d1, box d2, op) => self.fold_bin(d1, d2, *op),
      Dist::Integral(x, box d) => self.fold_integral(*x, d),
      Dist::Func(x, box d) => self.fold_func(*x, d),
      Dist::Distr(x, box d) => self.fold_distr(*x, d),
      Dist::Delta(box d, x) => self.fold_delta(d, *x),
      Dist::Pdf(box d, x) => self.fold_pdf(d, *x),
      Dist::App(box e1, box e2) => self.fold_app(e1, e2),
      Dist::Tuple(es) => self.fold_tuple(es),
      Dist::Proj(d, x) => self.fold_proj(d, *x),
      Dist::Record(h) => self.fold_record(h),
      Dist::Lebesgue(x) => self.fold_lebesgue(*x),
      Dist::Pred(box d1, box d2, op) => self.fold_pred(d1, d2, *op),
      _ => todo!("{}", d),
    }
  }
}

struct FreeVars {
  fv: HashSet<Var>,
  bv: HashMap<Var, usize>,
}

impl Visitor for FreeVars {
  fn visit_func(&mut self, x: Var, d: &Dist) {
    *self.bv.entry(x).or_insert(0) += 1;
    self.super_visit_func(x, d);
    *self.bv.entry(x).or_insert(0) -= 1;
  }

  fn visit_distr(&mut self, x: Var, d: &Dist) {
    *self.bv.entry(x).or_insert(0) += 1;
    self.super_visit_distr(x, d);
    *self.bv.entry(x).or_insert(0) -= 1;
  }

  fn visit_integral(&mut self, x: Var, d: &Dist) {
    *self.bv.entry(x).or_insert(0) += 1;
    self.super_visit_integral(x, d);
    *self.bv.entry(x).or_insert(0) -= 1;
  }

  fn visit_var(&mut self, x: Var) {
    if *self.bv.entry(x).or_insert(0) == 0 {
      self.fv.insert(x);
    }
  }
}

struct Subst {
  src: Var,
  dst: Dist,
  fv: HashSet<Var>,
}

impl Folder for Subst {
  fn fold_var(&mut self, v: Var) -> Var {
    if v == self.src {
      match self.dst {
        Dist::DVar(v2) => v2,
        _ => panic!("subst non-var {} into {}", self.dst, v),
      }
    } else {
      v
    }
  }

  fn fold_dvar(&mut self, v: Var) -> Dist {
    if v == self.src {
      self.dst.clone()
    } else {
      Dist::DVar(v)
    }
  }

  fn fold_func(&mut self, x: Var, d: &Dist) -> Dist {
    if x == self.src {
      Dist::Func(x, box d.clone())
    } else if self.fv.contains(&x) {
      let xp = v(format!("{}'", x));
      self.super_fold_func(xp, &d.subst(x, Dist::DVar(xp)))
    } else {
      self.super_fold_func(x, d)
    }
  }

  fn fold_distr(&mut self, x: Var, d: &Dist) -> Dist {
    if x == self.src {
      Dist::Distr(x, box d.clone())
    } else if self.fv.contains(&x) {
      let xp = v(format!("{}'", x));
      self.super_fold_distr(xp, &d.subst(x, Dist::DVar(xp)))
    } else {
      self.super_fold_distr(x, d)
    }
  }

  fn fold_integral(&mut self, x: Var, d: &Dist) -> Dist {
    if x == self.src {
      Dist::Integral(x, box d.clone())
    } else if self.fv.contains(&x) {
      let xp = v(format!("{}'", x));
      self.super_fold_integral(xp, &d.subst(x, Dist::DVar(xp)))
    } else {
      self.super_fold_integral(x, d)
    }
  }
}

impl Dist {
  pub fn is_value(&self, int_vars: &HashSet<Var>) -> bool {
    use Dist::*;
    match self {
      DVar(v) => !int_vars.contains(v),
      Rat(..) | Func(..) | Distr(..) => true,
      Tuple(ds) => ds.iter().all(|d| d.is_value(int_vars)),
      Bin(d1, d2, _) => d1.is_value(int_vars) && d2.is_value(int_vars),
      _ => false,
    }
  }

  pub fn free_vars(&self) -> HashSet<Var> {
    let mut pass = FreeVars {
      fv: HashSet::new(),
      bv: HashMap::new(),
    };
    pass.visit(self);
    pass.fv
  }

  pub fn subst(&self, x: Var, d: Dist) -> Dist {
    let fv = d.free_vars();
    let mut pass = Subst { src: x, dst: d, fv };
    pass.fold(self)
  }

  pub fn bin_many(mut args: Vec<Dist>, op: BinOp) -> Dist {
    assert!(args.len() > 0);
    let first = args.remove(0);
    args
      .into_iter()
      .fold(first, |acc, d| Dist::Bin(box acc, box d, op))
  }

  // pub fn aequiv(&self, other: &Dist) -> bool {
  //   use Dist::*;
  //   match self {
  //     DVar(_) | Rat(_, _) | E | Pi => self == other,
  //     _ => match (self, other) {
  //       (Neg(e1), Neg(e2)) | (Log(e1), Log(e2)) => e1.aequiv(e2),
  //       (Bin(ds1, op1), Bin(ds2, op2)) => {
  //         op1 == op2
  //           && ds1.len() == ds2.len()
  //           && ds1.iter().zip(ds2.iter()).all(|(d1, d2)| d1.aequiv(d2))
  //       }

  //       (Integral(x1, e1), Integral(x2, e2))
  //       | (Func(x1, e1), Func(x2, e2))
  //       | (Distr(x1, e1), Func(x2, e2)) => e2.subst(*x2, DVar(*x1)).aequiv(e1),

  //       (Delta(e1, x1), Delta(e2, x2)) | (Pdf(e1, x1), Pdf(e2, x2)) => {
  //         x1 == x2 && e1.aequiv(e2)
  //       }
  //       (App(e11, e12), App(e21, e22)) => e11.aequiv(e21) && e12.aequiv(e22),
  //       _ => {
  //         println!("false at {:?}, {:?}", self, other);
  //         false
  //       }
  //     },
  //   }
  // }
}

impl fmt::Display for Dist {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    match self {
      Dist::DVar(v) => v.fmt(f),
      Dist::Rat(rat) => write!(f, "{}", rat),
      Dist::Func(var, d) => write!(f, "(λ{}. {})", var, d),
      Dist::Distr(var, d) => write!(f, "(Λ{}. {})", var, d),
      Dist::Delta(d, var) => write!(f, "δ({})⟦{}⟧", d, var),
      Dist::Pdf(d, var) => write!(f, "{}⟦{}⟧", d, var),
      Dist::Integral(var, d) => write!(f, "(∫d{} {})", var, d),
      Dist::Bin(d1, d2, op) => {
        write!(f, "{} {} {}", d1, op, d2)
      }
      Dist::App(d1, d2) => write!(f, "{}({})", d1, d2),
      Dist::Tuple(ds) => {
        write!(f, "(")?;
        for d in ds.iter().take(ds.len() - 1) {
          write!(f, "{}, ", d)?;
        }
        if ds.len() > 0 {
          write!(f, "{}", ds.last().unwrap())?;
        }
        write!(f, ")")
      }
      Dist::Proj(d, x) => write!(f, "{}.{}", d, x),
      Dist::Record(h) => {
        write!(f, "{{")?;
        let kvs = h.iter().collect::<Vec<_>>();
        if kvs.len() > 0 {
          for (k, v) in kvs.iter().take(kvs.len() - 1) {
            write!(f, "{}: {}, ", k, v)?;
          }
          let (k, v) = kvs.last().unwrap();
          write!(f, "{}: {})", k, v)?;
        }
        write!(f, "}}")
      }
      Dist::Lebesgue(x) => write!(f, "λ⟦{}⟧", x),
      Dist::Pred(d1, d2, op) => write!(f, "[{} {} {}]", d1, op, d2),
      _ => todo!(),
    }
  }
}
