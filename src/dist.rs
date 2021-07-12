use crate::lang::*;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredOp {
  Eq,
  Leq,
  Neq,
  Le,
}

#[derive(Clone, PartialEq, Eq)]
pub enum Dist {
  DVar(Var),
  Rat(isize, usize),
  E,
  Pi,
  Neg(Box<Dist>),
  Bin(Vec<Dist>, BinOp),
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

impl Dist {
  pub fn empty_state() -> Self {
    Dist::Record(HashMap::new())
  }
}

pub trait Visitor {
  fn visit_var(&mut self, v: Var) {
    self.super_visit_var(v)
  }

  fn super_visit_var(&mut self, v: Var) {}

  fn visit_rat(&mut self, num: isize, denom: usize) {
    self.super_visit_rat(num, denom)
  }

  fn super_visit_rat(&mut self, num: isize, denom: usize) {}

  fn visit_bin(&mut self, ds: &Vec<Dist>, op: BinOp) {
    self.super_visit_bin(ds, op)
  }

  fn super_visit_bin(&mut self, ds: &Vec<Dist>, op: BinOp) {
    for d in ds.iter() {
      self.visit(d);
    }
  }

  fn visit_integral(&mut self, x: Var, d: &Dist) {
    self.super_visit_integral(x, d)
  }

  fn super_visit_integral(&mut self, x: Var, d: &Dist) {
    self.visit(d);
  }

  fn visit_func(&mut self, x: Var, d: &Dist) {
    self.super_visit_func(x, d)
  }

  fn super_visit_func(&mut self, x: Var, d: &Dist) {
    self.visit(d);
  }

  fn visit_distr(&mut self, x: Var, d: &Dist) {
    self.super_visit_distr(x, d)
  }

  fn super_visit_distr(&mut self, x: Var, d: &Dist) {
    self.visit(d);
  }

  fn visit_delta(&mut self, d: &Dist, x: Var) {
    self.super_visit_delta(d, x)
  }

  fn super_visit_delta(&mut self, d: &Dist, x: Var) {
    self.visit(d);
  }

  fn visit_pdf(&mut self, d: &Dist, x: Var) {
    self.super_visit_pdf(d, x)
  }

  fn super_visit_pdf(&mut self, d: &Dist, x: Var) {
    self.visit(d);
  }

  fn visit_app(&mut self, e1: &Dist, e2: &Dist) {
    self.super_visit_app(e1, e2)
  }

  fn super_visit_app(&mut self, e1: &Dist, e2: &Dist) {
    self.visit(e1);
    self.visit(e2);
  }

  fn visit(&mut self, d: &Dist) {
    match d {
      Dist::DVar(v) => self.visit_var(*v),
      Dist::Rat(num, denom) => self.visit_rat(*num, *denom),
      Dist::Bin(ds, op) => self.visit_bin(&ds, *op),
      Dist::Integral(x, box d) => self.visit_integral(*x, d),
      Dist::Func(x, box d) => self.visit_func(*x, d),
      Dist::Distr(x, box d) => self.visit_distr(*x, d),
      Dist::Delta(box d, x) => self.visit_delta(d, *x),
      Dist::Pdf(box d, x) => self.visit_pdf(d, *x),
      Dist::App(box e1, box e2) => self.visit_app(e1, e2),
      _ => todo!("{:?}", d),
    }
  }
}

pub trait Folder {
  fn fold_var(&mut self, v: Var) -> Dist {
    self.super_fold_var(v)
  }

  fn super_fold_var(&mut self, v: Var) -> Dist {
    Dist::DVar(v)
  }

  fn fold_rat(&mut self, num: isize, denom: usize) -> Dist {
    self.super_fold_rat(num, denom)
  }

  fn super_fold_rat(&mut self, num: isize, denom: usize) -> Dist {
    Dist::Rat(num, denom)
  }

  fn fold_bin(&mut self, ds: &Vec<Dist>, op: BinOp) -> Dist {
    self.super_fold_bin(ds, op)
  }

  fn super_fold_bin(&mut self, ds: &Vec<Dist>, op: BinOp) -> Dist {
    Dist::Bin(ds.iter().map(|d| self.fold(d)).collect(), op)
  }

  fn fold_integral(&mut self, x: Var, d: &Dist) -> Dist {
    self.super_fold_integral(x, d)
  }

  fn super_fold_integral(&mut self, x: Var, d: &Dist) -> Dist {
    Dist::Integral(x, box self.fold(d))
  }

  fn fold_func(&mut self, x: Var, d: &Dist) -> Dist {
    self.super_fold_func(x, d)
  }

  fn super_fold_func(&mut self, x: Var, d: &Dist) -> Dist {
    Dist::Func(x, box self.fold(d))
  }

  fn fold_distr(&mut self, x: Var, d: &Dist) -> Dist {
    self.super_fold_distr(x, d)
  }

  fn super_fold_distr(&mut self, x: Var, d: &Dist) -> Dist {
    Dist::Distr(x, box self.fold(d))
  }

  fn fold_delta(&mut self, d: &Dist, x: Var) -> Dist {
    self.super_fold_delta(d, x)
  }

  fn super_fold_delta(&mut self, d: &Dist, x: Var) -> Dist {
    Dist::Delta(box self.fold(d), x)
  }

  fn fold_pdf(&mut self, d: &Dist, x: Var) -> Dist {
    self.super_fold_pdf(d, x)
  }

  fn super_fold_pdf(&mut self, d: &Dist, x: Var) -> Dist {
    Dist::Pdf(box self.fold(d), x)
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
    Dist::Proj(box self.fold(d), x)
  }

  fn fold_record(&mut self, h: &HashMap<Var, Dist>) -> Dist {
    self.super_fold_record(h)
  }

  fn super_fold_record(&mut self, h: &HashMap<Var, Dist>) -> Dist {
    Dist::Record(h.iter().map(|(k, v)| { (*k, self.fold(v)) }).collect())
  }

  fn fold(&mut self, d: &Dist) -> Dist {
    match d {
      Dist::DVar(v) => self.fold_var(*v),
      Dist::Rat(num, denom) => self.fold_rat(*num, *denom),
      Dist::Bin(ds, op) => self.fold_bin(&ds, *op),
      Dist::Integral(x, box d) => self.fold_integral(*x, d),
      Dist::Func(x, box d) => self.fold_func(*x, d),
      Dist::Distr(x, box d) => self.fold_distr(*x, d),
      Dist::Delta(box d, x) => self.fold_delta(d, *x),
      Dist::Pdf(box d, x) => self.fold_pdf(d, *x),
      Dist::App(box e1, box e2) => self.fold_app(e1, e2),
      Dist::Tuple(es) => self.fold_tuple(es),
      Dist::Proj(d, x) => self.fold_proj(d, *x),
      Dist::Record(h) => self.fold_record(h),
      _ => todo!("{:?}", d),
    }
  }
}

struct Subst {
  src: Var,
  dst: Dist,
}
impl Folder for Subst {
  fn fold_var(&mut self, v: Var) -> Dist {
    if v == self.src {
      self.dst.clone()
    } else {
      Dist::DVar(v)
    }
  }

  fn fold_delta(&mut self, d: &Dist, x: Var) -> Dist {
    if x == self.src {
      match self.dst {
        Dist::DVar(y) => Dist::Delta(box d.clone(), y),
        _ => panic!(
          "Substituting into delta {:?} with non-variable {:?}",
          Dist::Delta(box d.clone(), x),
          self.dst
        ),
      }
    } else {
      self.super_fold_delta(d, x)
    }
  }

  fn fold_func(&mut self, x: Var, d: &Dist) -> Dist {
    if x == self.src {
      Dist::Func(x, box d.clone())
    } else {
      self.super_fold_func(x, d)
    }
  }
}

impl Dist {
  pub fn subst(&self, x: Var, d: Dist) -> Dist {
    let mut pass = Subst { src: x, dst: d };
    pass.fold(self)
  }

  pub fn aequiv(&self, other: &Dist) -> bool {
    use Dist::*;
    match self {
      DVar(_) | Rat(_, _) | E | Pi => self == other,
      _ => match (self, other) {
        (Neg(e1), Neg(e2)) | (Log(e1), Log(e2)) => e1.aequiv(e2),
        (Bin(ds1, op1), Bin(ds2, op2)) => {
          op1 == op2
            && ds1.len() == ds2.len()
            && ds1.iter().zip(ds2.iter()).all(|(d1, d2)| d1.aequiv(d2))
        }

        (Integral(x1, e1), Integral(x2, e2))
        | (Func(x1, e1), Func(x2, e2))
        | (Distr(x1, e1), Func(x2, e2)) => e2.subst(*x2, DVar(*x1)).aequiv(e1),

        (Delta(e1, x1), Delta(e2, x2)) | (Pdf(e1, x1), Pdf(e2, x2)) => {
          x1 == x2 && e1.aequiv(e2)
        }
        (App(e11, e12), App(e21, e22)) => e11.aequiv(e21) && e12.aequiv(e22),
        _ => {
          println!("false at {:?}, {:?}", self, other);
          false
        }
      },
    }
  }
}

impl fmt::Debug for Dist {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    match self {
      Dist::DVar(v) => v.fmt(f),
      Dist::Rat(num, denom) => write!(f, "{}", (*num as f64) / (*denom as f64)),
      Dist::Func(var, d) => write!(f, "(λ{:?}. {:?})", var, d),
      Dist::Distr(var, d) => write!(f, "(Λ{:?}. {:?})", var, d),
      Dist::Delta(d, var) => write!(f, "δ({:?})⟦{:?}⟧", d, var),
      Dist::Pdf(d, var) => write!(f, "{:?}⟦{:?}⟧", d, var),
      Dist::Integral(var, d) => write!(f, "∫d{:?} {:?}", var, d),
      Dist::Bin(ds, op) => {
        for d in ds.iter().take(ds.len() - 1) {
          write!(f, "{:?} {:?} ", d, op)?;
        }
        write!(f, "{:?}", ds.last().unwrap())
      }
      Dist::App(d1, d2) => write!(f, "{:?}({:?})", d1, d2),
      Dist::Tuple(ds) => {
        write!(f, "(")?;
        for d in ds.iter().take(ds.len() - 1) {
          write!(f, "{:?}, ", d)?;
        }
        if ds.len() > 0 {
          write!(f, "{:?}", ds.last().unwrap())?;
        }
        write!(f, ")")
      }
      Dist::Proj(d, x) => write!(f, "{:?}.{:?}", d, x),
      Dist::Record(h) => {
        write!(f, "{{")?;
        let kvs = h.iter().collect::<Vec<_>>();
        if kvs.len() > 0 {
          for (k, v) in kvs.iter().take(kvs.len() - 1) {
            write!(f, "{:?}: {:?}, ", k, v)?;
          }
          let (k, v) = kvs.last().unwrap();
          write!(f, "{:?}: {:?})", k, v);
        }
        write!(f, "}}")
      }
      _ => todo!(),
    }
  }
}
