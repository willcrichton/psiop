use crate::lang::*;
use num::BigRational;
use pretty::{Doc, RcDoc};
use std::collections::{HashMap, HashSet};
use std::fmt;

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
  RecSet(Box<Dist>, Var, Box<Dist>),
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

  fn visit_binder(&mut self, x: Var, d: &Dist) {
    self.super_visit_binder(x, d);
  }

  fn bound_vars(&mut self) -> Option<&mut BoundVars> {
    None
  }

  fn super_visit_binder(&mut self, x: Var, d: &Dist) {
    if let Some(bv) = self.bound_vars() {
      bv.bind(x);
    }
    match d {
      Dist::Func(_, box d) => self.visit_func(x, d),
      Dist::Distr(_, box d) => self.visit_distr(x, d),
      Dist::Integral(_, box d) => self.visit_integral(x, d),
      _ => unreachable!(),
    };
    if let Some(bv) = self.bound_vars() {
      bv.unbind(x);
    }
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

  fn super_visit_lebesgue(&mut self, x: Var) {
    self.visit_var(x);
  }

  fn visit_pred(&mut self, d1: &Dist, d2: &Dist, op: PredOp) {
    self.super_visit_pred(d1, d2, op);
  }

  fn super_visit_pred(&mut self, d1: &Dist, d2: &Dist, _op: PredOp) {
    self.visit(d1);
    self.visit(d2);
  }

  fn visit_rec_set(&mut self, d1: &Dist, x: Var, d2: &Dist) {
    self.super_visit_rec_set(d1, x, d2);
  }

  fn super_visit_rec_set(&mut self, d1: &Dist, x: Var, d2: &Dist) {
    self.visit(d1);
    self.visit_var(x);
    self.visit(d2);
  }

  fn visit(&mut self, d: &Dist) {
    self.super_visit(d)
  }

  fn super_visit(&mut self, d: &Dist) {
    match d {
      Dist::DVar(v) => self.visit_dvar(*v),
      Dist::Rat(rat) => self.visit_rat(rat),
      Dist::Bin(box d1, box d2, op) => self.visit_bin(d1, d2, *op),
      Dist::Integral(x, _) | Dist::Func(x, _) | Dist::Distr(x, _) => {
        self.visit_binder(*x, d)
      }
      Dist::Delta(box d, x) => self.visit_delta(d, *x),
      Dist::Pdf(box d, x) => self.visit_pdf(d, *x),
      Dist::App(box e1, box e2) => self.visit_app(e1, e2),
      Dist::Tuple(es) => self.visit_tuple(es),
      Dist::Proj(d, x) => self.visit_proj(d, *x),
      Dist::Record(h) => self.visit_record(h),
      Dist::Lebesgue(x) => self.visit_lebesgue(*x),
      Dist::Pred(box d1, box d2, op) => self.visit_pred(d1, d2, *op),
      Dist::RecSet(box d1, x, box d2) => self.visit_rec_set(d1, *x, d2),
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

  fn fold_rat(&mut self, rat: BigRational) -> Dist {
    self.super_fold_rat(rat)
  }

  fn super_fold_rat(&mut self, rat: BigRational) -> Dist {
    Dist::Rat(rat.clone())
  }

  fn fold_bin(&mut self, d1: Dist, d2: Dist, op: BinOp) -> Dist {
    self.super_fold_bin(d1, d2, op)
  }

  fn super_fold_bin(&mut self, d1: Dist, d2: Dist, op: BinOp) -> Dist {
    Dist::Bin(box self.fold(d1), box self.fold(d2), op)
  }

  fn fold_integral(&mut self, x: Var, d: Dist) -> Dist {
    self.super_fold_integral(x, d)
  }

  fn super_fold_integral(&mut self, x: Var, d: Dist) -> Dist {
    Dist::Integral(self.fold_var(x), box self.fold(d))
  }

  fn fold_func(&mut self, x: Var, d: Dist) -> Dist {
    self.super_fold_func(x, d)
  }

  fn super_fold_func(&mut self, x: Var, d: Dist) -> Dist {
    Dist::Func(self.fold_var(x), box self.fold(d))
  }

  fn fold_distr(&mut self, x: Var, d: Dist) -> Dist {
    self.super_fold_distr(x, d)
  }

  fn super_fold_distr(&mut self, x: Var, d: Dist) -> Dist {
    Dist::Distr(self.fold_var(x), box self.fold(d))
  }

  fn fold_binder(&mut self, x: Var, d: Dist) -> Dist {
    self.super_fold_binder(x, d)
  }

  fn bound_vars(&mut self) -> Option<&mut BoundVars> {
    None
  }

  fn super_fold_binder(&mut self, x: Var, d: Dist) -> Dist {
    if let Some(bv) = self.bound_vars() {
      bv.bind(x);
    }
    let d2 = match d {
      Dist::Func(_, box d) => self.fold_func(x, d),
      Dist::Distr(_, box d) => self.fold_distr(x, d),
      Dist::Integral(_, box d) => self.fold_integral(x, d),
      _ => unreachable!(),
    };
    if let Some(bv) = self.bound_vars() {
      bv.unbind(x);
    }
    d2
  }

  fn fold_delta(&mut self, d: Dist, x: Var) -> Dist {
    self.super_fold_delta(d, x)
  }

  fn super_fold_delta(&mut self, d: Dist, x: Var) -> Dist {
    Dist::Delta(box self.fold(d), self.fold_var(x))
  }

  fn fold_pdf(&mut self, d: Dist, x: Var) -> Dist {
    self.super_fold_pdf(d, x)
  }

  fn super_fold_pdf(&mut self, d: Dist, x: Var) -> Dist {
    Dist::Pdf(box self.fold(d), self.fold_var(x))
  }

  fn fold_app(&mut self, e1: Dist, e2: Dist) -> Dist {
    self.super_fold_app(e1, e2)
  }

  fn super_fold_app(&mut self, e1: Dist, e2: Dist) -> Dist {
    Dist::App(box self.fold(e1), box self.fold(e2))
  }

  fn fold_tuple(&mut self, es: Vec<Dist>) -> Dist {
    self.super_fold_tuple(es)
  }

  fn super_fold_tuple(&mut self, es: Vec<Dist>) -> Dist {
    Dist::Tuple(es.into_iter().map(|e| self.fold(e)).collect())
  }

  fn fold_proj(&mut self, d: Dist, x: Var) -> Dist {
    self.super_fold_proj(d, x)
  }

  fn super_fold_proj(&mut self, d: Dist, x: Var) -> Dist {
    Dist::Proj(box self.fold(d), x)
  }

  fn fold_record(&mut self, h: HashMap<Var, Dist>) -> Dist {
    self.super_fold_record(h)
  }

  fn super_fold_record(&mut self, h: HashMap<Var, Dist>) -> Dist {
    Dist::Record(h.into_iter().map(|(k, v)| (k, self.fold(v))).collect())
  }

  fn fold_lebesgue(&mut self, x: Var) -> Dist {
    self.super_fold_lebesgue(x)
  }

  fn super_fold_lebesgue(&mut self, x: Var) -> Dist {
    Dist::Lebesgue(self.fold_var(x))
  }

  fn fold_pred(&mut self, d1: Dist, d2: Dist, op: PredOp) -> Dist {
    self.super_fold_pred(d1, d2, op)
  }

  fn super_fold_pred(&mut self, d1: Dist, d2: Dist, op: PredOp) -> Dist {
    Dist::Pred(box self.fold(d1), box self.fold(d2), op)
  }

  fn fold_rec_set(&mut self, d1: Dist, x: Var, d2: Dist) -> Dist {
    self.super_fold_rec_set(d1, x, d2)
  }

  fn super_fold_rec_set(&mut self, d1: Dist, x: Var, d2: Dist) -> Dist {
    Dist::RecSet(box self.fold(d1), x, box self.fold(d2))
  }

  fn fold(&mut self, d: Dist) -> Dist {
    match d {
      Dist::DVar(v) => self.fold_dvar(v),
      Dist::Rat(rat) => self.fold_rat(rat),
      Dist::Bin(box d1, box d2, op) => self.fold_bin(d1, d2, op),
      Dist::Integral(x, _) | Dist::Func(x, _) | Dist::Distr(x, _) => {
        self.fold_binder(x, d)
      }
      Dist::Delta(box d, x) => self.fold_delta(d, x),
      Dist::Pdf(box d, x) => self.fold_pdf(d, x),
      Dist::App(box e1, box e2) => self.fold_app(e1, e2),
      Dist::Tuple(es) => self.fold_tuple(es),
      Dist::Proj(box d, x) => self.fold_proj(d, x),
      Dist::Record(h) => self.fold_record(h),
      Dist::Lebesgue(x) => self.fold_lebesgue(x),
      Dist::Pred(box d1, box d2, op) => self.fold_pred(d1, d2, op),
      Dist::RecSet(box d1, x, box d2) => self.fold_rec_set(d1, x, d2),
      _ => todo!("{}", d),
    }
  }
}

struct FreeVars {
  fv: HashSet<Var>,
  bv: BoundVars,
}

impl Visitor for FreeVars {
  fn bound_vars(&mut self) -> Option<&mut BoundVars> {
    Some(&mut self.bv)
  }

  fn visit_var(&mut self, x: Var) {
    if !self.bv.is_bound(x) {
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

  fn fold_binder(&mut self, x: Var, d: Dist) -> Dist {
    if x == self.src {
      d.clone()
    } else if self.fv.contains(&x) {
      let xp = x.fresh();
      let d2 = match d {
        Dist::Func(_, d) => Dist::Func(xp, box d.subst(x, Dist::DVar(xp))),
        Dist::Distr(_, d) => Dist::Distr(xp, box d.subst(x, Dist::DVar(xp))),
        Dist::Integral(_, d) => {
          Dist::Integral(xp, box d.subst(x, Dist::DVar(xp)))
        }
        _ => unreachable!(),
      };
      self.super_fold_binder(xp, d2)
    } else {
      self.super_fold_binder(x, d)
    }
  }
}

impl Dist {
  pub fn is_value(&self, bv: &BoundVars) -> bool {
    use Dist::*;
    match self {
      DVar(v) => !bv.is_bound(*v),
      Rat(..) | Func(..) | Distr(..) => true,
      Tuple(ds) => ds.iter().all(|d| d.is_value(bv)),
      Bin(d1, d2, _) | Pred(d1, d2, _) => d1.is_value(bv) && d2.is_value(bv),
      Record(h) => h.values().all(|d| d.is_value(bv)),
      _ => false,
    }
  }

  pub fn free_vars(&self) -> HashSet<Var> {
    let mut pass = FreeVars {
      fv: HashSet::new(),
      bv: BoundVars::default(),
    };
    pass.visit(self);
    pass.fv
  }

  pub fn subst(&self, x: Var, d: Dist) -> Dist {
    let fv = d.free_vars();
    let mut pass = Subst { src: x, dst: d, fv };
    pass.fold(self.clone())
  }

  pub fn bin_many(mut args: Vec<Dist>, op: BinOp) -> Dist {
    assert!(args.len() > 0);
    let first = args.remove(0);
    args
      .into_iter()
      .fold(first, |acc, d| Dist::Bin(box acc, box d, op))
  }

  fn to_doc(&self) -> RcDoc<()> {
    match self {
      Dist::DVar(x) => x.to_doc(),
      Dist::Rat(n) => RcDoc::as_string(format!("{}", n)),
      Dist::Bin(d1, d2, op) => {
        let inner = d1
          .to_doc()
          .group()
          .append(Doc::line())
          .append(RcDoc::as_string(format!("{}", op)))
          .append(Doc::space())
          .append(d2.to_doc().group())
          .group();
        match op {
          BinOp::Add | BinOp::Sub => {
            RcDoc::text("(").append(inner).append(RcDoc::text(")"))
          }
          _ => inner,
        }
      }
      Dist::Pred(d1, d2, op) => RcDoc::text("[")
        .append(d1.to_doc())
        .append(Doc::space())
        .append(RcDoc::as_string(format!("{}", op)))
        .append(Doc::space())
        .append(d2.to_doc())
        .append(RcDoc::text("]")),
      Dist::Distr(x, d) => RcDoc::text("(Λ")
        .append(x.to_doc())
        .append(RcDoc::text("."))
        .append(RcDoc::line().append(d.to_doc()).nest(1).group())
        .append(RcDoc::text(")")),
      Dist::Func(x, d) => RcDoc::text("(λ")
        .append(x.to_doc())
        .append(RcDoc::text("."))
        .append(RcDoc::line().append(d.to_doc()).nest(1).group())
        .append(RcDoc::text(")")),
      Dist::Integral(x, d) => RcDoc::text("(∫d")
        .append(x.to_doc())
        .append(RcDoc::line().append(d.to_doc()).nest(1).group())
        .append(RcDoc::text(")")),
      Dist::Delta(d, x) => RcDoc::text("δ(")
        .append(d.to_doc())
        .append(RcDoc::text(")⟦"))
        .append(x.to_doc())
        .append(RcDoc::text("⟧")),
      Dist::Pdf(d, x) => d
        .to_doc()
        .append(RcDoc::text("⟦"))
        .append(x.to_doc())
        .append(RcDoc::text("⟧")),
      Dist::Record(h) => RcDoc::text("{")
        .append(RcDoc::intersperse(
          h.iter()
            .map(|(k, v)| k.to_doc().append(": ").append(v.to_doc())),
          RcDoc::text(", "),
        ))
        .append(RcDoc::text("}")),
      Dist::Proj(d, x) => {
        d.to_doc().append(RcDoc::text(".")).append(x.to_doc())
      }
      Dist::RecSet(d1, x, d2) => d1
        .to_doc()
        .append(RcDoc::text("["))
        .append(x.to_doc())
        .append(RcDoc::text("↦"))
        .append(d2.to_doc())
        .append(RcDoc::text("]")),
      Dist::App(d1, d2) => {
        d1.to_doc().append(RcDoc::space()).append(d2.to_doc())
      }
      Dist::Lebesgue(x) => RcDoc::text("λ⟦")
        .append(x.to_doc())
        .append(RcDoc::text("⟧")),
      Dist::Tuple(ds) => RcDoc::text("(")
        .append(
          RcDoc::intersperse(ds.iter().map(|d| d.to_doc()), RcDoc::text(", "))
            .group(),
        )
        .append(RcDoc::text(")")),
      _ => todo!("{:?}", self),
    }
  }

  pub fn to_pretty(&self, width: usize) -> String {
    let mut w = Vec::new();
    self.to_doc().render(width, &mut w).unwrap();
    String::from_utf8(w).unwrap()
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
    write!(f, "{}", self.to_pretty(200))
  }
}
