use crate::dist::{Dist, Dist::*, Folder, Visitor};
use crate::dparse;
use crate::lang::{BinOp, BoundVars, PredOp, Var};
use num::{BigRational, One, Zero};
use std::cmp;
use std::collections::HashMap;

enum FindConstDeltaState {
  NotFound,
  Found(Dist),
  FoundTwice,
}

struct FindConstDelta {
  state: FindConstDeltaState,
  var: Var,
  bv: BoundVars,
}
impl Folder for FindConstDelta {
  fn bound_vars(&mut self) -> Option<&mut BoundVars> {
    Some(&mut self.bv)
  }

  fn fold_bin(&mut self, d1: &Dist, d2: &Dist, op: BinOp) -> Dist {
    macro_rules! check {
      ($d:expr, $other:expr) => {
        match $d {
          Delta(box d3, x) => {
            if *x == self.var {
              use FindConstDeltaState::*;
              self.state = match (d3.is_value(&self.bv), &self.state) {
                (false, _) => FoundTwice,
                (true, NotFound) => Found(d3.clone()),
                (true, Found(d3p)) if d3 == d3p => Found(d3.clone()),
                (_, Found(_) | FoundTwice) => FoundTwice,
              };
              return $other.clone();
            }
          }
          _ => {}
        }
      };
    }

    check!(d1, d2);
    check!(d2, d1);

    self.super_fold_bin(d1, d2, op)
  }
}

#[derive(Default)]
struct CollectIntegrals {
  vars: Vec<Var>,
  body: Option<Dist>,
}

impl Visitor for CollectIntegrals {
  fn visit(&mut self, d: &Dist) {
    match d {
      Integral(x, d) => {
        self.vars.push(*x);
        self.visit(d);
      }
      _ => {
        self.body = Some(d.clone());
      }
    }
  }
}

#[derive(Default)]
struct DeltaSubst {
  bv: BoundVars,
}

impl Folder for DeltaSubst {
  fn bound_vars(&mut self) -> Option<&mut BoundVars> {
    Some(&mut self.bv)
  }

  fn fold_integral(&mut self, x: Var, d: &Dist) -> Dist {
    let mut collector = CollectIntegrals::default();
    collector.visit(&Dist::Integral(x, box d.clone()));
    let body = collector.body.unwrap();

    for (i, x) in collector.vars.iter().enumerate() {
      let mut finder = FindConstDelta {
        bv: BoundVars::default(),
        state: FindConstDeltaState::NotFound,
        var: *x,
      };
      let d_fold = finder.fold(&body);

      if let FindConstDeltaState::Found(d2) = finder.state {
        // println!("  {}\nfolded under {} to\n  {}\n", d, x, d_fold);
        // println!(
        //   "  {}\n[{} -> {}]\n  {}\n",
        //   d_fold,
        //   x,
        //   d2,
        //   d_fold.subst(*x, d2.clone())
        // );
        println!("test: {}", d_fold);
        let base = self.fold(&d_fold.subst(*x, d2));
        return collector
          .vars
          .iter()
          .enumerate()
          .filter(|(j, _)| *j != i)
          .fold(base, |d, (_, x)| Integral(*x, box d));
      }
    }

    collector
      .vars
      .into_iter()
      .fold(self.fold(&body), |d, x| Integral(x, box d))
  }
}

struct CollectBinops {
  args: Vec<Dist>,
  op: BinOp,
}

impl Visitor for CollectBinops {
  fn visit(&mut self, d: &Dist) {
    match d {
      Bin(box d1, box d2, op2) if *op2 == self.op => {
        self.visit(d1);
        self.visit(d2);
      }
      _ => {
        self.args.push(d.clone());
      }
    }
  }
}

fn contains_x(f: &Dist, x: Var) -> bool {
  match f {
    DVar(z) => *z == x,
    Rat(..) => false,
    Bin(box d1, box d2, _op) => contains_x(d1, x) || contains_x(d2, x),
    _ => todo!("{}", f),
  }
}

fn linearize(f: &Dist, x: Var, y: Var) -> Dist {
  match f {
    DVar(z) if *z == x => DVar(y),
    DVar(..) | Rat(..) => f.clone(),
    Bin(box d1, box d2, op) => {
      let (with_x, without_x) = if contains_x(d1, x) {
        (d1, d2)
      } else {
        (d2, d1)
      };
      Bin(
        box linearize(with_x, x, y),
        box without_x.clone(),
        op.inverse(),
      )
    }
    _ => todo!("{}", f),
  }
}

struct Linearize;
impl Folder for Linearize {
  fn fold_bin(&mut self, d1: &Dist, d2: &Dist, op: BinOp) -> Dist {
    let mut collector = CollectBinops {
      args: Vec::new(),
      op,
    };
    collector.visit(d1);
    collector.visit(d2);

    let args = collector.args;
    let delta = args
      .iter()
      .enumerate()
      .filter(|(_, arg)| match arg {
        Delta(..) => true,
        _ => false,
      })
      .next();
    let lebesgue = args
      .iter()
      .enumerate()
      .filter(|(_, arg)| match arg {
        Lebesgue(..) => true,
        _ => false,
      })
      .next();
    match (delta, lebesgue) {
      (Some((i, Delta(box f, y))), Some((j, Lebesgue(x)))) => {
        let (x, y) = (*x, *y);
        let f_y = linearize(f, x, y);
        let args = args
          .into_iter()
          .enumerate()
          .filter(|(k, _)| *k != i && *k != j)
          .map(|(_, v)| v)
          .chain(vec![Lebesgue(y), Delta(box f_y, x)].into_iter())
          .collect::<Vec<_>>();
        Dist::bin_many(args, op)
      }
      _ => self.super_fold_bin(d1, d2, op),
    }
  }
}

struct PartialEval;
impl Folder for PartialEval {
  // fn fold_delta(&mut self, d: &Dist, x: Var) -> Dist {
  //   match d {
  //     Dist::Record(h) => {
  //       Dist::bin_many(h.iter().map(|(k, v)| {
  //         Delta(box v.clone(), Proj(box DVar(x), *k))
  //       }).collect(), BinOp::Mul)
  //     }
  //     _ => self.super_fold_delta(d, x)
  //   }
  // }

  fn fold_app(&mut self, e1: &Dist, e2: &Dist) -> Dist {
    match e1 {
      Func(x, d) => {
        println!(
          "  {}\n[{} -> {}]\n  {}\n",
          d,
          x,
          e2,
          d.subst(*x, e2.clone())
        );
        self.fold(&d.subst(*x, e2.clone()))
      }
      _ => self.super_fold_app(e1, e2),
    }
  }

  fn fold_pdf(&mut self, d: &Dist, x: Var) -> Dist {
    match d {
      Distr(y, d1) => {
        println!(
          "  {}\n[{} -> {}]\n  {}\n",
          d1,
          y,
          x,
          d1.subst(*y, Dist::DVar(x))
        );
        self.fold(&d1.subst(*y, Dist::DVar(x)))
      }
      _ => self.super_fold_pdf(d, x),
    }
  }

  fn fold_proj(&mut self, d: &Dist, x: Var) -> Dist {
    match d {
      Record(h) => h.get(&x).unwrap().clone(),
      Tuple(v) => {
        let i = format!("{}", x).parse::<usize>().unwrap();
        v[i].clone()
      }
      _ => self.super_fold_proj(d, x),
    }
  }

  fn fold_rec_set(&mut self, d1: &Dist, x: Var, d2: &Dist) -> Dist {
    match d1 {
      Record(h) => {
        let mut h = h.clone();
        h.insert(x, d2.clone());
        Record(h)
      }
      _ => self.super_fold_rec_set(d1, x, d2),
    }
  }

  // TODO: generalize this to CollectBinops
  // TODO: make Collect* pattern and corresponding filter/partition/etc.
  //   a more structured pattern

  fn fold_bin(&mut self, d1: &Dist, d2: &Dist, op: BinOp) -> Dist {
    (match (d1, d2) {
      (Rat(n1), Rat(n2)) => Some(match op {
        BinOp::Add => Rat(n1 + n2),
        BinOp::Sub => Rat(n1 - n2),
        BinOp::Mul => Rat(n1 * n2),
        BinOp::Div => Rat(n1 / n2),
      }),
      (d, Rat(n)) if n.is_one() && op == BinOp::Div => Some(d.clone()),
      (Rat(n), d) | (d, Rat(n)) => {
        if n.is_zero() {
          match op {
            BinOp::Add => Some(d.clone()),
            BinOp::Mul => Some(Rat(n.clone())),
            _ => None,
          }
        } else if n.is_one() {
          match op {
            BinOp::Mul => Some(d.clone()),
            _ => None,
          }
        } else {
          None
        }
      }
      _ => None,
    })
    .unwrap_or_else(|| self.super_fold_bin(d1, d2, op))
  }

  fn fold_pred(&mut self, d1: &Dist, d2: &Dist, op: PredOp) -> Dist {
    (match (d1, d2) {
      (Rat(n1), Rat(n2)) => {
        let b = match op {
          PredOp::Eq => n1 == n2,
          PredOp::Neq => n1 != n2,
          PredOp::Le => n1 < n2,
          PredOp::Leq => n1 <= n2,
        };
        Some(if b { dparse!("1") } else { dparse!("0") })
      }
      (Rat(n), d) | (d, Rat(n)) if n.is_zero() => match op {
        PredOp::Neq => Some(d.clone()),
        PredOp::Eq => match d {
          Pred(d1p, d2p, op_p) => Some(match op_p {
            PredOp::Leq => Pred(d2p.clone(), d1p.clone(), PredOp::Le),
            PredOp::Le => Pred(d2p.clone(), d1p.clone(), PredOp::Leq),
            PredOp::Eq => Pred(d1p.clone(), d2p.clone(), PredOp::Neq),
            PredOp::Neq => Pred(d1p.clone(), d2p.clone(), PredOp::Eq),
          }),
          _ => None,
        },
        _ => None,
      },
      _ => None,
    })
    .unwrap_or_else(|| self.super_fold_pred(d1, d2, op))
  }

  fn fold_integral(&mut self, x: Var, d: &Dist) -> Dist {
    match d {
      Rat(n) if n.is_zero() => Rat(n.clone()),
      Bin(d1, d2, BinOp::Mul) => {
        let pass = || {
          let mut collector = CollectBinops {
            args: Vec::new(),
            op: BinOp::Mul,
          };
          collector.visit(d1);
          collector.visit(d2);

          let (preds, dists): (Vec<_>, Vec<_>) =
            collector.args.into_iter().partition(|d| match d {
              Pred(..) => true,
              _ => false,
            });

          // println!("preds: {:?}\n dists: {:?}", preds, dists);
          
          if preds.len() != 2 {
            return None;
          }

          let mut range = Range::all();
          for pred in preds.into_iter() {
            match pred {
              Pred(box DVar(xp), box Rat(n), PredOp::Leq) if x == xp => {
                range = range.bound(&n, false);
              }
              Pred(box Rat(n), box DVar(xp), PredOp::Leq) if x == xp => {
                range = range.bound(&n, true);
              }
              _ => { return None; }
            }
          }

          let (lower, upper) = match (range.0, range.1) {
            (Some(l), Some(u)) => (l, u),
            _ => { return None; }
          };

          let vals = dists.into_iter().map(|d| {
            match d {
              Lebesgue(xp) if x == xp =>  {
                Some(Rat(&upper - &lower))
              }
              d if !d.free_vars().contains(&x) => {
                Some(d)
              }
              _ => None
            }
          }).collect::<Option<Vec<_>>>()?;
                  
          Some(Dist::bin_many(vals, BinOp::Mul))
        };

        pass().unwrap_or_else(|| self.super_fold_integral(x, d))
      }
      _ => self.super_fold_integral(x, d),
    }
  }
}

struct Rewrite;
impl Folder for Rewrite {
  fn fold_bin(&mut self, d1: &Dist, d2: &Dist, op: BinOp) -> Dist {
    match (op, (d1, d2)) {
      (BinOp::Mul, (Integral(x, di), d) | (d, Integral(x, di)))
        if !d.free_vars().contains(x) =>
      {
        let dp = Integral(*x, box Bin(di.clone(), box d.clone(), BinOp::Mul));
        println!("  {} {} {}\n->\n  {}\n", d1, op, d2, dp);
        self.fold(&dp)
      }
      (
        BinOp::Mul,
        (Bin(d1p, d2p, BinOp::Add), d) | (d, Bin(d1p, d2p, BinOp::Add)),
      ) => self.fold(&dparse!(
        "({d1p}*{d})+({d2p}*{d})",
        d1p = d1p,
        d2p = d2p,
        d = d
      )),
      _ => self.super_fold_bin(d1, d2, op),
    }
  }

  fn fold_integral(&mut self, x: Var, d: &Dist) -> Dist {
    let mut collector = CollectIntegrals::default();
    collector.visit(&Dist::Integral(x, box d.clone()));
    let body = collector.body.unwrap();

    match body {
      Bin(d1, d2, BinOp::Add) => {
        let (d1fv, d2fv) = (d1.free_vars(), d2.free_vars());
        let fv = &d1fv & &d2fv;
        let in_both_exprs = collector
          .vars
          .iter()
          .enumerate()
          .find(|(_, v)| fv.contains(*v));
        if let Some((i, v)) = in_both_exprs {
          let dp =
            dparse!("(∫d{v} {d1}) + (∫d{v} {d2})", v = v, d1 = d1, d2 = d2);
          collector
            .vars
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .fold(dp, |d, (_, x)| Integral(*x, box d))
        } else {
          self.super_fold_integral(x, d)
        }
      }
      _ => self.super_fold_integral(x, d),
    }
  }
}

#[derive(Debug)]
struct Range(Option<BigRational>, Option<BigRational>);
impl Range {
  pub fn all() -> Self {
    Range(None, None)
  }

  pub fn bound(&self, n: &BigRational, is_lower: bool) -> Self {
    if is_lower {
      Range(
        self
          .0
          .clone()
          .map(|lower| cmp::max(n.clone(), lower))
          .or_else(|| Some(n.clone())),
        self.1.clone(),
      )
    } else {
      Range(
        self.0.clone(),
        self
          .1
          .clone()
          .map(|upper| cmp::min(n.clone(), upper))
          .or_else(|| Some(n.clone())),
      )
    }
  }
}

struct SimplifyGuards;
impl Folder for SimplifyGuards {
  fn fold_bin(&mut self, d1: &Dist, d2: &Dist, op: BinOp) -> Dist {
    let pass = move || {
      if op != BinOp::Mul {
        return None;
      }
      let mut collector = CollectBinops {
        args: Vec::new(),
        op,
      };
      collector.visit(d1);
      collector.visit(d2);

      let args = collector.args;
      let guards = args
        .iter()
        .enumerate()
        .filter_map(|(i, d)| match d {
          Pred(box d1, box d2, op) => Some((i, d1, d2, *op)),
          _ => None,
        })
        .collect::<Vec<_>>();

      let inequalities = guards
        .iter()
        .filter(|(_, _, _, op)| match op {
          PredOp::Leq | PredOp::Le => true,
          _ => false,
        })
        .collect::<Vec<_>>();

      if inequalities.len() == 0 {
        return None;
      }

      let mut ranges = HashMap::new();
      for (_, d1, d2, _) in inequalities.iter() {
        match (d1, d2) {
          (DVar(v), Rat(n)) => {
            let range = ranges.entry(*v).or_insert_with(|| Range::all());
            // println!("{{:?}} lower {} -> {:?}", range, n, range.bound(n, false));
            *range = range.bound(n, false);
          }
          (Rat(n), DVar(v)) => {
            let range = ranges.entry(*v).or_insert_with(|| Range::all());
            *range = range.bound(n, true);
          }
          _ => {
            return None;
          }
        }
      }

      Some(Dist::bin_many(
        ranges
          .into_iter()
          .map(|(v, range)| match (range.0, range.1) {
            (Some(lower), Some(upper)) => {
              dparse!(
                "[{lower} ≤ {v}] * [{v} ≤ {upper}]",
                v = v,
                lower = lower,
                upper = upper
              )
            }
            (Some(lower), None) => dparse!("[{} ≤ {}]", lower, v),
            (None, Some(upper)) => dparse!("[{} ≤ {}]", v, upper),
            (None, None) => unreachable!(),
          })
          .chain(args.iter().enumerate().filter_map(|(i, d)| {
            inequalities
              .iter()
              .find(|(j, _, _, _)| *j == i)
              .is_none()
              .then(|| d.clone())
          }))
          .collect(),
        BinOp::Mul,
      ))
    };

    pass().unwrap_or_else(|| self.super_fold_bin(d1, d2, op))
  }
}

impl Dist {
  pub fn simplify(&self) -> Dist {
    let run_passes = |init| {
      let passes: Vec<(&'static str, Box<dyn Folder>)> = vec![
        ("partial", box PartialEval),
        ("delta", box DeltaSubst::default()),
        ("rewrite", box Rewrite),
        ("guards", box SimplifyGuards), // ("linearize", box Linearize),
      ];
      passes.into_iter().fold(init, |d, (name, mut pass)| {
        let d2 = pass.fold(&d);
        if d2 != d {
          println!("\n{}: {}\n", name, d2);
        }
        d2
      })
    };

    let mut history = vec![];
    let mut dist = self.clone();
    while !history.iter().any(|d2| dist == *d2) {
      history.push(dist.clone());
      dist = run_passes(dist);
    }
    dist
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::dparse;

  fn check_pass(tests: Vec<(&str, &str)>, f: impl Fn(&Dist) -> Dist) {
    for (input, desired_output) in tests {
      let (input, desired_output) =
        (dparse!("{}", input), dparse!("{}", desired_output));
      let actual_output = f(&input);
      assert_eq!(
        actual_output, desired_output,
        "actual: {}, desired: {}",
        actual_output, desired_output
      );
    }
  }

  #[test]
  fn test_delta_subst() {
    let tests = vec![
      ("∫dn n * δ(1)⟦n⟧", "1"),
      ("∫dn (∫dx x * n) * δ(1)⟦n⟧", "∫dx x * 1"),
    ];
    check_pass(tests, |dist| {
      let mut pass = DeltaSubst::default();
      pass.fold(dist)
    });
  }

  #[test]
  fn test_partial_eval() {
    let tests = vec![
      ("1 + 2", "3"),
      ("{x: 1}.x", "1"),
      ("{x: 1}[x ↦ 2]", "{x: 2}"),
      ("(1, 2).0", "1"),
      ("(Λx.x)⟦y⟧", "y"),
      ("(λx.x)1", "1"),
    ];
    check_pass(tests, |dist| {
      let mut pass = PartialEval;
      pass.fold(dist)
    });
  }

  #[test]
  fn test_linearize() {
    let tests = vec![("δ(x)⟦y⟧ * λ⟦x⟧", "λ⟦y⟧ * δ(y)⟦x⟧ ")];
    check_pass(tests, |dist| {
      let mut pass = Linearize;
      pass.fold(dist)
    });
  }

  // #[test]
  // fn test_simplify() {
  //   fn edist(s: &str) -> String {
  //     format!("{}", Expr::parse(s).unwrap().infer())
  //   }

  //   let t1 = edist("2 + uniform(0, 3)");

  //   let tests = vec![(t1.as_str(), "λσ. Λx. 1/3 * [0 ≤ x - 2] * [x - 2 ≤ 3] * λ⟦x⟧")];
  //   check_pass(tests, |dist| dist.simplify());
  // }
}
