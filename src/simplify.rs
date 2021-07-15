use crate::dist::{Dist, Dist::*, Folder, Visitor};
use crate::lang::{v, BinOp, PredOp, BoundVars, Var};
use num::Zero;

struct FindConstDelta {
  val: Option<Dist>,
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
            if *x == self.var && d3.is_value(&self.bv) {
              self.val = Some(d3.clone());
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
struct DeltaSubst {
  bv: BoundVars,
}

impl Folder for DeltaSubst {
  fn bound_vars(&mut self) -> Option<&mut BoundVars> {
    Some(&mut self.bv)
  }

  fn fold_integral(&mut self, x: Var, d: &Dist) -> Dist {
    let xp = v(format!("{}'", x));
    let dp = d.subst(x, DVar(xp));

    let mut finder = FindConstDelta {
      bv: BoundVars::default(),
      val: None,
      var: xp,
    };
    let d_fold = finder.fold(&dp);

    match finder.val {
      Some(d2) => {
        println!("  {}\nfolded under {} to\n  {}\n", dp, xp, d_fold);
        println!(
          "  {}\n[{} -> {}]\n  {}\n",
          d_fold,
          xp,
          d2,
          d_fold.subst(xp, d2.clone())
        );
        self.fold(&d_fold.subst(xp, d2))
      }
      None => self.super_fold_integral(x, d),
    }
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
  // delta(2 + x)[y] -> delta(y - 2)[x]
  // y = 2 + x
  // y = x - 2 => y + 2 = x
  // y = 2 - x => -(y - 2) = x
  // y = x / 2 => y * 2 = x
  // y = 2 / x => 1 / (y / 2) = x
  //
  // solve for x
  // println!("{:?}({:?})", f, x);
  // todo!()
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

  fn fold_bin(&mut self, d1: &Dist, d2: &Dist, op: BinOp) -> Dist {
    match (d1, d2) {
      (Rat(n1), Rat(n2)) => match op {
        BinOp::Add => Rat(n1 + n2),
        BinOp::Sub => Rat(n1 - n2),
        BinOp::Mul => Rat(n1 * n2),
        BinOp::Div => Rat(n1 / n2),
      },
      _ => self.super_fold_bin(d1, d2, op),
    }
  }

  fn fold_pred(&mut self, d1: &Dist, d2: &Dist, op: PredOp) -> Dist {
    (match (d1, d2) {
      (Rat(n), d) | (d, Rat(n)) if n.is_zero() => match op {
        PredOp::Neq => Some(d.clone()),
        _ => None,
      },
      _ => None,
    })
    .unwrap_or_else(|| self.super_fold_pred(d1, d2, op))
  }
}

struct PushdownMul;
impl Folder for PushdownMul {
  fn fold_bin(&mut self, d1: &Dist, d2: &Dist, op: BinOp) -> Dist {
    match (op, (d1, d2)) {
      (BinOp::Mul, (Integral(x, di), d) | (d, Integral(x, di)))
        if !d.free_vars().contains(x) =>
      {
        let dp = Integral(*x, box Bin(di.clone(), box d.clone(), BinOp::Mul));
        println!("  {} {} {}\n->\n  {}\n", d1, op, d2, dp);
        self.fold(&dp)
      }
      _ => self.super_fold_bin(d1, d2, op),
    }
  }
}

impl Dist {
  pub fn simplify(&self) -> Dist {
    let run_passes = |init| {
      let passes: Vec<(&'static str, Box<dyn Folder>)> = vec![
        ("partial", box PartialEval),
        ("delta", box DeltaSubst::default()),
        ("pushdown", box PushdownMul),
        // ("linearize", box Linearize),
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
  use crate::lang::Expr;
  use crate::parse::Parse;

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
