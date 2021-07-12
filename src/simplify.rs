use crate::dist::{Dist, Dist::*, Folder, Visitor};
use crate::lang::{v, BinOp, BoundVars, Var};

struct FindConstDelta {
  val: Option<Dist>,
  var: Var,
  int_vars: BoundVars,
}
impl Folder for FindConstDelta {
  fn fold_bin(&mut self, d1: &Dist, d2: &Dist, op: BinOp) -> Dist {
    macro_rules! check {
      ($d:expr, $other:expr) => {
        match $d {
          Delta(box d3, x) => {
            if *x == self.var && d3.is_value(&self.int_vars.bound_vars()) {
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

  fn fold_integral(&mut self, x: Var, d: &Dist) -> Dist {
    self.int_vars.bind(x);
    let dp = self.super_fold_integral(x, d);
    self.int_vars.unbind(x);
    dp
  }
}

#[derive(Default)]
struct DeltaSubst {
  int_vars: BoundVars,
}

impl Folder for DeltaSubst {
  fn fold_integral(&mut self, x: Var, d: &Dist) -> Dist {
    let xp = v(format!("{:?}'", x));
    let mut int_vars = self.int_vars.clone();
    int_vars.bind(xp);
    let mut finder = FindConstDelta {
      int_vars,
      val: None,
      var: xp,
    };
    let dp = d.subst(x, DVar(xp));
    let d_fold = finder.fold(&dp);

    match finder.val {
      Some(d2) => {
        println!("  {:?}\nfolded under {:?} to\n  {:?}\n", dp, xp, d_fold);
        println!(
          "  {:?}\n[{:?} -> {:?}]\n  {:?}\n",
          d_fold,
          xp,
          d2,
          d_fold.subst(xp, d2.clone())
        );
        self.fold(&d_fold.subst(xp, d2))
      }
      None => {
        self.int_vars.bind(x);
        let dp = self.super_fold_integral(x, d);
        self.int_vars.unbind(x);
        dp
      }
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
    _ => todo!("{:?}", f),
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
    _ => todo!("{:?}", f),
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
      Func(x, d) => self.fold(&d.subst(*x, e2.clone())),
      _ => self.super_fold_app(e1, e2),
    }
  }

  fn fold_pdf(&mut self, d: &Dist, x: Var) -> Dist {
    match d {
      Distr(y, d1) => self.fold(&d1.subst(*y, Dist::DVar(x))),
      _ => self.super_fold_pdf(d, x),
    }
  }

  fn fold_proj(&mut self, d: &Dist, x: Var) -> Dist {
    match d {
      Record(h) => h.get(&x).unwrap().clone(),
      Tuple(v) => {
        let i = format!("{:?}", x).parse::<usize>().unwrap();
        v[i].clone()
      }
      _ => self.super_fold_proj(d, x),
    }
  }
}

impl Dist {
  pub fn simplify(&self) -> Dist {
    let run_passes = |init| {
      let passes: Vec<(&'static str, Box<dyn Folder>)> = vec![
        ("partial", box PartialEval),
        ("delta", box DeltaSubst::default()),
        ("linearize", box Linearize),
      ];
      passes.into_iter().fold(init, |d, (name, mut pass)| {
        println!("\n{}: {:?}\n", name, d);
        pass.fold(&d)
      })
    };

    let mut dist = self.clone();
    loop {
      let new_dist = run_passes(dist.clone());
      if new_dist == dist
      /* new_dist.aequiv(&dist) */
      {
        break;
      } else {
        dist = new_dist;
      }
    }
    dist
  }
}
