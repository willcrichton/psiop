use crate::dist::{Dist, Dist::*, Folder, Visitor};
use crate::lang::{BinOp, Var};

struct FindConstDelta {
  val: Option<Dist>,
  var: Var,
}
impl Folder for FindConstDelta {
  fn fold_bin(&mut self, ds: &Vec<Dist>, op: BinOp) -> Dist {
    let mut ds2 = vec![];
    for d in ds.iter() {
      match d {
        Dist::Delta(box d2, x) => {
          if *x == self.var {
            self.val = Some(d2.clone());
          } else {
            ds2.push(self.fold(d));
          }
        }
        _ => ds2.push(self.fold(d))
      }
    }

    Dist::Bin(ds2, op)
  }
}

struct DeltaSubst;
impl Folder for DeltaSubst {
  fn fold_integral(&mut self, x: Var, d: &Dist) -> Dist {
    let mut finder = FindConstDelta { val: None, var: x };
    let d_fold = finder.fold(d);

    match finder.val {
      Some(d2) => {
        println!("  {:?}\nfolded under {:?} to\n  {:?}\n", d, x, d_fold);
        println!("  {:?}\n[{:?} -> {:?}]\n  {:?}\n", d_fold, x, d2, d_fold.subst(x, d2.clone()));
        self.fold(&d_fold.subst(x, d2))
      },
      None => self.super_fold_integral(x, d),
    }
  }
}

struct VarSubst;
impl Folder for VarSubst {
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
}

impl Dist {
  pub fn simplify(&self) -> Dist {
    let run_passes = |init| {
      let passes: Vec<Box<dyn Folder>> = vec![box VarSubst, box DeltaSubst];
      passes.into_iter().fold(init, |d, mut pass| pass.fold(&d))
    };

    let mut dist = self.clone();
    loop {
      let new_dist = run_passes(dist.clone());
      println!("\nstep: {:?}\n", new_dist);
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
