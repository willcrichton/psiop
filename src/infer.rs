use crate::dist::{Dist, Dist::*};
use crate::dparse;
use crate::lang::{v, Expr, Stmt, Var};
use lazy_static::lazy_static;
use maplit::hashmap;
use std::collections::HashMap;

fn rec(d: Dist, s: Var, v: Var) -> Box<Dist> {
  box Pdf(box App(box d, box DVar(s)), v)
}

lazy_static! {
  static ref PRELUDE: HashMap<Var, Dist> = {
    hashmap! {
      v("uniform") => dparse!("λt. Λx. 1 / (t.1 - t.0) * [t.0 ≤ x] * [x ≤ t.1] * λ⟦x⟧")
    }
  };
}

impl Expr {
  pub fn infer(&self) -> Dist {
    let state = v("σ");
    let (arg, body) = match self {
      Expr::Rat(n) => (v("n"), dparse!("δ({})⟦n⟧", n)),
      Expr::EVar(x) => {
        let val = match PRELUDE.get(x) {
          Some(d) => format!("{}", d),
          None => format!("σ.{}", x),
        };
        (v("r"), dparse!("δ({})⟦r⟧", val))
      }
      Expr::Bin(e1, e2, _) | Expr::Pred(e1, e2, _) => {
        let inner = match self {
          Expr::Bin(_, _, binop) => dparse!("y {} z", binop),
          Expr::Pred(_, _, predop) => dparse!("[y {} z]", predop),
          _ => unreachable!(),
        };
        (
          v("x"),
          dparse!(
            "∫dy ∫dz ({})(σ)⟦y⟧ * ({})(σ)⟦z⟧ * δ({})⟦x⟧",
            e1.infer(),
            e2.infer(),
            inner
          ),
        )
      }
      Expr::App(box e1, box e2) => (
        v("o"),
        dparse!(
          "∫df ∫da ({})(σ)⟦f⟧ * ({})(σ)⟦a⟧ * (f a)⟦o⟧",
          e1.infer(),
          e2.infer()
        ),
      ),
      Expr::Tuple(es) => {
        let vars = (0..es.len()).map(|i| format!("x{}", i)).collect::<Vec<_>>();
        let base = es
          .iter()
          .zip(vars.iter())
          .map(|(e, x)| format!("({})(σ)⟦{}⟧", e.infer(), x))
          .chain(
            vec![format!(
              "δ(({}))⟦x⟧",
              vars.clone().into_iter().collect::<Vec<_>>().join(",")
            )]
            .into_iter(),
          )
          .collect::<Vec<_>>()
          .join("*");
        (
          v("x"),
          dparse!(
            "{}",
            vars
              .iter()
              .fold(base, |prog, x| format!("∫d{} {}", x, prog))
          ),
        )
      }
      _ => todo!("{:?}", self),
    };

    Func(state, box Distr(arg, box body))
  }
}

impl Stmt {
  pub fn infer(&self) -> Dist {
    use Dist::*;
    let state = v("σ");

    let (arg, body) = match self {
      Stmt::Seq(box s1, box s2) => (
        v("σ2"),
        dparse!("∫dσ1 ({})(σ)⟦σ1⟧ · ({})(σ1)⟦σ2⟧", s1.infer(), s2.infer()),
      ),
      Stmt::Init(x, e) => {
        let xp = v(format!("{}1", x));
        (
          v("σ1"),
          dparse!(
            "∫d{xp} ({d})(σ)⟦{xp}⟧·δ(σ[{x}↦{xp}])⟦σ1⟧",
            x = x,
            xp = xp,
            d = e.infer()
          ),
        )
      }
      Stmt::Observe(e) => {
        let d = e.infer();
        (v("σ1"), dparse!("δ(σ)⟦σ1⟧·(∫dx ({})(σ)⟦x⟧·[x≠0])", d))
      }
      _ => todo!("{:?}", self),
    };

    Func(state, box Distr(arg, box body))
  }
}
