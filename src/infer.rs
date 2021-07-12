use crate::dist::Dist;
use crate::lang::{v, BinOp, Expr};
use num::BigInt;

impl Expr {
  pub fn infer(&self) -> Dist {
    use Dist::*;
    let state = v("σ");
    let (arg, body) = match self {
      Expr::Int(n) => {
        let arg = v("r");
        (arg, Delta(box Rat(BigInt::from(*n).into()), arg))
      }
      Expr::EVar(x) => {
        let arg = v("r");
        (arg, Delta(box Proj(box DVar(state), *x), arg))
      }
      Expr::Bin(box e1, box e2, binop) => {
        let (x, y, z) = (v("x"), v("y"), v("z"));
        let (d1, d2) = (e1.infer(), e2.infer());
        (
          x,
          Integral(
            y,
            box Integral(
              z,
              box Dist::bin_many(
                vec![
                  Pdf(box App(box d1, box DVar(state)), y),
                  Pdf(box App(box d2, box DVar(state)), z),
                  Delta(box Bin(box DVar(y), box DVar(z), *binop), x),
                ],
                BinOp::Mul,
              ),
            ),
          ),
        )
      }
      Expr::App(box e1, box e2) => {
        let (output, func, arg) = (v("o"), v("f"), v("a"));
        let (d1, d2) = (e1.infer(), e2.infer());
        (
          output,
          Integral(
            func,
            box Integral(
              arg,
              box Dist::bin_many(
                vec![
                  Pdf(box App(box d1, box DVar(state)), func),
                  Pdf(box App(box d2, box DVar(state)), arg),
                  // TODO: this doesn't work w/ curried functions
                  Pdf(box App(box DVar(func), box DVar(arg)), output),
                ],
                BinOp::Mul,
              ),
            ),
          ),
        )
      }
      Expr::Tuple(es) => {
        let arg = v("x");
        if es.len() == 0 {
          (arg, Delta(box Tuple(vec![]), arg))
        } else {
          let ds = es
            .iter()
            .enumerate()
            .map(|(i, e)| (e.infer(), v(format!("x{}", i))))
            .collect::<Vec<_>>();
          let init = Dist::bin_many(
            ds.clone()
              .into_iter()
              .map(|(d, x)| Pdf(box App(box d, box DVar(state)), x))
              .chain(
                vec![Delta(
                  box Tuple(ds.iter().map(|(_, x)| DVar(*x)).collect()),
                  arg,
                )]
                .into_iter(),
              )
              .collect(),
            BinOp::Mul,
          );
          (
            arg,
            ds.into_iter()
              .fold(init, |acc, (d, x)| Integral(x, box acc)),
          )
        }
      }
      _ => todo!("{:?}", self),
    };

    Func(state, box Distr(arg, box body))
  }
}
