#![feature(box_syntax, box_patterns)]

mod dist;
mod infer;
mod lang;
mod simplify;

use dist::Dist;
use lang::{v, BinOp};
use std::collections::HashMap;

fn main() {
  let e = {
    use lang::Expr::*;
    Bin(
      box Int(2),
      box App(box EVar(v("uniform")), box Tuple(vec![Int(0), Int(3)])),
      BinOp::Add,
    )
  };

  let mut state = HashMap::new();

  #[rustfmt::skip]
  state.insert(v("uniform"), {
    use Dist::*;
    let (t, x) = (v("t"), v("x"));
    let (a, b) = (Proj(box DVar(t), v("0")), Proj(box DVar(t), v("1")));
    Func(t, box Distr(x, box Dist::bin_many(vec![
      Bin(box Rat(1, 1), box Bin(box b, box a, BinOp::Sub), BinOp::Div),
      Lebesgue(x)
    ], BinOp::Mul)))
  });

  let d = Dist::App(box e.infer(), box Dist::Record(state));
  println!("initial: {:?}", d);

  let d2 = d.simplify();
  println!("simplified: {:?}", d2);
}
