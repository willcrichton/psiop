#![feature(box_syntax, box_patterns)]

mod dist;
mod lang;
mod simplify;
mod infer;

use lang::{v, BinOp};
use dist::Dist;

fn main() {
  let e = {
    use lang::Expr::*;
    Bin(
      box Int(2),
      // box Int(3),
      box App(
        box EVar(v("uniform")),
        box Tuple(vec![Int(0), Int(3)]),
      ),
      BinOp::Add,
    )
  };

  let d = Dist::App(box e.infer(), box Dist::empty_state());
  println!("initial: {:?}", d);

  let d2 =  d.simplify();
  println!("simplified: {:?}", d2);
}
