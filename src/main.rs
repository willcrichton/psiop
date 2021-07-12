#![feature(box_syntax, box_patterns)]
#![allow(dead_code)]

mod dist;
mod infer;
mod lang;
mod parse;
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

  state.insert(
    v("uniform"),
    Dist::parse("λt. Λx. 1 / (t.1 - t.0) * [t.0 ≤ x] * [x ≤ t.1] * λ⟦x⟧")
      .unwrap(),
  );

  let d = Dist::App(box e.infer(), box Dist::Record(state));
  println!("initial: {}", d);

  let d2 = d.simplify();
  println!("simplified: {}", d2);
}
