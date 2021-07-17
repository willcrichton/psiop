#![feature(box_syntax, box_patterns)]
#![allow(dead_code)]

mod dist;
mod infer;
mod lang;
mod parse;
mod simplify;

pub use lang::{Expr, Stmt};
pub use dist::Dist;
pub use parse::Parse;

use std::collections::HashMap;

#[test]
fn tmp() {
  let s = Stmt::parse(r#"
  p := uniform(0, 1);
if p ≤ 80/100 {
  song := 1;
  p_h := 5/10
} else {
  if p ≤ 95/100 {
    song := 2;
    p_h := 9/10
  } else {
    song := 3;
    p_h := 3/10
  }
};
observe(flip(p_h) = 1);
return song"#).unwrap();
  println!("stmt: {:?}\n", s);

  let d = Dist::App(box s.infer(), box Dist::Record(HashMap::new()));
  println!("initial: {}\n", d);

  let d2 = d.simplify();
  println!("simplified: {}", d2);
}