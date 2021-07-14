#![feature(box_syntax, box_patterns)]
#![allow(dead_code)]

mod dist;
mod infer;
mod lang;
mod parse;
mod simplify;

use dist::Dist;
use lang::Stmt;
use parse::Parse;
use std::collections::HashMap;

fn main() {
  let s = Stmt::parse("x := 2 + uniform(0, 3); observe(3 ≤ x)").unwrap();
  println!("stmt: {:?}", s);

  let d = Dist::App(box s.infer(), box Dist::Record(HashMap::new()));
  println!("initial: {}", d);

  let d2 = d.simplify();
  println!("simplified: {}", d2);
}
