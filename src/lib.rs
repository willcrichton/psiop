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

// use dist::Dist;
// use lang::{Expr, Stmt};
// use parse::Parse;
// use std::collections::HashMap;

// fn main() {
//   let s = Stmt::parse(
//     r#"
// has_cancer := Bernoulli(1/1000); 
// if has_cancer {
//   p_test_positive := 9/10
// } else {
//   p_test_positive := 1/10
// };
// test_positive := Bernoulli(p_test_positive)
// "#,
//   )
//   .unwrap();
//   println!("stmt: {:?}", s);
//   // let s = Expr::parse("2 + uniform(0, 3)").unwrap();
//   // println!("expr: {:?}", s);

//   let d = Dist::App(box s.infer(), box Dist::Record(HashMap::new()));
//   println!("initial: {}", d);

//   let d2 = d.simplify();
//   println!("simplified: {}", d2);
// }
