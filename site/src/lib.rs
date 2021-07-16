#![feature(box_syntax)]

use wasm_bindgen::prelude::*;
use psiop::{Parse, Dist, Stmt};
use std::collections::HashMap;

/* 
has_cancer := flip(1/1000); 
if has_cancer {
  p_test_positive := 9/10
} else {
  p_test_positive := 1/10
};
test_positive := flip(p_test_positive)
*/

#[wasm_bindgen]
pub fn get_dist(prog: String) -> Result<String, JsValue> {
  let s = Stmt::parse(prog).map_err(|e| e.to_string())?;
  let d = Dist::App(box s.infer(), box Dist::Record(HashMap::new()));
  return Ok(d.simplify().to_pretty(80));
}
