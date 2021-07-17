#![feature(box_syntax)]

use wasm_bindgen::prelude::*;
use psiop::{Parse, Dist, Stmt};
use std::collections::HashMap;

#[wasm_bindgen]
pub fn init() {
  std::panic::set_hook(box console_error_panic_hook::hook);
}

#[wasm_bindgen]
pub fn get_dist(prog: String) -> Result<String, JsValue> {
  let s = Stmt::parse(prog).map_err(|e| e.to_string())?;
  let d = Dist::App(box s.infer(), box Dist::Record(HashMap::new()));
  let dp = d.simplify();
  return Ok(dp.to_pretty(80));
}
