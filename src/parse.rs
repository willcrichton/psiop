use crate::dist::Dist;
use crate::lang::{BinOp, Expr, PredOp, Stmt, Var};
use anyhow::{anyhow, bail, Result};
use logos::Logos;
use num::{BigInt, BigRational};
use peg::error::ParseError;

#[rustfmt::skip]
#[derive(Logos, Debug, Clone)]
pub enum Token<'a> {
  #[token("(")] Lparen,
  #[token(")")] Rparen,
  #[token("+")] Add,
  #[token("-")] Sub,
  #[regex("[·*]")] Mul,
  #[token("÷")] Div,
  #[token("/")] RatDiv,
  #[token("⟦")] Ldbl,
  #[token("⟧")] Rdbl,
  #[token("[")] Lsqr,
  #[token("]")] Rsqr,
  #[token("{")] Lbrc,
  #[token("}")] Rbrc,
  #[token(",")] Comma,
  #[token(";")] Semi,
  #[token(":")] Colon,
  #[token(":=")] Assign,
  #[token("=")] Eq,
  #[token("≠")] Neq,
  #[token("≤")] Leq,
  #[token("<")] Le,
  #[token("λ")] Lambda,
  #[token("Λ")] CapLambda,
  #[token("δ")] Delta,
  #[token("observe")] Observe,
  #[token(".")] Dot,
  #[token("∫")] Integral,
  #[token("↦")] Mapsto,
  #[token("if")] If,
  #[token("else")] Else,
  #[token("return")] Return,

  #[regex("[a-zA-ZͰ-Ͽ&&[^λΛδ∫]][a-zA-ZͰ-Ͽ0-9_]*'*", |lex| lex.slice())]
  Ident(&'a str),

  #[regex("[0-9]+", |lex| {
    lex.slice().parse::<isize>().map(|n| BigInt::from(n).into())
  })]
  Number(BigRational),

  #[regex("//[^\n$]*[\n$]", logos::skip, priority = 2)]
  Comment,
  #[regex(r"[ \t\n\f]+", logos::skip, priority = 1)]
  Whitespace,
  #[error]
  Error,
}

impl BinOp {
  fn from_token(token: &Token) -> Self {
    match token {
      Token::Add => BinOp::Add,
      Token::Sub => BinOp::Sub,
      Token::Div | Token::RatDiv => BinOp::Div,
      Token::Mul => BinOp::Mul,
      _ => unreachable!(),
    }
  }
}

impl PredOp {
  fn from_token(token: &Token) -> Self {
    match token {
      Token::Eq => PredOp::Eq,
      Token::Neq => PredOp::Neq,
      Token::Le => PredOp::Le,
      Token::Leq => PredOp::Leq,
      _ => unreachable!(),
    }
  }
}

peg::parser! {grammar psiop_parser<'t>() for [Token<'t>] {
  use Token::*;
  rule var() -> Var = [Ident(v)] { Var::new(v) }
  rule predop() -> PredOp = op:$([Eq] / [Neq] / [Le] / [Leq]) {
    PredOp::from_token(&op[0])
  }
  rule record_entry() -> (Var, Dist) = x:var() [Colon] d:dist()
    { (x, d) }

  pub rule dist() -> Dist = precedence! {
    d1:(@) op:$([Add] / [Sub]) d2:@
      { Dist::Bin(box d1, box d2, BinOp::from_token(&op[0])) }
    --
    d1:(@) op:$([Mul] / [Div]) d2:@
      { Dist::Bin(box d1, box d2, BinOp::from_token(&op[0])) }
    --
    d:(@) [Ldbl] v:var() [Rdbl]
      { Dist::Pdf(box d, v) }
    d1:(@) [Lsqr] x:var() [Mapsto] d2:dist() [Rsqr]
      { Dist::RecSet(box d1, x, box d2)}
    --
    d1:(@) () d2:@
      { Dist::App(box d1, box d2) }
    --
    d:(@) [Dot] v:var()
      { Dist::Proj(box d, v) }
    d:(@) [Dot] [Number(n)]
      { Dist::Proj(box d, Var::new(format!("{}", n))) }
    --
    [Number(n1)] [RatDiv] [Number(n2)]
      { Dist::Rat(n1/n2) }
    [Lbrc] rs:record_entry() ** [Comma] [Rbrc]
      { Dist::Record(rs.into_iter().collect()) }
    [Delta] [Lparen] d:dist() [Rparen] [Ldbl] x:var() [Rdbl]
      { Dist::Delta(box d, x) }
    [Integral] v:var() d:dist()
      {
        let v = v.to_string();
        let mut s = v.chars();
        assert!(s.next().unwrap() == 'd');
        Dist::Integral(Var::new(s.collect::<String>()), box d)
      }
    [Lsqr] d1:dist() op:predop() d2:dist() [Rsqr]
      { Dist::Pred(box d1, box d2, op) }
    [Lambda] v:var() [Dot] d:dist()
      { Dist::Func(v, box d) }
    [CapLambda] v:var() [Dot] d:dist()
      { Dist::Distr(v, box d) }
    [Lambda] [Ldbl] v:var() [Rdbl]
      { Dist::Lebesgue(v) }
    [Number(n)]
      { Dist::Rat(n) }
    v:var()
      { Dist::DVar(v) }
    [Lparen] d:dist() [Rparen]
      { d }
    [Lparen] ds:(dist() **<2,> [Comma]) [Rparen]
      { Dist::Tuple(ds) }
  }

  pub rule expr() -> Expr = precedence! {
    e1:(@) op:predop() e2:@ {
      Expr::Pred(box e1, box e2, op)
    }
    --
    e1:(@) op:$([Add] / [Sub]) e2:@ {
      Expr::Bin(box e1, box e2, BinOp::from_token(&op[0]))
    }
    --
    e1:(@) op:$([Mul] / [RatDiv]) e2:@ {
      Expr::Bin(box e1, box e2, BinOp::from_token(&op[0]))
    }
    --
    e1:(@) () e2:@ { Expr::App(box e1, box e2) }
    --
    [Number(n)] { Expr::Rat(n) }
    v:var() { Expr::EVar(v) }
    [Lparen] [Rparen] { Expr::Tuple(vec![]) }
    [Lparen] es:(expr() **<2,> [Comma]) [Rparen] { Expr::Tuple(es) }
    [Lparen] e:expr() [Rparen] { e }
  }

  pub rule stmt() -> Stmt = precedence! {
    s1:@ [Semi] s2:(@) { Stmt::Seq(box s1, box s2) }
    --
    [If] e:expr() [Lbrc] s1:stmt() [Rbrc] [Else] [Lbrc] s2:stmt() [Rbrc]
      { Stmt::If(e, box s1, box s2) }
    [Observe] [Lparen] e:expr() [Rparen] { Stmt::Observe(e) }
    [Return] e:expr() { Stmt::Return(e) }
    x:var() [Assign] e:expr() { Stmt::Init(x, e) }
  }
}}

pub trait Parse: Sized {
  fn parse_tokens<'a>(tokens: &[Token<'a>]) -> Result<Self, ParseError<usize>>;

  fn parse(s: impl AsRef<str>) -> Result<Self> {
    let s = s.as_ref();
    let sb = s.as_bytes();
    let mut lex = Token::lexer(s);
    let mut spans = Vec::new();
    let mut tokens = Vec::new();

    loop {
      match lex.next() {
        Some(token) => {
          if let Token::Error = token {
            bail!("Bad token at {:?}", lex.span());
          }
          tokens.push(token);
          spans.push(lex.span());
        }
        None => {
          spans.push(lex.span());
          break;
        }
      }
    }

    Self::parse_tokens(&tokens).map_err(|e| {
      let span = &spans[e.location];
      let substr = String::from_utf8_lossy(&sb[span.clone()]);
      let prefix = String::from_utf8_lossy(&sb[..span.start]);
      anyhow!(
        "Failed to parse \"{}\" at {}, expected {:?}",
        substr,
        prefix.chars().count(),
        e.expected.tokens().collect::<Vec<_>>()
      )
    })
  }
}

impl Parse for Expr {
  fn parse_tokens<'a>(tokens: &[Token<'a>]) -> Result<Self, ParseError<usize>> {
    psiop_parser::expr(tokens)
  }
}

impl Parse for Stmt {
  fn parse_tokens<'a>(tokens: &[Token<'a>]) -> Result<Self, ParseError<usize>> {
    psiop_parser::stmt(tokens)
  }
}

impl Parse for Dist {
  fn parse_tokens<'a>(tokens: &[Token<'a>]) -> Result<Self, ParseError<usize>> {
    psiop_parser::dist(tokens)
  }
}

#[macro_export]
macro_rules! dparse {
  ($($e:tt)*) => {
    {
      use crate::parse::Parse;
      let s = format!($($e)*);
      Dist::parse(&s).expect(&s)
    }
  }
}

#[cfg(test)]
mod test {
  use super::Parse;
  use crate::dist::{Dist, Dist::*};

  #[test]
  fn test_dist() {
    macro_rules! is {
      ($t:path) => {
        (box |e: Dist| match e {
          $t(..) => true,
          _ => false,
        }) as Box<dyn Fn(Dist) -> bool>
      };
    }

    let tests = vec![
      (is!(Rat), vec!["0", "1", "11", "1/2"]),
      (is!(DVar), vec!["x", "x'", "α"]),
      (is!(Lebesgue), vec!["λ⟦x⟧"]),
      (is!(Delta), vec!["δ(σ)⟦σ'⟧"]),
      (is!(Bin), vec!["1+2", "1-2", "1*2", "1·2", "1÷2"]),
      (is!(Pred), vec!["[a=b]", "[a≠b]", "[a≤b]", "[a<b]"]),
      (is!(Pdf), vec!["x⟦x⟧", "x.a⟦x⟧"]),
      (is!(Proj), vec!["t.x", "t.0"]),
      (is!(RecSet), vec!["σ[x ↦ 1+2]"]),
      (is!(Record), vec!["{x: 1, y: 2}"]),
      (is!(Integral), vec!["∫dy 1 + 2"]),
      (is!(Distr), vec!["Λy. 1 + 2"]),
      (is!(Func), vec!["λy. 1 + 2"]),
      (is!(App), vec!["f 1", "f (1 + 1)", "f(1, 2)"]),
      (is!(Tuple), vec!["(1, 2)", "(1, 2, 3)"]),
    ];

    for (f, ts) in tests {
      for t in ts {
        let d = Dist::parse(t).expect(t);
        assert!(f(d));
      }
    }
  }
}
