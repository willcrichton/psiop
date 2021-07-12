use crate::dist::{Dist, PredOp};
use crate::lang::{v, BinOp};

use anyhow::Result;
use lazy_static::lazy_static;
use num::BigInt;
use pest::{
  iterators::{Pair, Pairs},
  prec_climber::{Assoc, Operator, PrecClimber},
  Parser,
};
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "rpsi.pest"]
struct RpsiParser;

lazy_static! {
  static ref PREC_CLIMBER: PrecClimber<Rule> = {
    use Assoc::*;
    use Rule::*;

    PrecClimber::new(vec![
      Operator::new(add, Left) | Operator::new(sub, Left),
      Operator::new(mul, Left) | Operator::new(div, Left),
      Operator::new(dot, Left),
    ])
  };
}

fn parse_pair(pair: Pair<Rule>) -> Dist {
  parse_pairs(Pairs::single(pair))
}

fn parse_pairs(pairs: Pairs<Rule>) -> Dist {
  println!("{:#?}", pairs);
  PREC_CLIMBER.climb(
    pairs,
    |pair: Pair<Rule>| match pair.as_rule() {
      Rule::rat | Rule::number => {
        Dist::Rat(BigInt::from(pair.as_str().parse::<isize>().unwrap()).into())
      }
      Rule::dvar | Rule::var => Dist::DVar(v(pair.as_str())),
      Rule::dist => parse_pairs(pair.into_inner()),
      Rule::tuple => {
        let mut pairs = pair.into_inner();
        let ds = pairs.map(|pair| parse_pair(pair)).collect();
        Dist::Tuple(ds)
      }
      Rule::distr | Rule::func => {
        let rule = pair.as_rule();
        let mut pairs = pair.into_inner();
        let arg = v(pairs.next().unwrap().as_str());
        let body = parse_pair(pairs.next().unwrap());
        match rule {
          Rule::distr => Dist::Distr(arg, box body),
          Rule::func => Dist::Func(arg, box body),
          _ => unreachable!(),
        }
      }
      Rule::lebesgue => {
        let mut pairs = pair.into_inner();
        let arg = v(pairs.next().unwrap().as_str());
        Dist::Lebesgue(arg)
      }
      Rule::pred => {
        let mut pairs = pair.into_inner();
        let lhs = parse_pair(pairs.next().unwrap());
        let op = match pairs.next().unwrap().as_rule() {
          Rule::eq => PredOp::Eq,
          Rule::neq => PredOp::Neq,
          Rule::le => PredOp::Le,
          Rule::leq => PredOp::Leq,
          _ => unreachable!(),
        };
        let rhs = parse_pair(pairs.next().unwrap());
        Dist::Pred(box lhs, box rhs, op)
      }
      rule => todo!("{:?}", rule),
    },
    |lhs: Dist, op: Pair<Rule>, rhs: Dist| {
      let op = match op.as_rule() {
        Rule::add => BinOp::Add,
        Rule::sub => BinOp::Sub,
        Rule::mul => BinOp::Mul,
        Rule::div => BinOp::Div,
        Rule::dot => {
          return Dist::Proj(box lhs, v(format!("{}", rhs)));
        }
        _ => unreachable!(),
      };
      Dist::Bin(box lhs, box rhs, op)
    },
  )
}

impl Dist {
  pub fn parse(s: impl AsRef<str>) -> Result<Self> {
    let pairs = RpsiParser::parse(Rule::dist_full, s.as_ref())?;
    // println!("{:#?}", pairs);
    Ok(parse_pairs(pairs))
  }
}
