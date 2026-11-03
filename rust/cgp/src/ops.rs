//! Operators of every Study 2 set the CGP runs on, with the semantics of reach.grammar: GUARDED_OPS and the EML-family
//! `lde` (real, complex128, guarded: finite inputs that overflow or underflow give NaN) and BOOL_OPS (Boolean, packed
//! truth tables). Cross-checked against Python by tests/test_cgp_semantics.py.
use crate::cnum::{self as c, C, NAN_C};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Add, Sub, Mul, Div, Exp, Log, Sin, Cos, Sqrt, Square, Eml, Reml, Lde,
    And, Or, Nand, Nor, Xor, Xnor, Not, Imp, Rimp, Nimp, Rnimp,
}

pub const ALL: [Op; 24] = [
    Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Exp, Op::Log, Op::Sin, Op::Cos, Op::Sqrt, Op::Square, Op::Eml,
    Op::Reml, Op::Lde, Op::And, Op::Or, Op::Nand, Op::Nor, Op::Xor, Op::Xnor, Op::Not, Op::Imp, Op::Rimp,
    Op::Nimp, Op::Rnimp,
];

impl Op {
    /// The operator names of reach.grammar.
    pub fn name(self) -> &'static str {
        match self {
            Op::Add => "+", Op::Sub => "-", Op::Mul => "*", Op::Div => "/", Op::Exp => "exp", Op::Log => "log",
            Op::Sin => "sin", Op::Cos => "cos", Op::Sqrt => "sqrt", Op::Square => "square", Op::Eml => "eml",
            Op::Reml => "reml", Op::Lde => "lde", Op::And => "and", Op::Or => "or", Op::Nand => "nand",
            Op::Nor => "nor", Op::Xor => "xor", Op::Xnor => "xnor", Op::Not => "not", Op::Imp => "imp",
            Op::Rimp => "rimp", Op::Nimp => "nimp", Op::Rnimp => "rnimp",
        }
    }

    pub fn parse(name: &str) -> Option<Op> {
        ALL.iter().copied().find(|o| o.name() == name)
    }

    pub fn arity(self) -> usize {
        match self {
            Op::Exp | Op::Log | Op::Sin | Op::Cos | Op::Sqrt | Op::Square | Op::Not => 1,
            _ => 2,
        }
    }

    pub fn is_bool(self) -> bool {
        matches!(self, Op::And | Op::Or | Op::Nand | Op::Nor | Op::Xor | Op::Xnor | Op::Not | Op::Imp | Op::Rimp
                 | Op::Nimp | Op::Rnimp)
    }
}

/// The guard constants of reach.grammar, passed in by the wrapper: EXP_HI, EXP_LO and np.finfo(float).tiny.
#[derive(Clone, Copy, Debug)]
pub struct Guards {
    pub exp_hi: f64,
    pub exp_lo: f64,
    pub tiny: f64,
}

impl Default for Guards {
    fn default() -> Guards {
        Guards { exp_hi: f64::MAX.ln(), exp_lo: f64::MIN_POSITIVE.ln(), tiny: f64::MIN_POSITIVE }
    }
}

/// grammar._guard: NaN where every input is finite and the result is not.
fn guard(r: C, inputs: &[C]) -> C {
    if inputs.iter().all(|x| x.is_finite()) && !r.is_finite() { NAN_C } else { r }
}

/// grammar.fexp: exp, NaN when a finite input overflows or underflows.
pub fn fexp(a: C, g: &Guards) -> C {
    let r = c::exp(a);
    if a.is_finite() && (a.re > g.exp_hi || a.re < g.exp_lo) { NAN_C } else { r }
}

/// grammar.fdiv: IEEE division; NaN when finite inputs give a non-finite or an underflowed result (x/0 stays inf).
pub fn fdiv(a: C, b: C, g: &Guards) -> C {
    let r = c::div(a, b);
    let fin = a.is_finite() && b.is_finite();
    let under = fin && a.nonzero() && c::abs(r) < g.tiny;
    if (fin && b.nonzero() && !r.is_finite()) || under { NAN_C } else { r }
}

/// grammar.fmul.
pub fn fmul(a: C, b: C, g: &Guards) -> C {
    let r = c::mul(a, b);
    let fin = a.is_finite() && b.is_finite();
    let under = fin && a.nonzero() && b.nonzero() && c::abs(r) < g.tiny;
    if (fin && !r.is_finite()) || under { NAN_C } else { r }
}

fn eml(a: C, b: C, g: &Guards) -> C {
    let e = fexp(a, g);
    let l = c::log(b);
    guard(c::sub(e, l), &[e, l])
}

/// One real operator on one point (b is ignored by unary operators).
pub fn real(op: Op, a: C, b: C, g: &Guards) -> C {
    match op {
        Op::Add => guard(c::add(a, b), &[a, b]),
        Op::Sub => guard(c::sub(a, b), &[a, b]),
        Op::Mul => fmul(a, b, g),
        Op::Div => fdiv(a, b, g),
        Op::Exp => fexp(a, g),
        Op::Log => c::log(a),
        Op::Sin => guard(c::sin(a), &[a]),
        Op::Cos => guard(c::cos(a), &[a]),
        Op::Sqrt => guard(c::sqrt(a), &[a]),
        Op::Square => fmul(a, a, g),
        Op::Eml => eml(a, b, g),
        Op::Reml => eml(b, a, g),
        Op::Lde => fdiv(c::log(a), fexp(b, g), g),
        _ => panic!("{} is not a real operator", op.name()),
    }
}

/// One Boolean operator on packed truth tables; m is the all-rows mask.
pub fn boolean(op: Op, m: u64, a: u64, b: u64) -> u64 {
    match op {
        Op::Nand => m ^ (a & b),
        Op::Nor => m ^ (a | b),
        Op::And => a & b,
        Op::Or => a | b,
        Op::Xor => a ^ b,
        Op::Xnor => m ^ a ^ b,
        Op::Not => m ^ a,
        Op::Imp => (m ^ a) | b,
        Op::Rimp => (m ^ b) | a,
        Op::Nimp => a & (m ^ b),
        Op::Rnimp => (m ^ a) & b,
        _ => panic!("{} is not a Boolean operator", op.name()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_and_arities() {
        for o in ALL {
            assert_eq!(Op::parse(o.name()), Some(o));
        }
        assert_eq!(Op::Not.arity(), 1);
        assert_eq!(Op::Square.arity(), 1);
        assert_eq!(Op::Lde.arity(), 2);
    }

    #[test]
    fn guarded_semantics() {
        let g = Guards::default();
        // exp of a finite input above EXP_HI is NaN, not inf (D021)
        assert!(real(Op::Exp, C::new(710.0, 0.0), C::new(0.0, 0.0), &g).re.is_nan());
        // ln 0 = -inf is kept; exp(-inf) = 0 is kept
        assert_eq!(real(Op::Log, C::new(0.0, 0.0), C::new(0.0, 0.0), &g).re, f64::NEG_INFINITY);
        assert_eq!(real(Op::Exp, C::new(f64::NEG_INFINITY, 0.0), C::new(0.0, 0.0), &g).re, 0.0);
        // x/0 stays infinite; finite overflow of a product is NaN
        assert!(real(Op::Div, C::new(1.0, 0.0), C::new(0.0, 0.0), &g).re.is_infinite());
        assert!(real(Op::Mul, C::new(1e200, 0.0), C::new(1e200, 0.0), &g).re.is_nan());
        // eml(x, 1) = e^x
        let e = real(Op::Eml, C::new(1.0, 0.0), C::new(1.0, 0.0), &g);
        assert_eq!(e.re, std::f64::consts::E);
    }

    #[test]
    fn boolean_truth_tables() {
        let m = 0b1111u64;
        let (a, b) = (0b1100u64, 0b1010u64);
        assert_eq!(boolean(Op::Nand, m, a, b), 0b0111);
        assert_eq!(boolean(Op::Nor, m, a, b), 0b0001);
        assert_eq!(boolean(Op::Imp, m, a, b), 0b1011);
        assert_eq!(boolean(Op::Rimp, m, a, b), 0b1101);
        assert_eq!(boolean(Op::Nimp, m, a, b), 0b0100);
        assert_eq!(boolean(Op::Rnimp, m, a, b), 0b0010);
        assert_eq!(boolean(Op::Xnor, m, a, b), 0b1001);
        assert_eq!(boolean(Op::Not, m, a, 0), 0b0011);
    }
}
