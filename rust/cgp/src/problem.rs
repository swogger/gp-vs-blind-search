//! Fitness, validation and exactness exactly as the Python tree GP (reach.engines.treegp.run_gp, reach.tasks.s2_run):
//! real: nMSE = grammar.mse_against(pred, target, penalty) / var_norm on a fresh 75-point batch; validation nMSE on
//! the 100 validation points; exact = validation nMSE <= tol and equal grid values under the 9-decimal scale-aware
//! rule of grammar.scaled_func_hash. Boolean: Hamming errors on the full truth table; exact = zero errors.
//! There is no size term anywhere (REQ-RULE-01).
use crate::cnum::{self as c, C};
use crate::genome::Program;
use crate::ops::{self, Guards};

#[derive(Clone)]
pub struct Real {
    pub consts: Vec<f64>,
    pub guards: Guards,
    pub penalty: f64,
    pub var_norm: f64,
    pub tol: f64,
    pub scale: f64,
    pub round_factor: f64,
    pub lo: f64,
    pub hi: f64,
    pub batch_n: usize,
    pub target: Program,
    pub val_x: Vec<C>,
    pub val_y: Vec<C>,
    pub grid_x: Vec<C>,
    pub grid_y: Vec<C>,
}

pub struct Bool {
    pub n_vars: usize,
    pub consts: Vec<u64>,
    pub target: u64,
}

pub enum Problem {
    Real(Real),
    Bool(Bool),
}

impl Real {
    /// Program values on the points xs (one variable; the constants are the remaining inputs).
    pub fn eval(&self, p: &Program, xs: &[C], buf: &mut Vec<C>) -> Vec<C> {
        let n = xs.len();
        if p.out == 0 {
            return xs.to_vec();
        }
        if p.out < p.n_in {
            return vec![C::new(self.consts[p.out - 1], 0.0); n];
        }
        buf.clear();
        buf.resize((p.n_in + p.nodes.len()) * n, C::new(0.0, 0.0));
        buf[..n].copy_from_slice(xs);
        for (i, &v) in self.consts.iter().enumerate() {
            for k in 0..n {
                buf[(1 + i) * n + k] = C::new(v, 0.0);
            }
        }
        for (j, &(op, a, b)) in p.nodes.iter().enumerate() {
            let dst = (p.n_in + j) * n;
            for k in 0..n {
                buf[dst + k] = ops::real(op, buf[a * n + k], buf[b * n + k], &self.guards);
            }
        }
        buf[p.out * n..(p.out + 1) * n].to_vec()
    }

    /// grammar.mse_against(pred, target, penalty) / var_norm.
    pub fn nmse(&self, pred: &[C], target: &[C]) -> f64 {
        let v: Vec<f64> = pred.iter().zip(target).map(|(&p, &t)| {
            let d = c::abs(c::sub(p, t));
            if d.is_finite() { d * d } else { self.penalty }
        }).collect();
        c::np_sum(&v) / v.len() as f64 / self.var_norm
    }

    /// Equal grid values under the rule of grammar.scaled_func_hash: v / scale rounded to 9 decimals (real and
    /// imaginary parts), -0 as +0, every non-finite value as NaN, NaN equal to NaN.
    pub fn same_on_grid(&self, pred: &[C]) -> bool {
        let s = C::new(self.scale, 0.0);
        let key = |x: f64| {
            let r = c::np_round(x, self.round_factor) + 0.0;
            if r.is_finite() { Some(r) } else { None }
        };
        pred.iter().zip(&self.grid_y).all(|(&p, &t)| {
            let (a, b) = (c::div(p, s), c::div(t, s));
            key(a.re) == key(b.re) && key(a.im) == key(b.im)
        })
    }

    pub fn val_nmse(&self, p: &Program, buf: &mut Vec<C>) -> f64 {
        let pred = self.eval(p, &self.val_x, buf);
        self.nmse(&pred, &self.val_y)
    }

    pub fn exact(&self, p: &Program, val: f64, buf: &mut Vec<C>) -> bool {
        val <= self.tol && self.same_on_grid(&self.eval(p, &self.grid_x, buf))
    }
}

impl Bool {
    pub fn mask(&self) -> u64 {
        let rows = 1u32 << self.n_vars;
        if rows == 64 { u64::MAX } else { (1u64 << rows) - 1 }
    }

    /// Truth-table column of input i (grammar.var_masks: row r has x_i = bit (n-1-i) of r), then the constants.
    pub fn inputs(&self) -> Vec<u64> {
        let n = self.n_vars;
        let rows = 1usize << n;
        let mut v: Vec<u64> = (0..n).map(|i| {
            (0..rows).filter(|r| (r >> (n - 1 - i)) & 1 == 1).fold(0u64, |acc, r| acc | (1u64 << r))
        }).collect();
        for &k in &self.consts {
            v.push(if k != 0 { self.mask() } else { 0 });
        }
        v
    }

    pub fn eval(&self, p: &Program, inputs: &[u64], buf: &mut Vec<u64>) -> u64 {
        let m = self.mask();
        buf.clear();
        buf.extend_from_slice(inputs);
        for &(op, a, b) in &p.nodes {
            let v = ops::boolean(op, m, buf[a], buf[b]);
            buf.push(v);
        }
        buf[p.out]
    }

    pub fn errors(&self, tt: u64) -> f64 {
        ((tt ^ self.target) & self.mask()).count_ones() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Op;

    #[test]
    fn boolean_inputs_follow_grammar_var_masks() {
        let b = Bool { n_vars: 3, consts: vec![0, 1], target: 0 };
        // var_masks(3): x1 = rows 4-7, x2 = rows 2,3,6,7, x3 = odd rows
        assert_eq!(b.inputs(), vec![0xF0, 0xCC, 0xAA, 0, 0xFF]);
        let x = Bool { n_vars: 6, consts: vec![], target: 0 };
        assert_eq!(x.mask(), u64::MAX);
    }

    #[test]
    fn boolean_eval_and_errors() {
        let b = Bool { n_vars: 2, consts: vec![], target: 0b0110 };
        let p = Program { n_in: 2, nodes: vec![(Op::Xor, 0, 1)], out: 2 };
        let mut buf = Vec::new();
        let tt = b.eval(&p, &b.inputs(), &mut buf);
        assert_eq!(tt, 0b0110);
        assert_eq!(b.errors(tt), 0.0);
        assert_eq!(b.errors(0b1111), 2.0);
    }
}
