//! (1+4)-ES with neutral drift (docs/design/cgp.md, "Search"): one random parent; each generation four offspring by
//! single active mutation; the best offspring (ties broken uniformly at random) replaces the parent if its fitness is
//! equal or better. Selection compares fitness only: there is no size term (REQ-RULE-01).
//!
//! Evaluation counting (C8): the initial parent costs 1; each generation costs 4 (Boolean) or 5 (real: the parent is
//! re-scored on the generation's fresh batch). A run never stops early except on exact recovery or its CPU limit, so a
//! run to a budget is the prefix of a longer run with the same seed (REQ-ENG-03).
use crate::cnum::C;
use crate::cpu;
use crate::genome::{Genome, Program, Shape};
use crate::problem::Problem;
use crate::rng::{Rng, SplitMix64};

pub const OFFSPRING: usize = 4;

pub struct Settings {
    pub seed: u64,
    pub max_evals: u64,
    pub cpu_limit: f64,
}

#[derive(Debug)]
pub struct Outcome {
    pub terminal_state: &'static str,
    pub claim_exact: bool,
    pub gen_exact: Option<u64>,
    pub evals_exact: Option<u64>,
    pub eval_index_exact: Option<u64>,
    pub gen_tol: Option<u64>,
    pub gens_reached: u64,
    pub evals_reached: u64,
    pub evals_per_gen: u64,
    pub best: Program,
    pub best_val: f64,
    pub parent: Program,
    pub parent_val: f64,
    pub replacements: u64,
    pub neutral_replacements: u64,
    pub cpu_seconds: f64,
}

/// Fitness used for selection: real nMSE on the batch, Boolean Hamming errors.
struct Scorer<'a> {
    prob: &'a Problem,
    rbuf: Vec<C>,
    bbuf: Vec<u64>,
    binputs: Vec<u64>,
}

impl<'a> Scorer<'a> {
    fn new(prob: &'a Problem) -> Scorer<'a> {
        let binputs = match prob {
            Problem::Bool(b) => b.inputs(),
            Problem::Real(_) => Vec::new(),
        };
        Scorer { prob, rbuf: Vec::new(), bbuf: Vec::new(), binputs }
    }

    fn fitness(&mut self, p: &Program, batch: &[C], batch_y: &[C]) -> f64 {
        match self.prob {
            Problem::Real(r) => {
                let pred = r.eval(p, batch, &mut self.rbuf);
                r.nmse(&pred, batch_y)
            }
            Problem::Bool(b) => {
                let tt = b.eval(p, &self.binputs, &mut self.bbuf);
                b.errors(tt)
            }
        }
    }

    /// The monitoring score of tree GP's best-so-far: validation nMSE (real) or errors (Boolean). Not counted.
    fn monitor(&mut self, p: &Program, fit: f64) -> f64 {
        match self.prob {
            Problem::Real(r) => r.val_nmse(p, &mut self.rbuf),
            Problem::Bool(_) => fit,
        }
    }

    fn exact(&mut self, p: &Program, val: f64) -> bool {
        match self.prob {
            Problem::Real(r) => r.exact(p, val, &mut self.rbuf),
            Problem::Bool(_) => val <= 0.0,
        }
    }
}

pub fn run(shape: &Shape, prob: &Problem, st: &Settings) -> Outcome {
    let c0 = cpu::seconds();
    let mut sm = SplitMix64(st.seed);
    let mut rng = Rng::from_splitmix(&mut sm);          // genomes, mutation, tie-breaks
    let mut brng = Rng::from_splitmix(&mut sm);         // batch points (real)
    let is_real = matches!(prob, Problem::Real(_));
    let per_gen = OFFSPRING as u64 + is_real as u64;
    let mut sc = Scorer::new(prob);
    let mut batch: Vec<C> = Vec::new();
    let mut batch_y: Vec<C> = Vec::new();
    let mut tbuf: Vec<C> = Vec::new();
    let mut draw_batch = |brng: &mut Rng, batch: &mut Vec<C>, batch_y: &mut Vec<C>| {
        if let Problem::Real(r) = prob {
            batch.clear();
            for _ in 0..r.batch_n {
                batch.push(C::new(r.lo + (r.hi - r.lo) * brng.uniform(), 0.0));
            }
            *batch_y = r.eval(&r.target, batch, &mut tbuf);
        }
    };

    // generation 0: the initial parent (1 evaluation)
    let mut parent = Genome::random(shape, &mut rng);
    let act = parent.active(shape);
    let mut used = parent.used(shape, &act);
    let mut prog = parent.program(shape, &act);
    draw_batch(&mut brng, &mut batch, &mut batch_y);
    let mut p_fit = sc.fitness(&prog, &batch, &batch_y);
    let mut evals: u64 = 1;
    let mut p_val = sc.monitor(&prog, p_fit);
    let mut best = prog.clone();
    let mut best_val = p_val;
    let mut best_exact: Option<bool> = None;
    let (mut gen_tol, mut gen_exact, mut evals_exact, mut index_exact) = (None, None, None, None);
    let (mut replacements, mut neutral) = (0u64, 0u64);
    let mut state = "budget_exhausted";
    let mut gen: u64 = 0;
    let mut hit_index: u64 = 1;
    let mut children: Vec<(Genome, Vec<usize>, Program, f64)> = Vec::with_capacity(OFFSPRING);
    loop {
        // best-so-far and stopping, as tree GP: tolerance first, then the exact grid check of the best
        if best_val <= sc_tol(prob) {
            if gen_tol.is_none() {
                gen_tol = Some(gen);
            }
            if best_exact.is_none() {
                best_exact = Some(sc.exact(&best, best_val));
            }
            if best_exact == Some(true) {
                state = "converged";
                gen_exact = Some(gen);
                evals_exact = Some(evals);
                index_exact = Some(hit_index);
                break;
            }
        }
        if evals + per_gen > st.max_evals {
            break;
        }
        if gen % 64 == 0 && cpu::seconds() - c0 > st.cpu_limit {
            state = "timeout";
            break;
        }
        gen += 1;
        let start = evals;
        if is_real {
            draw_batch(&mut brng, &mut batch, &mut batch_y);
            p_fit = sc.fitness(&prog, &batch, &batch_y);
        }
        children.clear();
        for _ in 0..OFFSPRING {
            let g = parent.mutate_single(shape, &used, &mut rng);
            let a = g.active(shape);
            let p = g.program(shape, &a);
            let f = sc.fitness(&p, &batch, &batch_y);
            children.push((g, a, p, f));
        }
        evals += per_gen;
        let fmin = children.iter().map(|c| c.3).fold(f64::INFINITY, f64::min);
        let ties: Vec<usize> = (0..OFFSPRING).filter(|&i| children[i].3 == fmin).collect();
        let pick = if ties.is_empty() {
            None                                       // every offspring NaN (cannot happen: the penalty is finite)
        } else if ties.len() == 1 {
            Some(ties[0])
        } else {
            Some(ties[rng.below(ties.len() as u64) as usize])
        };
        if let Some(i) = pick {
            if children[i].3 <= p_fit {
                replacements += 1;
                if children[i].3 == p_fit {
                    neutral += 1;
                }
                let (g, a, p, f) = children.swap_remove(i);
                parent = g;
                used = parent.used(shape, &a);
                prog = p;
                p_fit = f;
                p_val = sc.monitor(&prog, p_fit);
                if p_val < best_val {
                    best = prog.clone();
                    best_val = p_val;
                    best_exact = None;
                    hit_index = start + is_real as u64 + i as u64 + 1;
                }
            }
        }
    }
    Outcome {
        terminal_state: state,
        claim_exact: state == "converged",
        gen_exact,
        evals_exact,
        eval_index_exact: index_exact,
        gen_tol,
        gens_reached: gen,
        evals_reached: evals,
        evals_per_gen: per_gen,
        best,
        best_val,
        parent: prog,
        parent_val: p_val,
        replacements,
        neutral_replacements: neutral,
        cpu_seconds: cpu::seconds() - c0,
    }
}

fn sc_tol(prob: &Problem) -> f64 {
    match prob {
        Problem::Real(r) => r.tol,
        Problem::Bool(_) => 0.0,
    }
}
