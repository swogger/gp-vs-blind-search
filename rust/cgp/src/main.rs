//! Cartesian GP engine for Study 2 (D072; docs/design/cgp.md). One run per process, driven by reach.tasks.s2_cgp.
//!
//!   cgp --source-hash          print the SHA-256 of the source this binary was built from
//!   cgp run SPEC               one (1+4)-ES run; prints one JSON line
//!   cgp eval SPEC              score given programs, cases and random genomes (Rust-Python cross-check)
//!   cgp ops SPEC               apply every listed real operator to given complex pairs (semantics check)
mod cnum;
mod cpu;
mod es;
mod genome;
mod ops;
mod problem;
mod rng;
mod spec;

use cnum::C;
use genome::{Genome, Program, Shape};
use ops::Guards;
use problem::{Bool, Problem, Real};
use spec::{complex_hex, jf, jo, parse_program, program_json, Spec};

const SOURCE_SHA256: &str = env!("CGP_SOURCE_SHA256");

fn build(sp: &Spec) -> (Shape, Problem) {
    let ops = sp.ops();
    let domain = sp.one("domain");
    let n_vars = sp.u64("n_vars") as usize;
    let consts = sp.tokens("consts");
    let n_in = n_vars + consts.len();
    let n_c = if sp.has("n_nodes") { sp.u64("n_nodes") as usize } else { 0 };
    let shape = Shape { n_in, n_c, ops: ops.clone() };
    let prob = match domain.as_str() {
        "real" => {
            assert_eq!(n_vars, 1, "real sets have one variable");
            assert!(ops.iter().all(|o| !o.is_bool()), "Boolean operator in a real set");
            let g = sp.tokens("guards");
            let target = parse_program(&sp.tokens("target"), n_in);
            Problem::Real(Real {
                consts: consts.iter().map(|s| spec::hex_f64(s)).collect(),
                guards: Guards { exp_hi: spec::hex_f64(&g[0]), exp_lo: spec::hex_f64(&g[1]), tiny: spec::hex_f64(&g[2]) },
                penalty: sp.f64_bits("penalty"),
                var_norm: sp.f64_bits("var_norm"),
                tol: sp.f64_bits("tol"),
                scale: sp.f64_bits("scale"),
                round_factor: sp.f64_bits("round_factor"),
                lo: sp.f64_bits("domain_lo"),
                hi: sp.f64_bits("domain_hi"),
                batch_n: sp.u64("batch_n") as usize,
                target,
                val_x: sp.complex("val_x"),
                val_y: sp.complex("val_y"),
                grid_x: sp.complex("grid_x"),
                grid_y: sp.complex("grid_y"),
            })
        }
        "bool" => {
            assert!(ops.iter().all(|o| o.is_bool()), "real operator in a Boolean set");
            assert!((1..=6).contains(&n_vars), "Boolean sets have 1 to 6 inputs");
            Problem::Bool(Bool {
                n_vars,
                consts: consts.iter().map(|s| s.parse().expect("Boolean constant")).collect(),
                target: u64::from_str_radix(&sp.one("target_tt"), 16).expect("target truth table"),
            })
        }
        d => panic!("unknown domain {d}"),
    };
    (shape, prob)
}

fn cmd_run(sp: &Spec) {
    let (shape, prob) = build(sp);
    assert!(shape.n_c >= 1, "n_nodes must be at least 1");
    let st = es::Settings { seed: sp.u64("seed"), max_evals: sp.u64("max_evals"), cpu_limit: sp.one("cpu_limit").parse().unwrap() };
    let o = es::run(&shape, &prob, &st);
    println!(
        "{{\"source_sha256\":\"{}\",\"terminal_state\":\"{}\",\"claim_exact\":{},\"gen_exact\":{},\"evals_exact\":{},\
\"eval_index_exact\":{},\"gen_tol\":{},\"gens_reached\":{},\"evals_reached\":{},\"evals_per_gen\":{},\"max_evals\":{},\
\"n_nodes\":{},\"seed\":{},\"best_val\":{},\"best\":{},\"parent_val\":{},\"parent\":{},\"replacements\":{},\
\"neutral_replacements\":{},\"cpu_seconds\":{}}}",
        SOURCE_SHA256, o.terminal_state, o.claim_exact, jo(o.gen_exact), jo(o.evals_exact), jo(o.eval_index_exact),
        jo(o.gen_tol), o.gens_reached, o.evals_reached, o.evals_per_gen, st.max_evals, shape.n_c, st.seed,
        jf(o.best_val), program_json(&o.best), jf(o.parent_val), program_json(&o.parent), o.replacements,
        o.neutral_replacements, jf(o.cpu_seconds)
    );
}

/// Scores of one program for the cross-check: everything the run uses to select and to stop.
fn eval_line(p: &Program, prob: &Problem, batch: &[C], batch_y: &[C]) -> String {
    match prob {
        Problem::Real(r) => {
            let mut buf = Vec::new();
            let bv = r.eval(p, batch, &mut buf);
            let fit = r.nmse(&bv, batch_y);
            let val = r.val_nmse(p, &mut buf);
            let gv = r.eval(p, &r.grid_x, &mut buf);
            let exact = r.exact(p, val, &mut buf);
            let tv = r.eval(&r.target, batch, &mut buf);
            format!("{{\"program\":{},\"fitness\":\"{}\",\"val\":\"{}\",\"exact\":{},\"batch\":\"{}\",\"grid\":\"{}\",\"target_batch\":\"{}\"}}",
                    program_json(p), spec::f64_hex(fit), spec::f64_hex(val), exact, complex_hex(&bv), complex_hex(&gv),
                    complex_hex(&tv))
        }
        Problem::Bool(b) => {
            let mut buf = Vec::new();
            let tt = b.eval(p, &b.inputs(), &mut buf);
            format!("{{\"program\":{},\"tt\":\"{:x}\",\"fitness\":{}}}", program_json(p), tt, jf(b.errors(tt)))
        }
    }
}

fn cmd_eval(sp: &Spec) {
    let (_, prob) = build(sp);
    let n_in = sp.u64("n_vars") as usize + sp.tokens("consts").len();
    let (batch, batch_y) = match &prob {
        Problem::Real(_) => (sp.complex("batch_x"), sp.complex("batch_y")),
        Problem::Bool(_) => (Vec::new(), Vec::new()),
    };
    for t in sp.all("program") {
        println!("{}", eval_line(&parse_program(&t, n_in), &prob, &batch, &batch_y));
    }
    for t in sp.all("case") {
        // case SCALE VAR_NORM PENALTY VAL_Y GRID_Y PROGRAM: the program scored against its own target values
        // (computed by Python), so the exact rule is checked on many targets in one call
        if let Problem::Real(r) = &prob {
            let (nv, ng) = (r.val_x.len(), r.grid_x.len());
            let h = |s: &String| spec::hex_f64(s);
            let cx = |s: &[String]| -> Vec<C> { s.chunks(2).map(|q| C::new(h(&q[0]), h(&q[1]))).collect() };
            let mut rc = r.clone();
            rc.scale = h(&t[0]);
            rc.var_norm = h(&t[1]);
            rc.penalty = h(&t[2]);
            rc.val_y = cx(&t[3..3 + 2 * nv]);
            rc.grid_y = cx(&t[3 + 2 * nv..3 + 2 * nv + 2 * ng]);
            let p = parse_program(&t[3 + 2 * nv + 2 * ng..], n_in);
            println!("{}", eval_line(&p, &Problem::Real(rc), &batch, &batch_y));
        } else {
            panic!("case lines are for real sets");
        }
    }
    for t in sp.all("sample") {
        // sample COUNT N_NODES SEED: random genomes, uniform over every gene (the initial distribution of a run)
        let (count, n_c, seed): (usize, usize, u64) = (t[0].parse().unwrap(), t[1].parse().unwrap(), t[2].parse().unwrap());
        let shape = Shape { n_in, n_c, ops: sp.ops() };
        let mut r = rng::Rng::from_splitmix(&mut rng::SplitMix64(seed));
        for _ in 0..count {
            let g = Genome::random(&shape, &mut r);
            let p = g.program(&shape, &g.active(&shape));
            println!("{}", eval_line(&p, &prob, &batch, &batch_y));
        }
    }
}

fn cmd_ops(sp: &Spec) {
    let g = sp.tokens("guards");
    let gd = Guards { exp_hi: spec::hex_f64(&g[0]), exp_lo: spec::hex_f64(&g[1]), tiny: spec::hex_f64(&g[2]) };
    let a = sp.complex("a");
    let b = sp.complex("b");
    for op in sp.ops() {
        let r: Vec<C> = a.iter().zip(&b).map(|(&x, &y)| ops::real(op, x, y, &gd)).collect();
        println!("{{\"op\":\"{}\",\"values\":\"{}\"}}", op.name(), complex_hex(&r));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("--source-hash") => println!("{SOURCE_SHA256}"),
        Some(cmd @ ("run" | "eval" | "ops")) => {
            let path = args.get(2).expect("usage: cgp run|eval|ops SPEC");
            let text = std::fs::read_to_string(path).expect("cannot read the spec file");
            let sp = Spec::parse(&text);
            match cmd {
                "run" => cmd_run(&sp),
                "eval" => cmd_eval(&sp),
                _ => cmd_ops(&sp),
            }
        }
        _ => {
            eprintln!("usage: cgp --source-hash | run SPEC | eval SPEC | ops SPEC");
            std::process::exit(2);
        }
    }
}
