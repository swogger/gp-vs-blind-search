//! The run specification written by the Python wrapper (reach.cgp.write_spec): one `key value...` line per setting.
//! Floats are passed as their IEEE-754 bit patterns (16 hex digits), so every number reaches the engine exactly.
use std::collections::HashMap;

use crate::cnum::C;
use crate::genome::Program;
use crate::ops::Op;

pub struct Spec {
    map: HashMap<String, Vec<Vec<String>>>,
}

pub fn hex_f64(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).unwrap_or_else(|_| panic!("bad float bits {s}")))
}

pub fn f64_hex(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

impl Spec {
    pub fn parse(text: &str) -> Spec {
        let mut map: HashMap<String, Vec<Vec<String>>> = HashMap::new();
        for line in text.lines() {
            let mut it = line.split_whitespace();
            if let Some(k) = it.next() {
                map.entry(k.to_string()).or_default().push(it.map(|s| s.to_string()).collect());
            }
        }
        Spec { map }
    }

    pub fn has(&self, k: &str) -> bool {
        self.map.contains_key(k)
    }

    pub fn all(&self, k: &str) -> Vec<Vec<String>> {
        self.map.get(k).cloned().unwrap_or_default()
    }

    pub fn tokens(&self, k: &str) -> Vec<String> {
        let v = self.map.get(k).unwrap_or_else(|| panic!("spec: missing {k}"));
        assert!(v.len() == 1, "spec: {k} given {} times", v.len());
        v[0].clone()
    }

    pub fn one(&self, k: &str) -> String {
        let t = self.tokens(k);
        assert!(t.len() == 1, "spec: {k} takes one value");
        t[0].clone()
    }

    pub fn u64(&self, k: &str) -> u64 {
        self.one(k).parse().unwrap_or_else(|_| panic!("spec: {k} is not an integer"))
    }

    pub fn f64_bits(&self, k: &str) -> f64 {
        hex_f64(&self.one(k))
    }

    pub fn complex(&self, k: &str) -> Vec<C> {
        let t = self.tokens(k);
        assert!(t.len() % 2 == 0, "spec: {k} needs pairs");
        t.chunks(2).map(|p| C::new(hex_f64(&p[0]), hex_f64(&p[1]))).collect()
    }

    pub fn ops(&self) -> Vec<Op> {
        self.tokens("ops").iter().map(|s| Op::parse(s).unwrap_or_else(|| panic!("unknown operator {s}"))).collect()
    }
}

/// A program as `out node node ...`, each node `op,a` (unary) or `op,a,b`, with compact addresses.
pub fn parse_program(tokens: &[String], n_in: usize) -> Program {
    let out: usize = tokens[0].parse().expect("program output");
    let mut nodes = Vec::new();
    for (k, t) in tokens[1..].iter().enumerate() {
        let parts: Vec<&str> = t.split(',').collect();
        let op = Op::parse(parts[0]).unwrap_or_else(|| panic!("unknown operator {}", parts[0]));
        assert_eq!(parts.len(), 1 + op.arity(), "bad node {t}");
        let a: usize = parts[1].parse().unwrap();
        let b: usize = if op.arity() == 2 { parts[2].parse().unwrap() } else { a };
        assert!(a < n_in + k && b < n_in + k, "node {t} reads a later address");
        nodes.push((op, a, b));
    }
    assert!(out < n_in + nodes.len(), "program output out of range");
    Program { n_in, nodes, out }
}

/// JSON of a program: {"out": k, "nodes": [["op", a], ["op", a, b], ...]}.
pub fn program_json(p: &Program) -> String {
    let nodes: Vec<String> = p.nodes.iter().map(|&(op, a, b)| {
        if op.arity() == 1 { format!("[\"{}\",{}]", op.name(), a) } else { format!("[\"{}\",{},{}]", op.name(), a, b) }
    }).collect();
    format!("{{\"n_in\":{},\"out\":{},\"nodes\":[{}]}}", p.n_in, p.out, nodes.join(","))
}

/// A float as JSON (Python's json reads Infinity and NaN).
pub fn jf(x: f64) -> String {
    if x.is_nan() {
        "NaN".into()
    } else if x.is_infinite() {
        if x > 0.0 { "Infinity".into() } else { "-Infinity".into() }
    } else {
        format!("{x:?}")
    }
}

pub fn jo<T: std::fmt::Display>(x: Option<T>) -> String {
    x.map_or("null".into(), |v| v.to_string())
}

pub fn complex_hex(v: &[C]) -> String {
    v.iter().map(|z| format!("{}{}", f64_hex(z.re), f64_hex(z.im))).collect()
}
