//! Single-row CGP genome with unrestricted levels-back (docs/design/cgp.md, "Representation").
//! Genes: node j has [function, connection 1, connection 2] at 3j..3j+2; the last gene is the output.
//! Addresses 0..n_in are the program inputs; node j has address n_in + j and may read any lower address.
use crate::ops::Op;
use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct Shape {
    pub n_in: usize,
    pub n_c: usize,
    pub ops: Vec<Op>,
}

impl Shape {
    pub fn n_genes(&self) -> usize {
        3 * self.n_c + 1
    }

    /// Number of valid values of gene i: functions over the set's operators, connections over the inputs and the
    /// earlier nodes, the output over all inputs and nodes.
    pub fn n_values(&self, i: usize) -> u64 {
        if i == 3 * self.n_c {
            (self.n_in + self.n_c) as u64
        } else if i % 3 == 0 {
            self.ops.len() as u64
        } else {
            (self.n_in + i / 3) as u64
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Genome {
    pub genes: Vec<u32>,
}

/// A program as an active graph with compact addresses: 0..n_in are the inputs, n_in + k is nodes[k].
/// A unary node has b == a (its second argument is ignored).
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub n_in: usize,
    pub nodes: Vec<(Op, usize, usize)>,
    pub out: usize,
}

impl Genome {
    pub fn random(s: &Shape, rng: &mut Rng) -> Genome {
        Genome { genes: (0..s.n_genes()).map(|i| rng.below(s.n_values(i)) as u32).collect() }
    }

    pub fn output(&self) -> usize {
        *self.genes.last().unwrap() as usize
    }

    /// Indices j of the active nodes (reachable from the output), ascending.
    pub fn active(&self, s: &Shape) -> Vec<usize> {
        let mut mark = vec![false; s.n_c];
        let out = self.output();
        if out >= s.n_in {
            mark[out - s.n_in] = true;
        }
        for j in (0..s.n_c).rev() {
            if !mark[j] {
                continue;
            }
            let op = s.ops[self.genes[3 * j] as usize];
            for k in 1..=op.arity() {
                let a = self.genes[3 * j + k] as usize;
                if a >= s.n_in {
                    mark[a - s.n_in] = true;
                }
            }
        }
        (0..s.n_c).filter(|&j| mark[j]).collect()
    }

    /// Used genes: the output gene, and the function gene and the connection genes within the arity of every active
    /// node. The second connection of a unary node is not used (docs/design/cgp.md, "Mixed arity").
    pub fn used(&self, s: &Shape, active: &[usize]) -> Vec<bool> {
        let mut u = vec![false; s.n_genes()];
        u[3 * s.n_c] = true;
        for &j in active {
            let op = s.ops[self.genes[3 * j] as usize];
            u[3 * j] = true;
            for k in 1..=op.arity() {
                u[3 * j + k] = true;
            }
        }
        u
    }

    /// Single active mutation (Goldman and Punch 2013, 2015): mutate uniformly chosen genes, each to a different valid
    /// value, until exactly one used gene has changed; changes to unused genes on the way are kept. A gene with a
    /// single valid value (the function gene of a one-operator set) cannot change and is drawn again.
    pub fn mutate_single(&self, s: &Shape, used: &[bool], rng: &mut Rng) -> Genome {
        let mut child = self.clone();
        loop {
            let i = rng.below(s.n_genes() as u64) as usize;
            let n = s.n_values(i);
            if n < 2 {
                continue;
            }
            let old = child.genes[i] as u64;
            let mut v = rng.below(n - 1);
            if v >= old {
                v += 1;
            }
            child.genes[i] = v as u32;
            if used[i] {
                return child;
            }
        }
    }

    pub fn program(&self, s: &Shape, active: &[usize]) -> Program {
        let mut addr = vec![usize::MAX; s.n_in + s.n_c];
        for i in 0..s.n_in {
            addr[i] = i;
        }
        let mut nodes = Vec::with_capacity(active.len());
        for (k, &j) in active.iter().enumerate() {
            addr[s.n_in + j] = s.n_in + k;
            let op = s.ops[self.genes[3 * j] as usize];
            let a = addr[self.genes[3 * j + 1] as usize];
            let b = if op.arity() == 2 { addr[self.genes[3 * j + 2] as usize] } else { a };
            nodes.push((op, a, b));
        }
        Program { n_in: s.n_in, nodes, out: addr[self.output()] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    fn shape(ops: &[Op], n_c: usize) -> Shape {
        Shape { n_in: 2, n_c, ops: ops.to_vec() }
    }

    #[test]
    fn random_genomes_are_valid_and_programs_feed_forward() {
        let s = shape(&[Op::Add, Op::Exp, Op::Log, Op::Mul], 50);
        let mut r = Rng::from_splitmix(&mut SplitMix64(3));
        for _ in 0..200 {
            let g = Genome::random(&s, &mut r);
            for i in 0..s.n_genes() {
                assert!((g.genes[i] as u64) < s.n_values(i));
            }
            let p = g.program(&s, &g.active(&s));
            for (k, &(_, a, b)) in p.nodes.iter().enumerate() {
                assert!(a < p.n_in + k && b < p.n_in + k);
            }
            assert!(p.out < p.n_in + p.nodes.len());
        }
    }

    #[test]
    fn single_mutation_changes_exactly_one_used_gene() {
        // mixed arity: the idle connection of a unary node is not used, so changing it alone would copy the parent
        let s = shape(&[Op::Add, Op::Exp, Op::Log], 30);
        let mut r = Rng::from_splitmix(&mut SplitMix64(11));
        for _ in 0..2000 {
            let p = Genome::random(&s, &mut r);
            let act = p.active(&s);
            let used = p.used(&s, &act);
            let c = p.mutate_single(&s, &used, &mut r);
            let changed_used = (0..s.n_genes()).filter(|&i| used[i] && c.genes[i] != p.genes[i]).count();
            assert_eq!(changed_used, 1);
        }
    }

    #[test]
    fn single_operator_sets_never_mutate_function_genes() {
        let s = shape(&[Op::Nand], 10);
        let mut r = Rng::from_splitmix(&mut SplitMix64(5));
        let p = Genome::random(&s, &mut r);
        let used = p.used(&s, &p.active(&s));
        for _ in 0..500 {
            let c = p.mutate_single(&s, &used, &mut r);
            assert!(c.genes.iter().all(|&g| g < (s.n_in + s.n_c) as u32));
            assert!((0..s.n_c).all(|j| c.genes[3 * j] == 0));
        }
    }

    #[test]
    fn output_on_an_input_has_no_active_nodes() {
        let s = shape(&[Op::Add], 3);
        let g = Genome { genes: vec![0, 0, 1, 0, 2, 0, 0, 0, 1, 1] };
        assert!(g.active(&s).is_empty());
        assert_eq!(g.program(&s, &[]).out, 1);
        let g = Genome { genes: vec![0, 0, 1, 0, 2, 0, 0, 0, 1, 4] };
        assert_eq!(g.active(&s), vec![2]);
        let g = Genome { genes: vec![0, 0, 1, 0, 2, 0, 0, 0, 1, 3] };
        assert_eq!(g.active(&s), vec![0, 1]);
        assert_eq!(g.program(&s, &[0, 1]), Program { n_in: 2, nodes: vec![(Op::Add, 0, 1), (Op::Add, 2, 0)], out: 3 });
    }
}
