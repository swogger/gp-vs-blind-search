//! xoshiro256** seeded by SplitMix64 (Blackman and Vigna), implemented here so that a run is fully determined by its
//! seed with no external crate. Checked against the published reference outputs (unit tests below).

pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    #[cfg(test)]
    pub fn from_state(s: [u64; 4]) -> Rng {
        Rng { s }
    }

    /// Four SplitMix64 outputs as the state (the seeding the authors recommend).
    pub fn from_splitmix(sm: &mut SplitMix64) -> Rng {
        Rng { s: [sm.next_u64(), sm.next_u64(), sm.next_u64(), sm.next_u64()] }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform on [0, 1) with 53 random bits.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in [0, n), unbiased (Lemire's multiply-shift with rejection). n > 0.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        let mut m = (self.next_u64() as u128) * (n as u128);
        if (m as u64) < n {
            let t = n.wrapping_neg() % n;
            while (m as u64) < t {
                m = (self.next_u64() as u128) * (n as u128);
            }
        }
        (m >> 64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_reference_outputs() {
        // seed 1234567: the reference sequence of Vigna's splitmix64.c
        let mut sm = SplitMix64(1234567);
        let want = [6457827717110365317u64, 3203168211198807973, 9817491932198370423, 4593380528125082431,
                    16408922859458223821];
        for w in want {
            assert_eq!(sm.next_u64(), w);
        }
    }

    #[test]
    fn xoshiro256starstar_reference_outputs() {
        // state {1, 2, 3, 4}: the reference outputs of Blackman and Vigna's xoshiro256starstar.c
        let mut r = Rng::from_state([1, 2, 3, 4]);
        let want = [11520u64, 0, 1509978240, 1215971899390074240, 1216172134540287360, 607988272756665600];
        for w in want {
            assert_eq!(r.next_u64(), w);
        }
    }

    #[test]
    fn below_is_in_range_and_roughly_uniform() {
        let mut r = Rng::from_splitmix(&mut SplitMix64(7));
        let mut c = [0u32; 7];
        for _ in 0..70_000 {
            c[r.below(7) as usize] += 1;
        }
        for k in c {
            assert!((9_400..10_600).contains(&k), "{c:?}");
        }
        for _ in 0..1000 {
            let u = r.uniform();
            assert!((0.0..1.0).contains(&u));
        }
    }
}
