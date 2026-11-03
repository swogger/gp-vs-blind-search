//! Complex numbers with numpy's complex128 semantics, as numpy 2.0.2 computes them on the study machine (checked bit
//! for bit by tests/test_cgp_semantics.py). Found while building this crate (guardrail row 75):
//! - exp, log, sqrt, sin and cos call the platform maths library (cexp, clog, csqrt, csin, ccos), as numpy does;
//! - multiplication and division use fused multiply-adds where numpy's compiled loops contract them;
//! - the absolute value is numpy's SIMD hypot: larger * sqrt(fma(ratio, ratio, 1));
//! - sums are numpy's pairwise summation (eight accumulators up to 128 terms).

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C {
    pub re: f64,
    pub im: f64,
}

/// np.nan as a complex128 value: nan + 0j.
pub const NAN_C: C = C { re: f64::NAN, im: 0.0 };

#[cfg_attr(target_os = "linux", link(name = "m"))]
extern "C" {
    fn cexp(z: C) -> C;
    fn clog(z: C) -> C;
    fn csqrt(z: C) -> C;
    fn csin(z: C) -> C;
    fn ccos(z: C) -> C;
}

impl C {
    pub fn new(re: f64, im: f64) -> C {
        C { re, im }
    }
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    /// numpy's `z != 0` for complex z (NaN compares unequal).
    pub fn nonzero(self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
}

pub fn add(a: C, b: C) -> C {
    C::new(a.re + b.re, a.im + b.im)
}

pub fn sub(a: C, b: C) -> C {
    C::new(a.re - b.re, a.im - b.im)
}

pub fn mul(a: C, b: C) -> C {
    C::new(a.re.mul_add(b.re, -(a.im * b.im)), a.re.mul_add(b.im, a.im * b.re))
}

/// numpy's complex division (Smith's method, with the multiply-adds contracted).
pub fn div(a: C, b: C) -> C {
    let (ar, ai, br, bi) = (a.re, a.im, b.re, b.im);
    let (abr, abi) = (br.abs(), bi.abs());
    if abr >= abi {
        if abr == 0.0 && abi == 0.0 {
            C::new(ar / abr, ai / abr)
        } else {
            let rat = bi / br;
            let scl = 1.0 / bi.mul_add(rat, br);
            C::new(ai.mul_add(rat, ar) * scl, (-ar).mul_add(rat, ai) * scl)
        }
    } else {
        let rat = br / bi;
        let scl = 1.0 / br.mul_add(rat, bi);
        C::new(ar.mul_add(rat, ai) * scl, ai.mul_add(rat, -ar) * scl)
    }
}

/// numpy's np.abs for complex128 (its SIMD hypot).
pub fn abs(a: C) -> f64 {
    let (mut re, mut im) = (a.re.abs(), a.im.abs());
    let (re_inf, im_inf) = (re == f64::INFINITY, im == f64::INFINITY);
    if re_inf {
        im = f64::INFINITY;
    }
    if im_inf {
        re = f64::INFINITY;
    }
    let (re_nan, im_nan) = (re.is_nan(), im.is_nan());
    if re_nan {
        im = f64::NAN;
    }
    if im_nan {
        re = f64::NAN;
    }
    let larger = re.max(im);
    let smaller = im.min(re);
    let ratio = if larger == 0.0 || smaller == f64::INFINITY { 0.0 } else { smaller / larger };
    ratio.mul_add(ratio, 1.0).sqrt() * larger
}

pub fn exp(a: C) -> C {
    unsafe { cexp(a) }
}
pub fn log(a: C) -> C {
    unsafe { clog(a) }
}
pub fn sqrt(a: C) -> C {
    unsafe { csqrt(a) }
}
pub fn sin(a: C) -> C {
    unsafe { csin(a) }
}
pub fn cos(a: C) -> C {
    unsafe { ccos(a) }
}

/// numpy's pairwise summation of a float64 array (np.sum).
pub fn np_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        np_sum(&a[..n2]) + np_sum(&a[n2..])
    }
}

/// numpy's np.round(x, decimals) for decimals >= 0: rint(x * f) / f with f = 10^decimals.
pub fn np_round(x: f64, factor: f64) -> f64 {
    (x * factor).round_ties_even() / factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn division_by_zero_and_specials() {
        let r = div(C::new(1.0, 0.0), C::new(0.0, 0.0));
        assert!(r.re.is_infinite() && r.im.is_nan());
        let r = div(C::new(6.0, 0.0), C::new(2.0, 0.0));
        assert_eq!((r.re, r.im), (3.0, 0.0));
    }

    #[test]
    fn abs_handles_inf_and_nan_like_numpy() {
        assert_eq!(abs(C::new(f64::NAN, f64::INFINITY)), f64::INFINITY);
        assert!(abs(C::new(f64::NAN, 1.0)).is_nan());
        assert_eq!(abs(C::new(3.0, 4.0)), 5.0);
        assert_eq!(abs(C::new(0.0, 0.0)), 0.0);
    }

    #[test]
    fn pairwise_sum_orders() {
        let a: Vec<f64> = (0..75).map(|i| 1.0 / (i as f64 + 1.0)).collect();
        let naive: f64 = a.iter().sum();
        assert!((np_sum(&a) - naive).abs() < 1e-12);
        assert_eq!(np_sum(&[1.0, 2.0, 3.0]), 6.0);
    }

    #[test]
    fn libm_complex_functions_link() {
        let e = exp(C::new(0.0, 0.0));
        assert_eq!((e.re, e.im), (1.0, 0.0));
        let l = log(C::new(0.0, 0.0));
        assert!(l.re == f64::NEG_INFINITY);
        let s = sqrt(C::new(-4.0, 0.0));
        assert_eq!((s.re, s.im), (0.0, 2.0));
    }
}
