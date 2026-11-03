//! Process CPU time (user + system) from clock_gettime(CLOCK_PROCESS_CPUTIME_ID). Every limit is CPU time, never wall
//! clock (REQ-RULE-02).
#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

extern "C" {
    fn clock_gettime(clk: i32, tp: *mut Timespec) -> i32;
}

#[cfg(target_os = "macos")]
const CLOCK_PROCESS_CPUTIME_ID: i32 = 12;
#[cfg(target_os = "linux")]
const CLOCK_PROCESS_CPUTIME_ID: i32 = 2;

pub fn seconds() -> f64 {
    let mut t = Timespec { tv_sec: 0, tv_nsec: 0 };
    let rc = unsafe { clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &mut t) };
    assert_eq!(rc, 0, "clock_gettime failed");
    t.tv_sec as f64 + t.tv_nsec as f64 * 1e-9
}

#[cfg(test)]
mod tests {
    #[test]
    fn cpu_time_advances_with_work() {
        let t0 = super::seconds();
        let mut x = 0u64;
        while super::seconds() - t0 < 0.05 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        }
        assert!(super::seconds() - t0 >= 0.05 && x != 7);
    }
}
