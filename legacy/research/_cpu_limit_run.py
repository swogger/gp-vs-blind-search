#!/usr/bin/env python3
"""
Run a command under a hard CPU-time limit (not wall clock) and report CPU used.

Usage: python3 _cpu_limit_run.py <cpu_seconds> <command> [args...]

RLIMIT_CPU is per-process and inherited across exec, so the child receives
SIGXCPU after <cpu_seconds> of actual CPU time regardless of machine load.
This wrapper is single-threaded, so preexec_fn is safe here (it would not be
in the multi-threaded scheduler that invokes this script).

Emits to stderr:  __CPU_SECONDS__=<float>   __TERMSIG__=<int or 0>
"""
import os, resource, signal, subprocess, sys

limit = int(sys.argv[1])
cmd   = sys.argv[2:]

def _apply_limit():
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit + 5))

proc = subprocess.Popen(cmd, preexec_fn=_apply_limit,
                        stdout=sys.stdout, stderr=subprocess.DEVNULL)
_, status, ru = os.wait4(proc.pid, 0)
cpu = ru.ru_utime + ru.ru_stime
termsig = os.WTERMSIG(status) if os.WIFSIGNALED(status) else 0
sys.stderr.write(f"__CPU_SECONDS__={cpu:.2f}\n__TERMSIG__={termsig}\n")
sys.exit(0 if termsig == 0 else 1)
