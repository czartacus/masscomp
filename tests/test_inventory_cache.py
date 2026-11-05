"""The compiled kernels must not depend on which inventory populated numba's on-disk cache: numba bakes
module globals into cached code, so a kernel that read N_SEG / EPS / NOINS / DUMMY as globals would run
with the plan inventory's constants under MASSCOMP_INVENTORY=extended if the plan inventory compiled
first (2026-11-05). This test runs the lattice-vs-brute-force check in a subprocess under the other
inventory, after this process has used (and cached) the kernels under its own."""
import os, subprocess, sys, textwrap
from masscomp.segments import N_SEG
from masscomp.transducer import EdgeParams
from masscomp.ffbs import log_Z
from masscomp import events as ev

CHECK = textwrap.dedent("""
    import math, itertools, sys
    import numpy as np
    from masscomp import events as ev
    from masscomp.segments import N_SEG
    from masscomp.transducer import EdgeParams
    from masscomp.ffbs import log_Z
    lam = np.array([4.5, 3.3, 1.7, 5.1])
    e_in = EdgeParams(lam, {}); e1 = EdgeParams(lam * 0.7, {(3, 5): 2.0})
    x = (3, 7, 1); z1 = (5, 7); T = 2
    for in_mode in (0, 1):
        sc = []
        for L in range(1, T + 1):
            for y in itertools.product(range(N_SEG), repeat=L):
                v = (e_in.logprob(x, y) if in_mode == 1 else ev.log_G(y)) + e1.logprob(y, z1)
                sc.append(v)
        m = max(sc); bf = m + math.log(sum(math.exp(v - m) for v in sc))
        lz = log_Z(in_mode, (e_in.log_sub_np, e_in.log_ins_np), x, [(e1.log_sub_np, e1.log_ins_np, z1)], T=T)
        assert abs(lz - bf) < 1e-6, (N_SEG, in_mode, lz, bf)
    print("ok", N_SEG)
""")


def test_other_inventory_in_subprocess():
    lam = ev.LAM_MU
    e = EdgeParams(lam, {})
    log_Z(1, (e.log_sub_np, e.log_ins_np), (1, 2), [(e.log_sub_np, e.log_ins_np, (1, 2))], T=3)   # compile + cache here
    other = "plan" if N_SEG == 40 else "extended"
    env = dict(os.environ, MASSCOMP_INVENTORY=other, PYTHONPATH=os.getcwd())
    r = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert ("ok 40" if other == "extended" else "ok 29") in r.stdout
