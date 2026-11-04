"""Exact conditional draw of one ancestral form given its neighbours (forward filtering, backward
sampling over the alignment lattice).

Target for node v, concept c:  P_in(y) * prod_k T_k(z_k | y)
  P_in  = T_in(y | x) for an inherited (x = parent form) or borrowed (x = donor form, adaptation
          tables) edge, or the innovation prior G(y) for a replaced edge / the joined root;
  T_k   = the outgoing edge transducers (children inheriting from v, edges borrowing from v),
          at most two handled inside the lattice; any further ones go into the MH correction.

The edge transducer (transducer.py) is a per-segment process: before each input segment one
insertion slot (conditioned on the previous input segment, DUMMY at the start), then substitute or
delete, and one final slot after the last segment. Generating y left to right, the slot that
FOLLOWS a segment s is conditioned on s, so it can be resolved when s is emitted and the lattice
needs no "previous segment" state. State = (i, u, j1, j2, t): i input segments of x consumed, u = 1
if the insertion slot before x[i] has already produced a segment, j_k output segments of z_k
consumed, t = len(y) so far (capped at T). Deletions of x[i] are epsilon moves within a layer t.

Cost ~ (n+1) * 2 * (m1+1) * (m2+1) * (T+1) * N_SEG * 10 ops, a few ms in numba at typical lengths.
"""
import math
import numpy as np
from numba import njit
from .segments import N_SEG, N_OUT, EPS
from .transducer import DUMMY, NOINS, _lae_j
from .events import MIN_LEN, MAX_LEN, LOG_G_LEN

NEG = -1e30
T_MAX = 8            # == mcmc.MAX_FORM_LEN; forms longer than this are outside the sampler's support


@njit(cache=True)
def _log_G_t(t):
    if t < MIN_LEN or t > MAX_LEN:
        return -40.0 - 8.0 * (max(0, t - MAX_LEN) + max(0, MIN_LEN - t))
    return LOG_G_LEN - t * math.log(N_SEG)


@njit(cache=True)
def _prev_x(x, i):
    return x[i - 1] if i > 0 else DUMMY


@njit(cache=True)
def _out_step(ls, li, z, m, j, s, dsub, dins):
    """log weight of out-edge transition from j by (dsub: 1 = s substituted by z[j], 0 = s deleted)
    then (dins: 1 = slot after s inserts the next z segment, 0 = no insertion). NEG if impossible."""
    jj = j
    if dsub == 1:
        if jj >= m:
            return NEG
        w = ls[s, z[jj]]; jj += 1
    else:
        w = ls[s, EPS]
    if dins == 1:
        if jj >= m:
            return NEG
        w += li[s, z[jj]]
    else:
        w += li[s, NOINS]
    return w


@njit(cache=True)
def forward(in_mode, ls_in, li_in, x, n, K, ls1, li1, z1, m1, ls2, li2, z2, m2, T):
    """in_mode: 0 = prior G (x unused), 1 = transducer from x. K in {0,1,2} active out-edges.
    Returns alpha[i, u, j1, j2, t] (log) and the terminal log weights term[i, u, j1, j2, t]."""
    alpha = np.full((n + 1, 2, m1 + 1, m2 + 1, T + 1), NEG)
    # initial slots of the out-edges (conditioned on DUMMY)
    for a1 in range(2):
        if K < 1 and a1 == 1:
            continue
        w1 = 0.0
        if K >= 1:
            if a1 == 1:
                if m1 < 1:
                    continue
                w1 = li1[DUMMY, z1[0]]
            else:
                w1 = li1[DUMMY, NOINS]
        for a2 in range(2):
            if K < 2 and a2 == 1:
                continue
            w2 = 0.0
            if K >= 2:
                if a2 == 1:
                    if m2 < 1:
                        continue
                    w2 = li2[DUMMY, z2[0]]
                else:
                    w2 = li2[DUMMY, NOINS]
            alpha[0, 0, a1, a2, 0] = w1 + w2
    tmp_in = np.full((n + 1, 2), NEG)      # after the T_in step for one (j1, j2) source and symbol
    for t in range(T + 1):
        # epsilon moves (deletions of x[i]) within layer t, increasing i
        if in_mode == 1:
            for i in range(n):
                for u in range(2):
                    w = ls_in[x[i], EPS]
                    if u == 0:
                        w += li_in[_prev_x(x, i), NOINS]
                    for j1 in range(m1 + 1):
                        for j2 in range(m2 + 1):
                            a = alpha[i, u, j1, j2, t]
                            if a > NEG / 2:
                                alpha[i + 1, 0, j1, j2, t] = _lae_j(alpha[i + 1, 0, j1, j2, t], a + w)
        if t == T:
            break
        # emissions to layer t + 1
        for j1 in range(m1 + 1):
            for j2 in range(m2 + 1):
                for s in range(N_SEG):
                    # T_in step: source (i, u) -> target (i', u') with symbol s
                    any_ = False
                    for i in range(n + 1):
                        for u in range(2):
                            tmp_in[i, u] = NEG
                    for i in range(n + 1):
                        for u in range(2):
                            a = alpha[i, u, j1, j2, t]
                            if a <= NEG / 2:
                                continue
                            any_ = True
                            if in_mode == 0:
                                tmp_in[0, 0] = _lae_j(tmp_in[0, 0], a - math.log(N_SEG))
                                continue
                            if u == 0:
                                # insertion from the slot before x[i] (or the final slot)
                                tmp_in[i, 1] = _lae_j(tmp_in[i, 1], a + li_in[_prev_x(x, i), s])
                                if i < n:
                                    tmp_in[i + 1, 0] = _lae_j(tmp_in[i + 1, 0], a + li_in[_prev_x(x, i), NOINS] + ls_in[x[i], s])
                            else:
                                if i < n:
                                    tmp_in[i + 1, 0] = _lae_j(tmp_in[i + 1, 0], a + ls_in[x[i], s])
                    if not any_:
                        continue
                    # out-edge steps
                    for dsub1 in range(2):
                        for dins1 in range(2):
                            if K < 1:
                                if dsub1 == 1 or dins1 == 1:
                                    continue
                                w1 = 0.0; jj1 = j1
                            else:
                                w1 = _out_step(ls1, li1, z1, m1, j1, s, dsub1, dins1)
                                if w1 <= NEG / 2:
                                    continue
                                jj1 = j1 + dsub1 + dins1
                            for dsub2 in range(2):
                                for dins2 in range(2):
                                    if K < 2:
                                        if dsub2 == 1 or dins2 == 1:
                                            continue
                                        w2 = 0.0; jj2 = j2
                                    else:
                                        w2 = _out_step(ls2, li2, z2, m2, j2, s, dsub2, dins2)
                                        if w2 <= NEG / 2:
                                            continue
                                        jj2 = j2 + dsub2 + dins2
                                    w = w1 + w2
                                    for i in range(n + 1):
                                        for u in range(2):
                                            b = tmp_in[i, u]
                                            if b > NEG / 2:
                                                alpha[i, u, jj1, jj2, t + 1] = _lae_j(alpha[i, u, jj1, jj2, t + 1], b + w)
    term = np.full((n + 1, 2, m1 + 1, m2 + 1, T + 1), NEG)
    for u in range(2):
        for t in range(1, T + 1):
            if in_mode == 0:
                if u == 1:
                    continue
                w = _log_G_t(t) + t * math.log(N_SEG)    # the per-segment 1/N_SEG was paid at emission
            else:
                w = li_in[_prev_x(x, n), NOINS] if u == 0 else 0.0
            term[n, u, m1, m2, t] = w
    return alpha, term


@njit(cache=True, fastmath=True)
def forward_lin(in_mode, ps_in, pi_in, x, n, K, ps1, pi1, z1, m1, ps2, pi2, z2, m2, T):
    """Same recursion as `forward`, in linear space with one scale factor per layer t, and with the
    out-edge step written as a small matrix product over (symbol s) x (16 out-edge options), so the
    inner loops vectorise. Tables are probabilities (exp of the log tables). Returns alpha and term in
    LOG space, identical to `forward` up to rounding, so `backward` is unchanged."""
    NC = 16                                    # out-edge options: (dsub1, dins1, dsub2, dins2)
    alpha = np.zeros((n + 1, 2, m1 + 1, m2 + 1, T + 1))
    scale = np.zeros(T + 1)
    # per-edge option weights w_k[j, c, s] (0 = impossible) and targets jj_k[j, c]
    w1 = np.zeros((m1 + 1, 4, N_SEG)); jj1 = np.zeros((m1 + 1, 4), dtype=np.int64)
    w2 = np.zeros((m2 + 1, 4, N_SEG)); jj2 = np.zeros((m2 + 1, 4), dtype=np.int64)
    for j in range(m1 + 1):
        for dsub in range(2):
            for dins in range(2):
                c = 2 * dsub + dins
                jj1[j, c] = j + dsub + dins
                if K < 1:
                    if c == 0:
                        w1[j, c, :] = 1.0
                    continue
                if jj1[j, c] > m1:
                    continue
                for s in range(N_SEG):
                    jx = j
                    if dsub == 1:
                        w = ps1[s, z1[jx]]; jx += 1
                    else:
                        w = ps1[s, EPS]
                    if dins == 1:
                        w *= pi1[s, z1[jx]]
                    else:
                        w *= pi1[s, NOINS]
                    w1[j, c, s] = w
    for j in range(m2 + 1):
        for dsub in range(2):
            for dins in range(2):
                c = 2 * dsub + dins
                jj2[j, c] = j + dsub + dins
                if K < 2:
                    if c == 0:
                        w2[j, c, :] = 1.0
                    continue
                if jj2[j, c] > m2:
                    continue
                for s in range(N_SEG):
                    jx = j
                    if dsub == 1:
                        w = ps2[s, z2[jx]]; jx += 1
                    else:
                        w = ps2[s, EPS]
                    if dins == 1:
                        w *= pi2[s, z2[jx]]
                    else:
                        w *= pi2[s, NOINS]
                    w2[j, c, s] = w
    # combined option matrix per (j1, j2): W[j1, j2, s, c], c = 4*c1 + c2
    W = np.zeros((m1 + 1, m2 + 1, N_SEG, NC))
    for j1 in range(m1 + 1):
        for j2 in range(m2 + 1):
            for c1 in range(4):
                for c2 in range(4):
                    for s in range(N_SEG):
                        W[j1, j2, s, 4 * c1 + c2] = w1[j1, c1, s] * w2[j2, c2, s]
    # initial slots of the out-edges (conditioned on DUMMY)
    for a1 in range(2):
        if K < 1 and a1 == 1:
            continue
        v1 = 1.0
        if K >= 1:
            if a1 == 1:
                if m1 < 1:
                    continue
                v1 = pi1[DUMMY, z1[0]]
            else:
                v1 = pi1[DUMMY, NOINS]
        for a2 in range(2):
            if K < 2 and a2 == 1:
                continue
            v2 = 1.0
            if K >= 2:
                if a2 == 1:
                    if m2 < 1:
                        continue
                    v2 = pi2[DUMMY, z2[0]]
                else:
                    v2 = pi2[DUMMY, NOINS]
            alpha[0, 0, a1, a2, 0] = v1 * v2
    tmp = np.zeros((n + 1, 2, N_SEG))
    out = np.zeros((n + 1, 2, NC))
    inv_nseg = 1.0 / N_SEG
    for t in range(T + 1):
        # epsilon moves (deletions of x[i]) within layer t, increasing i
        if in_mode == 1:
            for i in range(n):
                for u in range(2):
                    w = ps_in[x[i], EPS]
                    if u == 0:
                        w *= pi_in[_prev_x(x, i), NOINS]
                    for j1 in range(m1 + 1):
                        for j2 in range(m2 + 1):
                            a = alpha[i, u, j1, j2, t]
                            if a > 0.0:
                                alpha[i + 1, 0, j1, j2, t] += a * w
        # scale layer t
        mx = 0.0
        for i in range(n + 1):
            for u in range(2):
                for j1 in range(m1 + 1):
                    for j2 in range(m2 + 1):
                        if alpha[i, u, j1, j2, t] > mx:
                            mx = alpha[i, u, j1, j2, t]
        if mx > 0.0:
            inv = 1.0 / mx
            for i in range(n + 1):
                for u in range(2):
                    for j1 in range(m1 + 1):
                        for j2 in range(m2 + 1):
                            alpha[i, u, j1, j2, t] *= inv
            scale[t] = (scale[t - 1] if t > 0 else 0.0) + math.log(mx)
        else:
            scale[t] = scale[t - 1] if t > 0 else 0.0
        if t == T:
            break
        # emissions to layer t + 1
        for j1 in range(m1 + 1):
            for j2 in range(m2 + 1):
                any_ = False
                for i in range(n + 1):
                    for u in range(2):
                        if alpha[i, u, j1, j2, t] > 0.0:
                            any_ = True
                if not any_:
                    continue
                for i in range(n + 1):
                    for u in range(2):
                        for s in range(N_SEG):
                            tmp[i, u, s] = 0.0
                for i in range(n + 1):
                    for u in range(2):
                        a = alpha[i, u, j1, j2, t]
                        if a <= 0.0:
                            continue
                        if in_mode == 0:
                            for s in range(N_SEG):
                                tmp[0, 0, s] += a * inv_nseg
                            continue
                        px = _prev_x(x, i)
                        if u == 0:
                            for s in range(N_SEG):
                                tmp[i, 1, s] += a * pi_in[px, s]
                            if i < n:
                                b = a * pi_in[px, NOINS]
                                for s in range(N_SEG):
                                    tmp[i + 1, 0, s] += b * ps_in[x[i], s]
                        else:
                            if i < n:
                                for s in range(N_SEG):
                                    tmp[i + 1, 0, s] += a * ps_in[x[i], s]
                # out-edge step: out[i, u, c] = sum_s tmp[i, u, s] * W[j1, j2, s, c]; rows of tmp that
                # received nothing are skipped
                for i in range(n + 1):
                    for u in range(2):
                        for c in range(NC):
                            out[i, u, c] = 0.0
                        live = False
                        for s in range(N_SEG):
                            if tmp[i, u, s] != 0.0:
                                live = True
                                break
                        if not live:
                            continue
                        for s in range(N_SEG):
                            v = tmp[i, u, s]
                            if v == 0.0:
                                continue
                            for c in range(NC):
                                out[i, u, c] += v * W[j1, j2, s, c]
                for c1 in range(4):
                    t1 = jj1[j1, c1]
                    if t1 > m1:
                        continue
                    for c2 in range(4):
                        t2 = jj2[j2, c2]
                        if t2 > m2:
                            continue
                        c = 4 * c1 + c2
                        for i in range(n + 1):
                            for u in range(2):
                                alpha[i, u, t1, t2, t + 1] += out[i, u, c]
    # back to log space
    alog = np.full((n + 1, 2, m1 + 1, m2 + 1, T + 1), NEG)
    for t in range(T + 1):
        for i in range(n + 1):
            for u in range(2):
                for j1 in range(m1 + 1):
                    for j2 in range(m2 + 1):
                        a = alpha[i, u, j1, j2, t]
                        if a > 0.0:
                            alog[i, u, j1, j2, t] = math.log(a) + scale[t]
    term = np.full((n + 1, 2, m1 + 1, m2 + 1, T + 1), NEG)
    for u in range(2):
        for t in range(1, T + 1):
            if in_mode == 0:
                if u == 1:
                    continue
                w = _log_G_t(t) + t * math.log(N_SEG)
            else:
                if u == 0:
                    pv = pi_in[_prev_x(x, n), NOINS]
                    w = math.log(pv) if pv > 0.0 else NEG
                else:
                    w = 0.0
            term[n, u, m1, m2, t] = w
    return alog, term


@njit(cache=True)
def backward(alpha, term, in_mode, ls_in, li_in, x, n, K, ls1, li1, z1, m1, ls2, li2, z2, m2, T, unif):
    """Draw one path backwards. unif: array of uniforms (one per decision, >= 2*T+n+2). Returns the
    form (array, length) and the log of the total forward mass (logZ)."""
    # terminal state
    logZ = NEG
    for u in range(2):
        for t in range(1, T + 1):
            v = alpha[n, u, m1, m2, t] + term[n, u, m1, m2, t]
            if v > NEG / 2:
                logZ = _lae_j(logZ, v)
    out = np.empty(T, dtype=np.int64)
    L = 0
    # sample the terminal
    r = unif[0] * math.exp(logZ); acc = 0.0; ti = -1; ui = -1
    for u in range(2):
        for t in range(1, T + 1):
            v = alpha[n, u, m1, m2, t] + term[n, u, m1, m2, t]
            if v > NEG / 2:
                acc += math.exp(v)
                if acc >= r and ti < 0:
                    ti = t; ui = u
    if ti < 0:
        ti = T; ui = 0
    i = n; u = ui; j1 = m1; j2 = m2; t = ti
    kdec = 1
    wbuf = np.empty(4 + N_SEG * 3 * 16)
    pi = np.empty(4 + N_SEG * 3 * 16, dtype=np.int64)
    pu = np.empty_like(pi); pj1 = np.empty_like(pi); pj2 = np.empty_like(pi); pt = np.empty_like(pi); ps = np.empty_like(pi)
    while not (i == 0 and t == 0):
        cnt = 0
        # deletion predecessors (same layer): (i-1, u') -> (i, 0)
        if in_mode == 1 and u == 0 and i >= 1:
            for up in range(2):
                a = alpha[i - 1, up, j1, j2, t]
                if a <= NEG / 2:
                    continue
                w = ls_in[x[i - 1], EPS]
                if up == 0:
                    w += li_in[_prev_x(x, i - 1), NOINS]
                wbuf[cnt] = a + w; pi[cnt] = i - 1; pu[cnt] = up; pj1[cnt] = j1; pj2[cnt] = j2; pt[cnt] = t; ps[cnt] = -1; cnt += 1
        # emission predecessors from layer t-1
        if t >= 1:
            for s in range(N_SEG):
                # out-edge predecessor options
                for dsub1 in range(2):
                    for dins1 in range(2):
                        if K < 1:
                            if dsub1 == 1 or dins1 == 1:
                                continue
                            w1 = 0.0; jp1 = j1
                        else:
                            jp1 = j1 - dsub1 - dins1
                            if jp1 < 0:
                                continue
                            w1 = _out_step(ls1, li1, z1, m1, jp1, s, dsub1, dins1)
                            if w1 <= NEG / 2:
                                continue
                        for dsub2 in range(2):
                            for dins2 in range(2):
                                if K < 2:
                                    if dsub2 == 1 or dins2 == 1:
                                        continue
                                    w2 = 0.0; jp2 = j2
                                else:
                                    jp2 = j2 - dsub2 - dins2
                                    if jp2 < 0:
                                        continue
                                    w2 = _out_step(ls2, li2, z2, m2, jp2, s, dsub2, dins2)
                                    if w2 <= NEG / 2:
                                        continue
                                wo = w1 + w2
                                # T_in predecessor options
                                if in_mode == 0:
                                    a = alpha[0, 0, jp1, jp2, t - 1]
                                    if a > NEG / 2:
                                        wbuf[cnt] = a + wo - math.log(N_SEG); pi[cnt] = 0; pu[cnt] = 0; pj1[cnt] = jp1; pj2[cnt] = jp2; pt[cnt] = t - 1; ps[cnt] = s; cnt += 1
                                    continue
                                if u == 1:
                                    # insertion: pred (i, 0)
                                    a = alpha[i, 0, jp1, jp2, t - 1]
                                    if a > NEG / 2:
                                        wbuf[cnt] = a + wo + li_in[_prev_x(x, i), s]; pi[cnt] = i; pu[cnt] = 0; pj1[cnt] = jp1; pj2[cnt] = jp2; pt[cnt] = t - 1; ps[cnt] = s; cnt += 1
                                else:
                                    if i >= 1:
                                        a = alpha[i - 1, 0, jp1, jp2, t - 1]
                                        if a > NEG / 2:
                                            wbuf[cnt] = a + wo + li_in[_prev_x(x, i - 1), NOINS] + ls_in[x[i - 1], s]; pi[cnt] = i - 1; pu[cnt] = 0; pj1[cnt] = jp1; pj2[cnt] = jp2; pt[cnt] = t - 1; ps[cnt] = s; cnt += 1
                                        a = alpha[i - 1, 1, jp1, jp2, t - 1]
                                        if a > NEG / 2:
                                            wbuf[cnt] = a + wo + ls_in[x[i - 1], s]; pi[cnt] = i - 1; pu[cnt] = 1; pj1[cnt] = jp1; pj2[cnt] = jp2; pt[cnt] = t - 1; ps[cnt] = s; cnt += 1
        if cnt == 0:
            break   # should not happen: state unreachable
        mx = wbuf[0]
        for q in range(1, cnt):
            if wbuf[q] > mx:
                mx = wbuf[q]
        tot = 0.0
        for q in range(cnt):
            tot += math.exp(wbuf[q] - mx)
        r = unif[kdec] * tot; kdec += 1
        acc = 0.0; pick = cnt - 1
        for q in range(cnt):
            acc += math.exp(wbuf[q] - mx)
            if acc >= r:
                pick = q; break
        if ps[pick] >= 0:
            out[L] = ps[pick]; L += 1
        i = pi[pick]; u = pu[pick]; j1 = pj1[pick]; j2 = pj2[pick]; t = pt[pick]
    # reverse
    form = np.empty(L, dtype=np.int64)
    for q in range(L):
        form[q] = out[L - 1 - q]
    return form, logZ


def _arr(y):
    return np.asarray(y, dtype=np.int64) if len(y) else np.zeros(0, dtype=np.int64)


_Z0 = np.zeros(0, dtype=np.int64)
_LS0 = np.zeros((N_SEG, N_OUT)); _LI0 = np.zeros((N_SEG + 1, N_SEG + 1))


_EXP = {}            # id(log table) -> (log table, exp table); the reference keeps the id valid
_EXP_MAX = 8192
_P0 = np.ones((N_SEG, N_OUT)); _PI0 = np.ones((N_SEG + 1, N_SEG + 1))
USE_LIN = True       # linear-space forward (forward_lin); False restores the log-space loops (slow, reference)


def _exp_table(a):
    e = _EXP.get(id(a))
    if e is None or e[0] is not a:
        if len(_EXP) >= _EXP_MAX:
            _EXP.clear()
        e = _EXP[id(a)] = (a, np.exp(a))
    return e[1]


_MEMO = {}            # lattice memo: ~60% of the forward passes in a sweep repeat one computed earlier in it
_MEMO_MAX = 4000


def clear_memo():
    _MEMO.clear()


def _prep(in_mode, in_tables, x, outs, T):
    key = (in_mode, id(in_tables[0]) if in_mode == 1 else 0, id(in_tables[1]) if in_mode == 1 else 0, x,
           tuple((id(ls), id(li), z) for ls, li, z in outs), T)
    hit = _MEMO.get(key)
    if hit is not None:
        held, largs, alpha, term = hit
        if all(a is b for a, b in zip(held, (in_tables if in_mode == 1 else ()) + tuple(t for ls, li, _ in outs for t in (ls, li)))):
            return largs, alpha, term
    largs, alpha, term = _prep_compute(in_mode, in_tables, x, outs, T)
    if len(_MEMO) >= _MEMO_MAX:
        _MEMO.clear()
    _MEMO[key] = ((in_tables if in_mode == 1 else ()) + tuple(t for ls, li, _ in outs for t in (ls, li)), largs, alpha, term)
    return largs, alpha, term


def _prep_compute(in_mode, in_tables, x, outs, T):
    K = len(outs)
    if USE_LIN:
        ps_in, pi_in = (_exp_table(in_tables[0]), _exp_table(in_tables[1])) if in_mode == 1 else (_P0, _PI0)
        xa = _arr(x) if in_mode == 1 else _Z0
        if K >= 1:
            ls1, li1, z1 = outs[0]; ps1, pi1 = _exp_table(ls1), _exp_table(li1)
        else:
            ps1, pi1, z1 = _P0, _PI0, ()
        if K >= 2:
            ls2, li2, z2 = outs[1]; ps2, pi2 = _exp_table(ls2), _exp_table(li2)
        else:
            ps2, pi2, z2 = _P0, _PI0, ()
        z1a, z2a = _arr(z1), _arr(z2)
        args = (in_mode, ps_in, pi_in, xa, len(xa), K, ps1, pi1, z1a, len(z1a), ps2, pi2, z2a, len(z2a), T)
        alpha, term = forward_lin(*args)
        # backward() takes log tables
        largs = (in_mode, in_tables[0] if in_mode == 1 else _LS0, in_tables[1] if in_mode == 1 else _LI0, xa, len(xa), K,
                 outs[0][0] if K >= 1 else _LS0, outs[0][1] if K >= 1 else _LI0, z1a, len(z1a),
                 outs[1][0] if K >= 2 else _LS0, outs[1][1] if K >= 2 else _LI0, z2a, len(z2a), T)
        return largs, alpha, term
    ls_in, li_in = in_tables if in_mode == 1 else (_LS0, _LI0)
    xa = _arr(x) if in_mode == 1 else _Z0
    ls1, li1, z1 = outs[0] if K >= 1 else (_LS0, _LI0, ())
    ls2, li2, z2 = outs[1] if K >= 2 else (_LS0, _LI0, ())
    z1a, z2a = _arr(z1), _arr(z2)
    args = (in_mode, ls_in, li_in, xa, len(xa), K, ls1, li1, z1a, len(z1a), ls2, li2, z2a, len(z2a), T)
    alpha, term = forward(*args)
    return args, alpha, term


def log_Z(in_mode, in_tables, x, outs, T=T_MAX):
    """log of the total mass sum_y P_in(y) * prod_k T_k(z_k | y) over the lattice (len(y) <= T)."""
    args, alpha, term = _prep(in_mode, in_tables, x, outs, T)
    n, m1, m2 = args[4], args[9], args[13]
    v = alpha[n, :, m1, m2, 1:] + term[n, :, m1, m2, 1:]
    v = v[v > NEG / 2]
    if len(v) == 0:
        return NEG
    m = v.max()
    return float(m + np.log(np.exp(v - m).sum()))


def draw(rng, in_mode, in_tables, x, outs, T=T_MAX):
    """in_mode 0 (prior G) or 1 (transducer from x with in_tables = (log_sub, log_ins));
    outs: list of up to two (log_sub, log_ins, z). Returns (form tuple, logZ)."""
    args, alpha, term = _prep(in_mode, in_tables, x, outs, T)
    unif = rng.random(2 * T + args[4] + 4)
    form, logZ = backward(alpha, term, *args, unif)
    return tuple(int(s) for s in form), float(logZ)
