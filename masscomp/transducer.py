"""Edge transducer (plan §3.1–3.2).

Generative process for one edge, input form y = (y_1..y_n):
  for i = 1..n:
      insertion slot before y_i (conditioned on previous INPUT segment, dummy at start):
          no insertion, or insert exactly one segment s''
      then y_i is substituted by s' in Σ or deleted (EPS)
  final insertion slot after y_n.
Each slot and each substitution is a proper softmax distribution, so the output
distribution is normalised. Π_e(y'|y) sums over all alignments by edit-DP.
The DP is written in plain Python on float lists: ~10x faster than numpy scalars here.
"""
import math
import numpy as np
from numba import njit
from .segments import N_SEG, N_OUT, EPS, DIST

FLOOR, FLOOR_PER_SEG = -40.0, 8.0
DUMMY = N_SEG          # "previous segment" index at the start of the form
NOINS = N_SEG          # "no insertion" index in the insertion table
NEG = -1e30


def _lae(a, b):
    if a < b:
        a, b = b, a
    if b <= NEG / 2:
        return a
    return a + math.log1p(math.exp(b - a))


@njit(cache=True, fastmath=False)
def _lae_j(a, b):
    if a < b:
        a, b = b, a
    if b <= -5e29:
        return a
    return a + math.log1p(math.exp(b - a))


@njit(cache=True, fastmath=False)
def _dp(ls, li, y1, y2):
    """Compiled edit-DP. ls: (N_SEG, N_OUT) log-sub table; li: (N_SEG+1, N_SEG+1) log-ins table."""
    n = y1.shape[0]; m = y2.shape[0]
    cur = np.full(m + 1, NEG)
    cur[0] = 0.0
    nxt = np.full(m + 1, NEG)
    for i in range(n):
        prev = y1[i - 1] if i > 0 else DUMMY
        s = y1[i]
        noins = li[prev, NOINS]
        ldel = ls[s, EPS]
        for j in range(m + 1):
            nxt[j] = NEG
        for j in range(m + 1):
            f = cur[j]
            if f <= -5e29:
                continue
            nxt[j] = _lae_j(nxt[j], f + noins + ldel)
            if j < m:
                o = y2[j]
                v = _lae_j(f + noins + ls[s, o], f + li[prev, o] + ldel)
                nxt[j + 1] = _lae_j(nxt[j + 1], v)
                if j + 1 < m:
                    nxt[j + 2] = _lae_j(nxt[j + 2], f + li[prev, o] + ls[s, y2[j + 1]])
        tmp = cur; cur = nxt; nxt = tmp
    prev = y1[n - 1] if n > 0 else DUMMY
    out = cur[m] + li[prev, NOINS]
    if m > 0:
        out = _lae_j(out, cur[m - 1] + li[prev, y2[m - 1]])
    return out


_ARR = {}


def _arr(form):
    """Tuple form -> int64 array, memoised (forms repeat constantly)."""
    a = _ARR.get(form)
    if a is None:
        a = np.array(form, dtype=np.int64)
        if len(_ARR) < 500000:
            _ARR[form] = a
    return a


class EdgeParams:
    """Dense log-probability tables for one edge, built from (lam, u_e)."""

    def __init__(self, lam, u=None):
        lam0, lam1, lam2, lam3 = lam
        u = u or {}
        psi = np.zeros((N_SEG, N_OUT))
        psi[:, :N_SEG] += lam0 * np.eye(N_SEG) - lam1 * DIST
        psi[:, EPS] += -lam2
        ins = np.zeros((N_SEG + 1, N_SEG + 1))
        ins[:N_SEG, :N_SEG] += -lam3 - lam1 * DIST
        ins[DUMMY, :N_SEG] += -lam3
        ins[:, NOINS] = 0.0
        for key, w in u.items():
            if key[0] == "ins":
                ins[:, key[1]] += w
            else:
                psi[key[0], key[1]] += w
        self.log_sub_np = psi - np.logaddexp.reduce(psi, axis=1, keepdims=True)
        self.log_ins_np = ins - np.logaddexp.reduce(ins, axis=1, keepdims=True)
        self.log_sub = self.log_sub_np.tolist()
        self.log_ins = self.log_ins_np.tolist()
        self._p_sub = np.exp(self.log_sub_np)
        self._p_ins = np.exp(self.log_ins_np)
        self._c_sub = np.cumsum(self._p_sub, axis=1)
        self._c_ins = np.cumsum(self._p_ins, axis=1)
        self._cache = {}

    def logprob(self, y1, y2):
        key = (y1, y2)
        v = self._cache.get(key)
        if v is None:
            v = self._logprob(y1, y2)
            if len(self._cache) < 200000:
                self._cache[key] = v
        return v

    def _logprob(self, y1, y2):
        v = _dp(self.log_sub_np, self.log_ins_np, _arr(y1), _arr(y2))
        if v <= -5e29:
            # The one-insertion-per-slot transducer gives exactly zero probability to an output
            # more than len(y1)+1 segments longer than its input. A hard zero inside the prior's
            # support breaks stepping-stone estimation (and MH deltas), so impossible outputs get
            # a soft floor that stands in for the multi-insertion tail the real process would have.
            # The tables then sum to slightly more than 1 (by ~e^-40 per excess output); negligible.
            return FLOOR - FLOOR_PER_SEG * max(0, len(y2) - 2 * len(y1) - 1)
        return v

    def _logprob_py(self, y1, y2):
        n, m = len(y1), len(y2)
        ls, li = self.log_sub, self.log_ins
        cur = [NEG] * (m + 1)
        cur[0] = 0.0
        for i in range(n):
            prev = y1[i - 1] if i > 0 else DUMMY
            s = y1[i]
            lis = li[prev]
            noins = lis[NOINS]
            ldel = ls[s][EPS]
            lss = ls[s]
            nxt = [NEG] * (m + 1)
            for j in range(m + 1):
                f = cur[j]
                if f <= NEG / 2:
                    continue
                nxt[j] = _lae(nxt[j], f + noins + ldel)
                if j < m:
                    o = y2[j]
                    v = _lae(f + noins + lss[o], f + lis[o] + ldel)
                    nxt[j + 1] = _lae(nxt[j + 1], v)
                    if j + 1 < m:
                        nxt[j + 2] = _lae(nxt[j + 2], f + lis[o] + lss[y2[j + 1]])
            cur = nxt
        prev = y1[n - 1] if n > 0 else DUMMY
        out = cur[m] + li[prev][NOINS]
        if m > 0:
            out = _lae(out, cur[m - 1] + li[prev][y2[m - 1]])
        return out

    def sample(self, y1, rng):
        """Inverse-CDF sampling from precomputed cumulative tables (rng.choice(p=...) is ~100x slower)."""
        n = len(y1)
        us = rng.random(2 * n + 1)
        out = []
        k_u = 0
        for i in range(n + 1):
            prev = y1[i - 1] if i > 0 else DUMMY
            k = int(np.searchsorted(self._c_ins[prev], us[k_u])); k_u += 1
            if k < NOINS:
                out.append(k)
            if i < n:
                o = int(np.searchsorted(self._c_sub[y1[i]], us[k_u])); k_u += 1
                if o < EPS:
                    out.append(o)
        return tuple(out)


def brute_force_logprob(ep, y1, y2):
    """Sum over alignments by explicit recursion (for tests only)."""
    ls, li = ep.log_sub, ep.log_ins

    def rec(i, j):
        n, m = len(y1), len(y2)
        prev = y1[i - 1] if i > 0 else DUMMY
        if i == n:
            if j == m:
                return li[prev][NOINS]
            if j == m - 1:
                return li[prev][y2[j]]
            return NEG
        total = NEG
        for ins_k in ([NOINS] + ([y2[j]] if j < m else [])):
            jj = j + (0 if ins_k == NOINS else 1)
            base = li[prev][ins_k]
            total = _lae(total, base + ls[y1[i]][EPS] + rec(i + 1, jj))
            if jj < m:
                total = _lae(total, base + ls[y1[i]][y2[jj]] + rec(i + 1, jj + 1))
        return total

    return float(rec(0, 0))
