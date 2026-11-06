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
from .segments import N_SEG, N_OUT, EPS, DIST, FORTITION

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
    """Compiled edit-DP. ls: (N_SEG, N_OUT) log-sub table; li: (N_SEG+1, N_SEG+1) log-ins table.
    The inventory constants are derived from the table (EPS = NOINS = DUMMY = N_SEG = ls.shape[0]), never
    read as globals: numba's disk cache bakes globals in (see ffbs.py)."""
    EPS = ls.shape[0]; NOINS = EPS; DUMMY = EPS
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


@njit(cache=True, fastmath=False)
def _dp3(ls3, li, y1, y2):
    """_dp with a left-context substitution table ls3[prev, s, o] (prev = the input segment before s,
    DUMMY at the start): the rules u_e[(s, o, prev)] condition on the preceding input segment."""
    EPS = ls3.shape[1]; NOINS = EPS; DUMMY = EPS
    n = y1.shape[0]; m = y2.shape[0]
    cur = np.full(m + 1, NEG)
    cur[0] = 0.0
    nxt = np.full(m + 1, NEG)
    for i in range(n):
        prev = y1[i - 1] if i > 0 else DUMMY
        s = y1[i]
        noins = li[prev, NOINS]
        ldel = ls3[prev, s, EPS]
        for j in range(m + 1):
            nxt[j] = NEG
        for j in range(m + 1):
            f = cur[j]
            if f <= -5e29:
                continue
            nxt[j] = _lae_j(nxt[j], f + noins + ldel)
            if j < m:
                o = y2[j]
                v = _lae_j(f + noins + ls3[prev, s, o], f + li[prev, o] + ldel)
                nxt[j + 1] = _lae_j(nxt[j + 1], v)
                if j + 1 < m:
                    nxt[j + 2] = _lae_j(nxt[j + 2], f + li[prev, o] + ls3[prev, s, y2[j + 1]])
        tmp = cur; cur = nxt; nxt = tmp
    prev = y1[n - 1] if n > 0 else DUMMY
    out = cur[m] + li[prev, NOINS]
    if m > 0:
        out = _lae_j(out, cur[m - 1] + li[prev, y2[m - 1]])
    return out


def _lse_rows(a):
    m = a.max(axis=1, keepdims=True)
    return m + np.log(np.exp(a - m).sum(axis=1, keepdims=True))


def _norm3(psi3):
    m = psi3.max(axis=2, keepdims=True)
    return psi3 - (m + np.log(np.exp(psi3 - m).sum(axis=2, keepdims=True)))


_ARR = {}


def _arr(form):
    """Tuple form -> int64 array, memoised (forms repeat constantly)."""
    a = _ARR.get(form)
    if a is None:
        a = np.array(form, dtype=np.int64)
        if len(_ARR) < 500000:
            _ARR[form] = a
    return a


# Rate-time coupling (MASSCOMP_RATE_TIME=1): every change logit (substitution, deletion, insertion) gains
# log(t_e / T_REF), so that for short edges P(change) grows in proportion to the edge length and for
# long ones it saturates through the softmax - a continuous-time substitution process in the log-linear
# form. Off (the plan's transducer, whose intensity ignores edge length, spec 12.13), the tables are
# unchanged for every t. T_REF is the length at which the two agree (the daughters' depth in the real-data
# forests), so a fitted lambda keeps its meaning.
RATE_TIME = __import__("os").environ.get("MASSCOMP_RATE_TIME", "0") == "1"
T_REF = float(__import__("os").environ.get("MASSCOMP_T_REF", "0.3"))
# Directional cost (MASSCOMP_DIRECTIONAL=1): a fifth weight lam4 on the consonantal strength GAINED by a
# substitution (segments.FORTITION; lenition and loss gain nothing), so that b -> p costs more than p -> b
# and a root that is the union of its daughters' segments stops being the cheapest story.
DIRECTIONAL = __import__("os").environ.get("MASSCOMP_DIRECTIONAL", "0") == "1"


class EdgeParams:
    """Dense log-probability tables for one edge, built from (lam, u_e); with t, the edge length, when
    the rate-time coupling is on."""

    def __init__(self, lam, u=None, t=None):
        lam0, lam1, lam2, lam3 = lam[:4]
        u = u or {}
        psi = np.zeros((N_SEG, N_OUT))
        psi[:, :N_SEG] += lam0 * np.eye(N_SEG) - lam1 * DIST
        psi[:, EPS] += -lam2
        if DIRECTIONAL and len(lam) > 4:
            psi[:, :N_SEG] -= lam[4] * FORTITION
        ins = np.zeros((N_SEG + 1, N_SEG + 1))
        ins[:N_SEG, :N_SEG] += -lam3 - lam1 * DIST
        ins[DUMMY, :N_SEG] += -lam3
        ins[:, NOINS] = 0.0
        self.t = t
        if RATE_TIME and t is not None:
            shift = math.log(t / T_REF)
            psi[:, :N_SEG] += shift * (1.0 - np.eye(N_SEG))
            psi[:, EPS] += shift
            ins[:, :N_SEG] += shift
        psi3 = None                                       # (N_SEG+1 contexts, N_SEG, N_OUT), only with conditioned rules
        for key, w in u.items():
            if key[0] == "ins":
                if len(key) == 3:                         # ("ins", o, ctx): insertion of o after ctx only
                    ins[key[2], key[1]] += w
                else:
                    ins[:, key[1]] += w
            elif len(key) == 3:                           # (s, o, ctx): s -> o after ctx only
                if psi3 is None:
                    psi3 = np.repeat(psi[None, :, :], N_SEG + 1, axis=0)
                psi3[key[2], key[0], key[1]] += w
            else:
                psi[key[0], key[1]] += w
                if psi3 is not None:
                    psi3[:, key[0], key[1]] += w
        self._psi, self._ins, self._psi3 = psi, ins, psi3   # unnormalised, kept for derive()
        self.log_sub_np = psi - _lse_rows(psi)             # the context-free table: the lattice proposals use it
        self.log_ins_np = ins - _lse_rows(ins)
        self.log_sub3_np = _norm3(psi3) if psi3 is not None else None
        self._cache = {}
        self._cache_lat = None

    @property
    def top_parents(self):
        """For each child segment o: the parent segments s != o among the 3 with the largest log P(o | s),
        as (s, logp) pairs, and the parent segment most likely to be deleted; used by the candidate
        sets of the Gibbs form move. Computed once per table."""
        tp = getattr(self, "_top_parents", None)
        if tp is None:
            ls = self.log_sub_np
            order = np.argsort(-ls[:, :N_SEG], axis=0)          # (N_SEG parents, N_SEG children)
            per_o = []
            for o in range(N_SEG):
                per_o.append([(int(s_), float(ls[s_, o])) for s_ in order[:3, o] if int(s_) != o])
            tp = self._top_parents = (per_o, int(np.argmax(ls[:, EPS])))
        return tp

    def derive(self, slot, delta):
        """The EdgeParams with `delta` added to one rule slot: for a substitution/deletion slot (s, o)
        only row s of the substitution table changes; for an insertion slot ("ins", o) the insertion
        table is renormalised. Tables that do not change are shared with self (they are never
        mutated), so the derived object costs one row or one small table, not a rebuild."""
        new = EdgeParams.__new__(EdgeParams)
        new._cache = {}; new._cache_lat = None; new.t = self.t
        new._psi, new._ins, new._psi3 = self._psi, self._ins, self._psi3
        new.log_sub_np, new.log_ins_np, new.log_sub3_np = self.log_sub_np, self.log_ins_np, self.log_sub3_np
        if slot[0] == "ins":
            ins = self._ins.copy()
            if len(slot) == 3:
                ins[slot[2], slot[1]] += delta
                li = self.log_ins_np.copy(); row = ins[slot[2]]; m = row.max()
                li[slot[2]] = row - (m + np.log(np.exp(row - m).sum()))
                new.log_ins_np = li
            else:
                ins[:, slot[1]] += delta
                new.log_ins_np = ins - _lse_rows(ins)
            new._ins = ins
        elif len(slot) == 3:
            s_, o, ctx = slot
            psi3 = (self._psi3 if self._psi3 is not None else np.repeat(self._psi[None, :, :], N_SEG + 1, axis=0)).copy()
            psi3[ctx, s_, o] += delta
            ls3 = self.log_sub3_np.copy() if self.log_sub3_np is not None else np.repeat(self.log_sub_np[None, :, :], N_SEG + 1, axis=0)
            row = psi3[ctx, s_]; m = row.max()
            ls3[ctx, s_] = row - (m + np.log(np.exp(row - m).sum()))
            new._psi3, new.log_sub3_np = psi3, ls3
        else:
            s_, o = slot
            psi = self._psi.copy(); psi[s_, o] += delta
            ls = self.log_sub_np.copy()
            row = psi[s_]; m = row.max()
            ls[s_] = row - (m + np.log(np.exp(row - m).sum()))
            new._psi, new.log_sub_np = psi, ls
            if self._psi3 is not None:
                psi3 = self._psi3.copy(); psi3[:, s_, o] += delta
                ls3 = self.log_sub3_np.copy()
                rows = psi3[:, s_, :]; m3 = rows.max(axis=1, keepdims=True)
                ls3[:, s_, :] = rows - (m3 + np.log(np.exp(rows - m3).sum(axis=1, keepdims=True)))
                new._psi3, new.log_sub3_np = psi3, ls3
        return new

    # the list and cumulative tables are only needed by the pure-Python reference DP and the simulator
    @property
    def log_sub(self):
        return self.log_sub_np.tolist()

    @property
    def log_ins(self):
        return self.log_ins_np.tolist()

    @property
    def _p_sub(self):
        return np.exp(self.log_sub_np)

    @property
    def _p_ins(self):
        return np.exp(self.log_ins_np)

    @property
    def _c_sub(self):
        c = getattr(self, "_c_sub_", None)
        if c is None:
            c = self._c_sub_ = np.cumsum(self._p_sub, axis=1)
        return c

    @property
    def _c_sub3(self):
        c = getattr(self, "_c_sub3_", None)
        if c is None:
            c = self._c_sub3_ = np.cumsum(np.exp(self.log_sub3_np), axis=2)
        return c

    @property
    def _c_ins(self):
        c = getattr(self, "_c_ins_", None)
        if c is None:
            c = self._c_ins_ = np.cumsum(self._p_ins, axis=1)
        return c

    @classmethod
    def from_tables(cls, log_sub, log_ins):
        """An EdgeParams over given (already normalised) log tables; used for proposal transducers."""
        self = cls.__new__(cls)
        self.log_sub_np = np.asarray(log_sub, dtype=float); self.log_ins_np = np.asarray(log_ins, dtype=float)
        self.log_sub3_np = None; self._psi3 = None
        self._cache = {}; self._cache_lat = None
        return self

    def reverse(self):
        """Approximate inverse transducer child -> parent, for proposing ancestral forms.

        Row o of the reverse substitution table is P(parent segment s | child segment o) under a
        uniform prior on s, i.e. column o of the forward table renormalised; its EPS entry (the
        child segment was inserted, so the parent has nothing here) is the forward insertion
        probability of o averaged over contexts. Reverse insertion of s (the parent had an s the
        child deleted) gets the forward deletion probability of s. A rule s -> o with large u_e
        therefore makes the reverse propose s wherever the child shows o, which the forward
        transducer applied to the child never does. Exact as a proposal (its density is computed by
        the same DP); approximate as an inverse."""
        r = getattr(self, "_rev", None)
        if r is None:
            ls, li = self.log_sub_np, self.log_ins_np
            rs = np.full((N_SEG, N_OUT), NEG)
            rs[:, :N_SEG] = ls[:, :N_SEG].T                                    # rs[o, s] = log P(o | s)
            rs[:, EPS] = np.logaddexp.reduce(li[:, :N_SEG], axis=0) - math.log(li.shape[0])
            rs -= np.logaddexp.reduce(rs, axis=1, keepdims=True)
            ri = np.zeros((N_SEG + 1, N_SEG + 1))
            ri[:, :N_SEG] = ls[:, EPS][None, :] - math.log(N_SEG)              # insert s back w.p. ~ P(s deleted) / N_SEG
            ri[:, NOINS] = 0.0
            ri -= np.logaddexp.reduce(ri, axis=1, keepdims=True)
            r = self._rev = EdgeParams.from_tables(rs, ri)
        return r

    def logprob(self, y1, y2):
        key = (y1, y2)
        v = self._cache.get(key)
        if v is None:
            v = self._logprob(y1, y2)
            if len(self._cache) < 200000:
                self._cache[key] = v
        return v

    @property
    def conditioned(self):
        return self.log_sub3_np is not None

    def logprob_lattice(self, y1, y2):
        """log probability under the context-free tables (log_sub_np, log_ins_np): the density the
        alignment lattice (ffbs.py) samples from. Equal to logprob() unless the edge carries
        context-conditioned substitution rules, in which case the lattice is a proposal and the MH
        ratio uses this for q and logprob() for the target."""
        if self.log_sub3_np is None:
            return self.logprob(y1, y2)
        if self._cache_lat is None:
            self._cache_lat = {}
        key = (y1, y2)
        v = self._cache_lat.get(key)
        if v is None:
            v = _dp(self.log_sub_np, self.log_ins_np, _arr(y1), _arr(y2))
            if v <= -5e29:
                v = FLOOR - FLOOR_PER_SEG * max(0, len(y2) - 2 * len(y1) - 1)
            if len(self._cache_lat) < 200000:
                self._cache_lat[key] = v
        return v

    def _logprob(self, y1, y2):
        if self.log_sub3_np is not None:
            v = _dp3(self.log_sub3_np, self.log_ins_np, _arr(y1), _arr(y2))
        else:
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
        c3 = self._c_sub3 if self.log_sub3_np is not None else None
        for i in range(n + 1):
            prev = y1[i - 1] if i > 0 else DUMMY
            k = int(np.searchsorted(self._c_ins[prev], us[k_u])); k_u += 1
            if k < NOINS:
                out.append(k)
            if i < n:
                row = c3[prev, y1[i]] if c3 is not None else self._c_sub[y1[i]]
                o = int(np.searchsorted(row, us[k_u])); k_u += 1
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
