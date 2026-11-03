"""MCMC over (y, event, donor, theta) (plan §5), with optional tempering of the likelihood.

Deviations from the plan, recorded here:
  * rho_c and omega are updated by random-walk MH on the log scale, not Gibbs: the single-event
    hazard probabilities are not conjugate to the Gamma prior.
  * t_root is updated by random-walk MH on the log scale, not slice sampling.
"""
import math
import numpy as np
from .segments import N_SEG, EPS
from .transducer import EdgeParams
from . import events as ev
from .likelihood import State
from .tree import with_root_length

MAX_FORM_LEN = 8


class Sampler:
    def __init__(self, ds, theta, rng, beta=1.0, init=None, init_from_prior=False):
        self.rng = rng
        self.beta = beta
        y, event, donor = {}, {}, {}
        f = ds.forest
        if len(f.roots) == 1 and f.family[f.roots[0]] == -1:
            # the sampler owns its own copy of the forest; the root-edge length is a parameter
            from dataclasses import replace
            ds = replace(ds, forest=with_root_length(f, theta.t_root))
            f = ds.forest
        if init is not None:
            y, event, donor = init
        elif init_from_prior:
            # a forward simulation under theta: a typical state of the prior, not a copy of the data
            from .simulate import simulate
            sim = simulate(f, theta, ds.M, rng)
            y = {k: v for k, v in sim.truth["y"].items() if k[0] not in set(f.leaves)}
            event, donor = sim.truth["event"], sim.truth["donor"]
        else:
            # initialise internal forms to a child's form, events to inherit
            leafset = set(f.leaves)
            for c in range(ds.M):
                for v in reversed(f.topo):
                    if v in leafset:
                        continue
                    ch = f.children[v][0]
                    src = ds.leaf_forms[(ch, c)] if (ch, c) in ds.leaf_forms else y[(ch, c)]
                    y[(v, c)] = (src + src)[:max(2, min(len(src), 6))]
                for e in f.edges:
                    event[(e, c)] = ev.INHERIT
        self.st = State(ds, theta, y, event, donor)
        self.acc = {k: [0, 0] for k in ["y", "gibbs", "joint", "lam", "u", "rho", "omega", "troot"]}
        self.inlattice = 1.0
        if init is None:
            self.move_events()   # remove impossible inherit terms from the initial state

    # ------------------------------------------------------------------ helpers
    def _accept(self, dlog):
        return dlog >= 0 or self.rng.random() < math.exp(dlog)

    def _delta_terms(self, pairs, over):
        """Sum of (prior + beta*lik) for the given (e,c) pairs under overrides, minus current."""
        st = self.st
        new = {}
        d = 0.0
        for (e, c) in pairs:
            pr, lk = st.edge_terms(e, c, **over.get((e, c), {}))
            new[(e, c)] = (pr, lk)
            d += (pr - st.t_prior[(e, c)]) + self.beta * (lk - st.t_lik[(e, c)])
        return d, new

    def _commit(self, new):
        for k, (pr, lk) in new.items():
            self.st.t_prior[k] = pr
            self.st.t_lik[k] = lk

    # ------------------------------------------------------------------ move 1: forms
    def move_forms(self, n_prop=5):
        st, f, rng = self.st, self.st.f, self.rng
        for v in f.internal:
            for c in range(st.M):
                for k_prop in range(n_prop):
                    cur = st.y[(v, c)]
                    L = len(cur)
                    kind = rng.integers(4) if k_prop % 2 == 0 else 3
                    logq = 0.0
                    if kind == 3:            # independence proposal from a child's form
                        # forward (apply the child's edge transducer to the child) or reverse
                        # (approximate inverse transducer: the parent segment the child's segment
                        # most plausibly came from, rules included); mixture over children
                        chs = f.children[v]
                        ch = chs[int(rng.integers(len(chs)))]
                        rev = rng.random() < 0.5
                        T = (lambda c2: st.ep[c2].reverse()) if rev else (lambda c2: st.ep[c2])
                        prop = T(ch).sample(st.y[(ch, c)], rng)
                        if len(prop) == 0 or len(prop) > MAX_FORM_LEN or prop == cur:
                            continue
                        lk_ = math.log(len(chs))
                        q_prop = _lse([T(c2).logprob(st.y[(c2, c)], prop) for c2 in chs]) - lk_
                        q_cur = _lse([T(c2).logprob(st.y[(c2, c)], cur) for c2 in chs]) - lk_
                        logq = q_cur - q_prop
                    elif kind == 0:          # substitute
                        pos = int(rng.integers(L))
                        s = int(rng.integers(N_SEG - 1)); s = s + 1 if s >= cur[pos] else s
                        prop = cur[:pos] + (s,) + cur[pos + 1:]
                    elif kind == 1:          # delete
                        if L <= 1:
                            continue
                        pos = int(rng.integers(L))
                        prop = cur[:pos] + cur[pos + 1:]
                        logq = -math.log(N_SEG)   # reverse insert chooses 1 of N_SEG segments
                    else:                    # insert
                        if L >= MAX_FORM_LEN:
                            continue
                        pos = int(rng.integers(L + 1))
                        prop = cur[:pos] + (int(rng.integers(N_SEG)),) + cur[pos:]
                        logq = math.log(N_SEG)
                    # terms
                    edges = st.affected_edges(v, c)
                    over = {}
                    for e in edges:
                        if e == v:
                            over[(e, c)] = dict(y_child=prop)
                        elif f.parent[e] == v:
                            over[(e, c)] = dict(y_parent=prop)
                        else:
                            over[(e, c)] = {}
                    # borrowings from v need the donor form overridden: temporarily set
                    old = st.y[(v, c)]
                    st.y[(v, c)] = prop
                    dterm, new = self._delta_terms([(e, c) for e in edges], over)
                    droot = 0.0
                    if f.parent[v] < 0:
                        droot = ev.log_G(prop) - st.t_prior[("root", v, c)]
                    self.acc["y"][1] += 1
                    if self._accept(dterm + droot + logq):
                        self._commit(new)
                        if f.parent[v] < 0:
                            st.t_prior[("root", v, c)] = ev.log_G(prop)
                        self.acc["y"][0] += 1
                    else:
                        st.y[(v, c)] = old

    # ------------------------------------------------------------------ move 1a: lattice Gibbs
    def candidates(self, v, c, max_per_child=12):
        """Candidate ancestral forms for (v, c), built from the NEIGHBOURS only (parent, children,
        donor-linked nodes) and never from the current form, so the set is a function of the
        conditioning state and a Gibbs draw over it is exact. Contents: each neighbour's form;
        for each child form, the single-segment variants that the edge transducer most favours
        as a parent (top `max_per_child` by log P(child_seg | parent_seg)); each child form with
        one segment deleted; and the child form with its best single insertion removed."""
        st, f = self.st, self.st.f
        S = set()
        if f.parent[v] >= 0:
            S.add(st.y[(f.parent[v], c)])
        neigh = list(f.children[v])
        for e in f.edges:
            if st.event.get((e, c)) == ev.BORROW and st.donor.get((e, c)) == v:
                neigh.append(e)
        if st.event.get((v, c)) == ev.BORROW:
            S.add(st.y[(st.donor[(v, c)], c)])
        for ch in neigh:
            x = st.y[(ch, c)]
            if not x:
                continue
            S.add(x)
            ls = st.ep[ch].log_sub_np if ch in st.ep else st.adapt.log_sub_np
            scored = []
            for i, o in enumerate(x):
                col = ls[:, o]                       # log P(o | s) for every parent segment s
                for s_ in np.argsort(-col)[:3]:
                    s_ = int(s_)
                    if s_ != o:
                        scored.append((col[s_], x[:i] + (s_,) + x[i + 1:]))
            scored.sort(key=lambda t: -t[0])
            for _, cand in scored[:max_per_child]:
                S.add(cand)
            if len(x) > 1:
                for i in range(len(x)):               # child inserted a segment: parent lacks it
                    S.add(x[:i] + x[i + 1:])
            if len(x) < MAX_FORM_LEN:                 # child deleted a segment: parent had one more
                ldel = ls[:, EPS]
                s_ = int(np.argmax(ldel))
                for i in range(len(x) + 1):
                    S.add(x[:i] + (s_,) + x[i:])
        S = {s for s in S if 0 < len(s) <= MAX_FORM_LEN}
        return S

    def move_forms_gibbs(self):
        """Exact Gibbs draw of y[v,c] over candidates(v,c) when the current form is in that set;
        otherwise no-op (the MH moves handle it). Returns fraction of (v,c) that were in-lattice."""
        st, f, rng = self.st, self.st.f, self.rng
        inlat = 0; tot = 0
        for v in f.internal:
            is_root = f.parent[v] < 0
            for c in range(st.M):
                cur = st.y[(v, c)]
                S = self.candidates(v, c)
                tot += 1
                if cur not in S:
                    continue
                inlat += 1
                cands = list(S)
                edges = st.affected_edges(v, c)
                logw = np.empty(len(cands)); terms = []
                for i, cand in enumerate(cands):
                    st.y[(v, c)] = cand
                    w = ev.log_G(cand) if is_root else 0.0
                    tt = {}
                    for e in edges:
                        pr, lk = st.edge_terms(e, c)
                        tt[(e, c)] = (pr, lk); w += pr + self.beta * lk
                    logw[i] = w; terms.append(tt)
                p = np.exp(logw - logw.max()); p /= p.sum()
                i = int(rng.choice(len(cands), p=p))
                st.y[(v, c)] = cands[i]
                self._commit(terms[i])
                if is_root:
                    st.t_prior[("root", v, c)] = ev.log_G(cands[i])
                self.acc["gibbs"][0] += cands[i] != cur; self.acc["gibbs"][1] += 1
        return inlat / tot if tot else 1.0

    # ------------------------------------------------------------------ move 1b: joint (event, form)
    def _gen_logq(self, v, c, kind, d, form):
        """Density of the model's own conditional for y_v given the event kind (used as a proposal)."""
        st, f = self.st, self.st.f
        if kind == ev.INHERIT:
            return st.ep[v].logprob(st.y[(f.parent[v], c)], form)
        if kind == ev.REPLACE:
            return ev.log_G(form)
        return st.adapt.logprob(st.y[(d, c)], form)

    def move_joint(self, n_prop=2):
        """For internal non-root v: propose a new event on its incoming edge together with a new
        form drawn from that event's generative distribution. Escapes the mode where sibling
        forms are pinned together by borrowing."""
        st, f, rng = self.st, self.st.f, self.rng
        for v in f.internal:
            if f.parent[v] < 0:
                continue
            cands = st.cands[v]
            n_opt = 2 + len(cands)
            for _ in range(n_prop):
                for c in range(st.M):
                    cur_form, cur_kind = st.y[(v, c)], st.event[(v, c)]
                    cur_d = st.donor.get((v, c))
                    i = int(rng.integers(n_opt))
                    if i < 2:
                        kind, d = i, None
                    else:
                        kind, d = ev.BORROW, cands[i - 2]
                    if kind == ev.INHERIT:
                        prop = st.ep[v].sample(st.y[(f.parent[v], c)], rng)
                    elif kind == ev.REPLACE:
                        prop = ev.sample_G(rng)
                    else:
                        prop = st.adapt.sample(st.y[(d, c)], rng)
                    if len(prop) == 0 or len(prop) > MAX_FORM_LEN:
                        continue
                    logq_new = self._gen_logq(v, c, kind, d, prop)
                    logq_old = self._gen_logq(v, c, cur_kind, cur_d, cur_form)
                    # terms
                    edges = st.affected_edges(v, c)
                    over = {(v, c): dict(y_child=prop, kind=kind, d=d)}
                    for e in edges:
                        if e != v:
                            over[(e, c)] = dict(y_parent=prop) if f.parent[e] == v else {}
                    st.y[(v, c)] = prop
                    dterm, new = self._delta_terms([(e, c) for e in edges], over)
                    self.acc["joint"][1] += 1
                    if self._accept(dterm + logq_old - logq_new):
                        self._commit(new)
                        st.event[(v, c)] = kind
                        if kind == ev.BORROW:
                            st.donor[(v, c)] = d
                        else:
                            st.donor.pop((v, c), None)
                        self.acc["joint"][0] += 1
                    else:
                        st.y[(v, c)] = cur_form

    # ------------------------------------------------------------------ move 2: events
    def move_events(self):
        st, f, rng = self.st, self.st.f, self.rng
        for e in f.edges:
            for c in range(st.M):
                opts, logw = [], []
                for k in (ev.INHERIT, ev.REPLACE):
                    pr, lk = st.edge_terms(e, c, kind=k)
                    opts.append((k, None, pr, lk)); logw.append(pr + self.beta * lk)
                for d in st.cands[e]:
                    pr, lk = st.edge_terms(e, c, kind=ev.BORROW, d=d)
                    opts.append((ev.BORROW, d, pr, lk)); logw.append(pr + self.beta * lk)
                w = np.array(logw); w = np.exp(w - w.max()); w /= w.sum()
                k, d, pr, lk = opts[int(rng.choice(len(opts), p=w))]
                st.event[(e, c)] = k
                if k == ev.BORROW:
                    st.donor[(e, c)] = d
                else:
                    st.donor.pop((e, c), None)
                st.t_prior[(e, c)] = pr; st.t_lik[(e, c)] = lk

    # ------------------------------------------------------------------ move 3: lambda
    def move_lam(self, step=None):
        """One joint proposal on all four components (one global recompute instead of four)."""
        st = self.st
        if step is None:
            # per-component scales from measured posterior sds (deletion/insertion rates are
            # weakly identified), widened at low beta where the prior is much broader
            step = np.array([0.08, 0.2, 0.3, 0.08]) * (1.0 + 5.0 * (1.0 - self.beta))
        old = st.theta.lam.copy()
        st.theta.lam = old + self.rng.normal(0, 1, size=4) * step
        d = self._try_global(ev.log_prior_lam(st.theta.lam) - ev.log_prior_lam(old))
        self.acc["lam"][1] += 1
        if d is not None:
            self.acc["lam"][0] += 1
        else:
            st.theta.lam = old

    def _try_global(self, dprior_param):
        """Rebuild edge params, recompute all edge terms, accept/reject; returns delta or None."""
        st = self.st
        old_ep, old_adapt = st.ep, st.adapt
        st.rebuild_edge_params()
        d = dprior_param
        new = {}
        for c in range(st.M):
            for e in st.f.edges:
                pr, lk = st.edge_terms(e, c)
                new[(e, c)] = (pr, lk)
                d += (pr - st.t_prior[(e, c)]) + self.beta * (lk - st.t_lik[(e, c)])
        if self._accept(d):
            self._commit(new)
            st.t_param += dprior_param
            return d
        st.ep, st.adapt = old_ep, old_adapt
        return None

    # ------------------------------------------------------------------ move 4: u (spike-and-slab)
    U_DATA_FRAC, U_DATA_MU, U_DATA_SD = 0.5, 4.0, 2.0

    def _u_counts(self, e):
        """Crude substitution/deletion/insertion counts on edge e from the current
        (parent, child) form pairs: equal length -> position-wise mismatches; child one shorter ->
        the segment(s) whose removal gives the child; child one longer -> the extra segment.
        Depends on the latent forms only (not on u_e), so forward and reverse proposal
        probabilities within move_u are computed from the same counts."""
        st, f = self.st, self.st.f
        p = f.parent[e]
        counts = {}
        for c in range(st.M):
            x, y = st.y[(p, c)], st.y[(e, c)]      # all pairs: a true inherited-with-u pair may currently be labelled borrow
            if len(x) == len(y):
                for a, b in zip(x, y):
                    if a != b:
                        counts[(a, b)] = counts.get((a, b), 0) + 1
            elif len(x) == len(y) + 1:
                for i in range(len(x)):
                    if x[:i] + x[i + 1:] == y:
                        counts[(x[i], EPS)] = counts.get((x[i], EPS), 0) + 1
            elif len(y) == len(x) + 1:
                for j in range(len(y)):
                    if y[:j] + y[j + 1:] == x:
                        counts[("ins", y[j])] = counts.get(("ins", y[j]), 0) + 1
        return counts

    def _u_slot_logq(self, slot, absent, counts):
        """log q(slot) for a birth over the given absent set: mixture of uniform and count-weighted."""
        tot = sum(counts.get(a, 0) for a in absent)
        q = (1 - self.U_DATA_FRAC) / len(absent)
        if tot > 0:
            q += self.U_DATA_FRAC * counts.get(slot, 0) / tot
        return math.log(q)

    def _u_w_logq(self, w, t):
        """log q(w): mixture of the prior slab and a data-informed positive component."""
        a = math.log(1 - self.U_DATA_FRAC) + _lognorm(w, t * ev.SIGMA2_U)
        b = math.log(self.U_DATA_FRAC) + _lognorm(w - self.U_DATA_MU, self.U_DATA_SD ** 2)
        return _lse([a, b])

    def move_u(self, n_prop=6, perturb_sd=0.5):
        """Birth / death / perturb on the sparse branch terms u_e, collapsed over the events on edge e.

        Why collapsed: a concept whose parent-child mismatch is really a branch rule (p -> f) can be
        explained by borrowing from a neighbour instead. Once the chain has done that, an ordinary
        u_e proposal changes no likelihood term (borrow and replace terms do not depend on u_e) and
        is rejected on its prior cost alone, so the rule is never found and omega inflates (seen in
        the M=100 recovery study: 0.3% inclusion of detectable rules, omega 2.7x over). Here the
        proposal is (u_e', events_e') with events_e' ~ p(. | u_e', rest), the exact conditional that
        move_events samples from, so the acceptance ratio is the ratio of the per-concept sums over
        all event options, Z_c(u') / Z_c(u), times the prior and proposal ratios on u_e.

        Birth proposes the slot from a mixture of uniform and the mismatch counts on the edge, and
        the value from a mixture of the prior slab and N(U_DATA_MU, U_DATA_SD^2); death uses the
        same densities in reverse."""
        st, f, rng = self.st, self.st.f, self.rng
        for e in f.edges:
            t = f.length[e]
            counts = self._u_counts(e)
            # event options that do not depend on u_e: (kind, donor, prior_part, lik_part), once per concept
            fixed = []
            for c in range(st.M):
                opts = []
                pr, lk = st.edge_terms(e, c, kind=ev.REPLACE); opts.append((ev.REPLACE, None, pr, lk))
                for d in st.cands[e]:
                    pr, lk = st.edge_terms(e, c, kind=ev.BORROW, d=d); opts.append((ev.BORROW, d, pr, lk))
                fixed.append(opts)
            # inherit terms under the current u_e, and the current per-concept log Z
            inh = [st.edge_terms(e, c, kind=ev.INHERIT) for c in range(st.M)]
            logZ = [_lse([pr + self.beta * lk for (_, _, pr, lk) in fixed[c]] + [inh[c][0] + self.beta * inh[c][1]]) for c in range(st.M)]
            for _ in range(n_prop):
                u = st.theta.u.setdefault(e, {})
                present = list(u.keys())
                kind = rng.integers(3)
                prop = dict(u)
                logq = 0.0
                if kind == 0:                                  # birth
                    absent = [sl for sl in ev.U_SLOTS if sl not in u]
                    if not absent:
                        continue
                    if rng.random() < self.U_DATA_FRAC:
                        tot = sum(counts.get(a, 0) for a in absent)
                        if tot > 0:
                            wts = np.array([counts.get(a, 0) for a in absent], dtype=float) / tot
                            slot = absent[int(rng.choice(len(absent), p=wts))]
                        else:
                            slot = absent[int(rng.integers(len(absent)))]
                    else:
                        slot = absent[int(rng.integers(len(absent)))]
                    if rng.random() < self.U_DATA_FRAC:
                        w = float(rng.normal(self.U_DATA_MU, self.U_DATA_SD))
                    else:
                        w = float(rng.normal(0, math.sqrt(t * ev.SIGMA2_U)))
                    prop[slot] = w
                    logq = -math.log(len(present) + 1) - (self._u_slot_logq(slot, absent, counts) + self._u_w_logq(w, t))
                elif kind == 1:                                # death
                    if not present:
                        continue
                    slot = present[int(rng.integers(len(present)))]
                    w = prop.pop(slot)
                    absent = [sl for sl in ev.U_SLOTS if sl not in prop]       # includes the removed slot
                    logq = (self._u_slot_logq(slot, absent, counts) + self._u_w_logq(w, t)) - (-math.log(len(present)))
                else:                                          # perturb
                    if not present:
                        continue
                    slot = present[int(rng.integers(len(present)))]
                    prop[slot] = u[slot] + float(rng.normal(0, perturb_sd))
                dprior = ev.log_prior_u(prop, t) - ev.log_prior_u(u, t)
                old_ep = st.ep[e]
                st.ep[e] = EdgeParams(st.theta.lam, prop)
                inh_new = [st.edge_terms(e, c, kind=ev.INHERIT) for c in range(st.M)]
                logZ_new = [_lse([pr + self.beta * lk for (_, _, pr, lk) in fixed[c]] + [inh_new[c][0] + self.beta * inh_new[c][1]]) for c in range(st.M)]
                d = dprior + logq + sum(logZ_new) - sum(logZ)
                self.acc["u"][1] += 1
                if self._accept(d):
                    st.theta.u[e] = prop
                    st.t_param += dprior
                    self.acc["u"][0] += 1
                    inh, logZ = inh_new, logZ_new
                    # events on edge e from their conditional under the new u_e
                    for c in range(st.M):
                        opts = fixed[c] + [(ev.INHERIT, None, inh[c][0], inh[c][1])]
                        w = np.array([pr + self.beta * lk for (_, _, pr, lk) in opts]); w = np.exp(w - w.max()); w /= w.sum()
                        k, dn, pr, lk = opts[int(rng.choice(len(opts), p=w))]
                        st.event[(e, c)] = k
                        if k == ev.BORROW:
                            st.donor[(e, c)] = dn
                        else:
                            st.donor.pop((e, c), None)
                        st.t_prior[(e, c)] = pr; st.t_lik[(e, c)] = lk
                else:
                    st.ep[e] = old_ep

    # ------------------------------------------------------------------ move 5/6: rates
    def _event_remainders(self, cs):
        """For each (e, c) with c in cs: the event options (kind, donor, pr_minus_evp, lk), i.e. the
        edge terms with the rate-dependent event log-probability removed. These depend on the forms
        and u_e only, so a rate proposal changes just the evp part and the per-(e, c) sum over
        options can be recomputed cheaply."""
        st, f = self.st, self.st.f
        rem = {}
        for c in cs:
            for e in f.edges:
                opts = []
                for k in (ev.INHERIT, ev.REPLACE):
                    pr, lk = st.edge_terms(e, c, kind=k)
                    opts.append((k, None, pr - st.evp[e][c][k], lk))
                for d in st.cands[e]:
                    pr, lk = st.edge_terms(e, c, kind=ev.BORROW, d=d)
                    opts.append((ev.BORROW, d, pr - st.evp[e][c][ev.BORROW], lk))
                rem[(e, c)] = opts
        return rem

    def _logZ(self, rem):
        st = self.st
        return {ec: _lse([r + st.evp[ec[0]][ec[1]][k] + self.beta * lk for (k, _, r, lk) in opts]) for ec, opts in rem.items()}

    def _resample_events(self, rem):
        """Gibbs draw of the events in rem from their conditional under the current evp."""
        st, rng = self.st, self.rng
        for (e, c), opts in rem.items():
            w = np.array([r + st.evp[e][c][k] + self.beta * lk for (k, _, r, lk) in opts]); w = np.exp(w - w.max()); w /= w.sum()
            k, dn, r, lk = opts[int(rng.choice(len(opts), p=w))]
            st.event[(e, c)] = k
            if k == ev.BORROW:
                st.donor[(e, c)] = dn
            else:
                st.donor.pop((e, c), None)
            st.t_prior[(e, c)] = r + st.evp[e][c][k]; st.t_lik[(e, c)] = lk

    def move_rates(self, step=0.3):
        """Log random walks on rho_c and omega, collapsed over the events they govern.

        Why collapsed: a state with many concepts labelled 'borrow' makes omega large, and a large
        omega makes 'borrow' cheap; with the events held fixed a rate proposal can only ever move
        within that loop (seen as omega 10-20x over truth with the planted rule unfound). The
        proposal is (rate', events') with events' from their exact conditional under rate', so the
        acceptance ratio is the product over (e, c) of Z_ec(rate') / Z_ec(rate)."""
        st, f, rng = self.st, self.st.f, self.rng
        for c in range(st.M):
            old = st.theta.rho[c]
            prop = old * math.exp(rng.normal(0, step))
            dprior = ev.log_prior_rho(np.array([prop])) - ev.log_prior_rho(np.array([old]))
            rem = self._event_remainders([c])
            logZ_old = self._logZ(rem)
            old_evp = {e: st.evp[e][c] for e in f.edges}
            st.theta.rho[c] = prop
            for e in f.edges:
                st.evp[e][c] = ev.event_logprobs(prop, st.theta.omega if st.cands[e] else 0.0, f.length[e])
            logZ_new = self._logZ(rem)
            d = dprior + math.log(prop / old) + sum(logZ_new.values()) - sum(logZ_old.values())
            self.acc["rho"][1] += 1
            if self._accept(d):
                st.t_param += dprior; self.acc["rho"][0] += 1
                self._resample_events(rem)
            else:
                st.theta.rho[c] = old
                for e in f.edges:
                    st.evp[e][c] = old_evp[e]
        # omega
        old = st.theta.omega
        prop = old * math.exp(rng.normal(0, step))
        dprior = ev.log_prior_omega(prop) - ev.log_prior_omega(old)
        rem = self._event_remainders(range(st.M))
        logZ_old = self._logZ(rem)
        old_evp = {e: list(st.evp[e]) for e in f.edges}
        st.theta.omega = prop
        st.rebuild_event_probs()
        logZ_new = self._logZ(rem)
        d = dprior + math.log(prop / old) + sum(logZ_new.values()) - sum(logZ_old.values())
        self.acc["omega"][1] += 1
        if self._accept(d):
            st.t_param += dprior; self.acc["omega"][0] += 1
            self._resample_events(rem)
        else:
            st.theta.omega = old; st.evp = old_evp

    # ------------------------------------------------------------------ move 7: t_root
    def move_troot(self, step=0.3):
        st, f, rng = self.st, self.st.f, self.rng
        if not st.joined:
            return
        old = st.theta.t_root
        prop = old * math.exp(rng.normal(0, step))
        dprior = ev.log_prior_troot(prop) - ev.log_prior_troot(old)
        for e in st.root_edges:
            dprior += ev.log_prior_u(st.theta.u.get(e, {}), prop) - ev.log_prior_u(st.theta.u.get(e, {}), old)
        old_len = list(f.length)
        for e in st.root_edges:
            f.length[e] = prop
        old_evp = {e: list(st.evp[e]) for e in st.root_edges}
        st.rebuild_event_probs(st.root_edges)
        d = dprior + math.log(prop / old); new = {}
        for c in range(st.M):
            for e in st.root_edges:
                k = st.event[(e, c)]
                pr = st.t_prior[(e, c)] - old_evp[e][c][k] + st.evp[e][c][k]
                new[(e, c)] = (pr, st.t_lik[(e, c)]); d += pr - st.t_prior[(e, c)]
        self.acc["troot"][1] += 1
        if self._accept(d):
            st.theta.t_root = prop; self._commit(new); st.t_param += dprior; self.acc["troot"][0] += 1
        else:
            f.length[:] = old_len
            for e in st.root_edges:
                st.evp[e] = old_evp[e]

    # ------------------------------------------------------------------ sweep
    def sweep(self, clamp_latents=False, clamp_u=False, clamp_params=False):
        if not clamp_latents:
            self.inlattice = self.move_forms_gibbs()
            self.move_forms(n_prop=1); self.move_joint(); self.move_events()
        if clamp_params:
            return
        self.move_lam()
        if not clamp_u:
            self.move_u()
        self.move_rates(); self.move_troot()

    def snapshot(self):
        st = self.st
        return dict(lam=st.theta.lam.copy(), rho=st.theta.rho.copy(), omega=st.theta.omega, t_root=st.theta.t_root,
                    u={e: dict(d) for e, d in st.theta.u.items()},
                    roots={(v, c): st.y[(v, c)] for v in st.f.internal if st.f.parent[v] < 0 or st.f.parent[v] in st.f.roots for c in range(st.M)},
                    log_lik=st.log_lik(), log_prior=st.log_prior())


def _lse(xs):
    m = max(xs)
    if m <= -1e29:
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _lognorm(w, var):
    return -0.5 * w * w / var - 0.5 * math.log(2 * math.pi * var)


def run_chain(ds, theta0, seed, n_sweeps, burn, thin=10, beta=1.0, init=None, record=True, init_from_prior=False, **sweep_kw):
    rng = np.random.default_rng(seed)
    s = Sampler(ds, theta0.copy(), rng, beta=beta, init=init, init_from_prior=init_from_prior)
    out = []
    for i in range(n_sweeps):
        s.sweep(**sweep_kw)
        if record and i >= burn and (i - burn) % thin == 0:
            out.append(s.snapshot())
    return s, out
