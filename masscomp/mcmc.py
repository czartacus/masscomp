"""MCMC over (y, event, donor, theta) (plan §5), with optional tempering of the likelihood.

Deviations from the plan, recorded here:
  * rho_c and omega are updated by random-walk MH on the log scale, not Gibbs: the single-event
    hazard probabilities are not conjugate to the Gamma prior.
  * t_root is updated by random-walk MH on the log scale, not slice sampling.
"""
import math
import itertools
import numpy as np
from .segments import N_SEG, EPS
from .transducer import EdgeParams
from . import events as ev
from .likelihood import State
from .tree import with_root_length

MAX_FORM_LEN = int(__import__("os").environ.get("MASSCOMP_MAX_FORM_LEN", "12"))   # sampler support for ancestral forms; 8 until 2026-11-04 (2% of
                                                                                     # prior-simulated ancestors were longer, 24% on some datasets)


class Sampler:
    def __init__(self, ds, theta, rng, beta=1.0, init=None, init_from_prior=False):
        self._fixed = None                     # per-sweep option tables, see _fixed_options
        self.rng = rng
        self.beta = beta
        self.bp = 1.0        # temperature on the prior-side terms (edge priors, events, root forms); 1 for
                             # inference and for the stepping stone, = beta in full-tempering PT (mixing only)
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
        self._cs = None      # concept restriction for the latent moves (move_tempered)
        self.acc = {k: [0, 0] for k in ["y", "gibbs", "ffbs", "jffbs", "regrow", "temper", "joint", "lam", "u", "rho", "omega", "troot"]}
        self.inlattice = 1.0
        if init is None and not init_from_prior and self.INIT == "lattice":
            self._init_lattice()
        if init is None:
            self.move_events()   # remove impossible inherit terms from the initial state

    def _init_lattice(self):
        """Leaf-anchored start: every internal form drawn bottom-up from the innovation prior times the
        transducers of its (up to two) inheriting children, i.e. the regrow proposal applied to the
        whole tree with all events INHERIT. Unlike the child-copy start it does not commit each
        ancestor to one lineage, which is the basin the copy start then has to escape (omega 4-7x
        the truth on seed 7 at depth 0.25, with the other children explained as borrowings)."""
        from .ffbs import draw as ffbs_draw
        st, f, rng = self.st, self.st.f, self.rng
        for c in range(st.M):
            for v in reversed(f.topo):
                if st.is_leaf[v]:
                    continue
                lat = [(st.ep[ch], st.y[(ch, c)]) for ch in f.children[v]][:2]
                y, _ = ffbs_draw(rng, 0, None, None, [(ep.log_sub_np, ep.log_ins_np, z) for ep, z in lat])
                if y and len(y) <= MAX_FORM_LEN:
                    st.y[(v, c)] = y
        st.recompute_all()

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
            d += self.bp * (pr - st.t_prior[(e, c)]) + self.beta * (lk - st.t_lik[(e, c)])
        return d, new

    def _commit(self, new):
        for k, (pr, lk) in new.items():
            self.st.t_prior[k] = pr
            self.st.t_lik[k] = lk

    def _concepts(self):
        return self._cs if self._cs is not None else range(self.st.M)

    # ------------------------------------------------------------------ move 1: forms
    def move_forms(self, n_prop=5):
        self._fixed = None                     # the forms may change: drop the per-sweep option tables
        st, f, rng = self.st, self.st.f, self.rng
        for v in f.internal:
            for c in self._concepts():
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
                        droot = self.bp * (ev.log_G(prop) - st.t_prior[("root", v, c)])
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
            top, s_del = (st.ep[ch] if ch in st.ep else st.adapt).top_parents
            scored = []
            for i, o in enumerate(x):
                for s_, lp in top[o]:                # the parent segments the edge most favours for o
                    scored.append((lp, x[:i] + (s_,) + x[i + 1:]))
            scored.sort(key=lambda t: -t[0])
            for _, cand in scored[:max_per_child]:
                S.add(cand)
            if len(x) > 1:
                for i in range(len(x)):               # child inserted a segment: parent lacks it
                    S.add(x[:i] + x[i + 1:])
            if len(x) < MAX_FORM_LEN:                 # child deleted a segment: parent had one more
                for i in range(len(x) + 1):
                    S.add(x[:i] + (s_del,) + x[i:])
        S = {s for s in S if 0 < len(s) <= MAX_FORM_LEN}
        return S

    def move_forms_gibbs(self):
        """Exact Gibbs draw of y[v,c] over candidates(v,c) when the current form is in that set;
        otherwise no-op (the MH moves handle it). Returns fraction of (v,c) that were in-lattice."""
        self._fixed = None                     # the forms may change: drop the per-sweep option tables
        st, f, rng = self.st, self.st.f, self.rng
        inlat = 0; tot = 0
        for v in f.internal:
            is_root = f.parent[v] < 0
            for c in self._concepts():
                cur = st.y[(v, c)]
                S = self.candidates(v, c)
                tot += 1
                if cur not in S:
                    continue
                inlat += 1
                cands = list(S)
                edges = st.affected_edges(v, c)
                # per-edge evaluators of (prior_part, lik_part) as a function of the candidate form; the
                # same arithmetic as edge_terms, with the state lookups hoisted out of the candidate loop
                evals = []
                for e in edges:
                    k = st.event[(e, c)]; lp_ev = st.evp[e][c][k]; leaf = st.is_leaf[e]
                    if e == v:                                     # incoming edge: cand is the child form
                        if k == ev.INHERIT:
                            yp = st.y[(f.parent[v], c)]; lpf = st.ep[v].logprob
                            evals.append((e, lambda cand, lpf=lpf, yp=yp, a=lp_ev: (a + lpf(yp, cand), 0.0)))
                        elif k == ev.REPLACE:
                            evals.append((e, lambda cand, a=lp_ev: (a + ev.log_G(cand), 0.0)))
                        else:
                            yd = st.y[(st.donor[(v, c)], c)]; a = lp_ev - math.log(len(st.cands[v])); lpf = st.adapt.logprob
                            evals.append((e, lambda cand, lpf=lpf, yd=yd, a=a: (a + lpf(yd, cand), 0.0)))
                    else:                                          # outgoing: cand is the parent (or donor) form
                        z = st.y[(e, c)]
                        if k == ev.INHERIT and f.parent[e] == v:
                            lpf = st.ep[e].logprob
                        elif k == ev.BORROW and st.donor.get((e, c)) == v:
                            lpf = st.adapt.logprob; lp_ev = lp_ev - math.log(len(st.cands[e]))
                        else:                                      # does not depend on cand
                            pr, lk = st.edge_terms(e, c)
                            evals.append((e, lambda cand, pr=pr, lk=lk: (pr, lk)))
                            continue
                        if leaf:
                            evals.append((e, lambda cand, lpf=lpf, z=z, a=lp_ev: (a, lpf(cand, z))))
                        else:
                            evals.append((e, lambda cand, lpf=lpf, z=z, a=lp_ev: (a + lpf(cand, z), 0.0)))
                bp, beta = self.bp, self.beta
                logw = np.empty(len(cands)); terms = []
                for i, cand in enumerate(cands):
                    w = bp * ev.log_G(cand) if is_root else 0.0
                    tt = {}
                    for e, fn in evals:
                        pr, lk = fn(cand)
                        tt[(e, c)] = (pr, lk); w += bp * pr + beta * lk
                    logw[i] = w; terms.append(tt)
                p = np.exp(logw - logw.max()); cum = np.cumsum(p)
                i = min(int(np.searchsorted(cum, rng.random() * cum[-1])), len(cands) - 1)
                st.y[(v, c)] = cands[i]
                self._commit(terms[i])
                if is_root:
                    st.t_prior[("root", v, c)] = ev.log_G(cands[i])
                self.acc["gibbs"][0] += cands[i] != cur; self.acc["gibbs"][1] += 1
        return inlat / tot if tot else 1.0

    # ------------------------------------------------------------------ move 1c: lattice FFBS
    def move_forms_ffbs(self, p_select=0.25):
        """Independence proposal for y[v,c] drawn EXACTLY from P_in(y) * prod_k T_k(z_k | y) over the
        alignment lattice (ffbs.py): the incoming edge's conditional (transducer from the parent or
        donor form, or the innovation prior) times up to two outgoing transducers (children inheriting
        from v, edges borrowing from v). At beta = 1 with at most two out-edges this is the exact
        conditional, so the move is a Gibbs step; otherwise the MH ratio corrects for the tempering
        of leaf-edge terms and for out-edges beyond the two in the lattice. Each (v, c) is visited with
        probability p_select per sweep (a draw costs a few ms). Floored (impossible) transducer values
        make q slightly inexact for states the lattice cannot reach; such states are left with
        probability ~e^-40 and never proposed, so the stationary distribution is unaffected in practice."""
        self._fixed = None                     # the forms may change: drop the per-sweep option tables
        rng = self.rng
        for v in self.st.f.internal:
            for c in self._concepts():
                if rng.random() < p_select:
                    self._ffbs_site(v, c)

    def _ffbs_site(self, v, c):
        from .ffbs import draw as ffbs_draw
        st, f, rng = self.st, self.st.f, self.rng
        is_root = f.parent[v] < 0
        cur = st.y[(v, c)]
        in_mode, tab, x, lat = self._ffbs_setup(v, c)
        prop, _ = ffbs_draw(rng, in_mode, (tab.log_sub_np, tab.log_ins_np) if tab is not None else None, x,
                            [(ep.log_sub_np, ep.log_ins_np, z) for ep, z in lat])
        if not prop or len(prop) > MAX_FORM_LEN or prop == cur:
            return
        logq = self._ffbs_score(in_mode, tab, x, lat, cur) - self._ffbs_score(in_mode, tab, x, lat, prop)
        edges = st.affected_edges(v, c)
        over = {}
        for e in edges:
            if e == v:
                over[(e, c)] = dict(y_child=prop)
            elif f.parent[e] == v:
                over[(e, c)] = dict(y_parent=prop)
            else:
                over[(e, c)] = {}
        st.y[(v, c)] = prop
        dterm, new = self._delta_terms([(e, c) for e in edges], over)
        droot = self.bp * (ev.log_G(prop) - st.t_prior[("root", v, c)]) if is_root else 0.0
        self.acc["ffbs"][1] += 1
        if self._accept(dterm + droot + logq):
            self._commit(new)
            if is_root:
                st.t_prior[("root", v, c)] = ev.log_G(prop)
            self.acc["ffbs"][0] += 1
        else:
            st.y[(v, c)] = cur

    def _ffbs_setup(self, v, c, kind=None, d=None):
        """(in_mode, tables, x, lattice outs, all outs) for the FFBS lattice of (v, c) under the given
        incoming event kind (default: the current one)."""
        st, f = self.st, self.st.f
        if kind is None:
            kind = st.event[(v, c)] if f.parent[v] >= 0 else ev.REPLACE
            d = st.donor.get((v, c))
        if kind == ev.INHERIT:
            in_mode, tab, x = 1, st.ep[v], st.y[(f.parent[v], c)]
        elif kind == ev.REPLACE:
            in_mode, tab, x = 0, None, None
        else:
            in_mode, tab, x = 1, st.adapt, st.y[(d, c)]
        outs = []
        for ch in f.children[v]:
            if st.event[(ch, c)] == ev.INHERIT:
                outs.append((st.ep[ch], st.y[(ch, c)]))
        for e in f.edges:
            if st.event.get((e, c)) == ev.BORROW and st.donor.get((e, c)) == v:
                outs.append((st.adapt, st.y[(e, c)]))
        return in_mode, tab, x, outs[:2]

    @staticmethod
    def _ffbs_score(in_mode, tab, x, lat, y):
        sc = ev.log_G(y) if in_mode == 0 else tab.logprob(x, y)
        for ep, z in lat:
            sc += ep.logprob(y, z)
        return sc

    def move_joint_ffbs(self, p_select=0.25):
        """Metropolised Gibbs over the incoming event of (v, c) — inherit / replace / borrow from each
        candidate donor — with the form integrated out over the lattice and then drawn exactly:
        propose a kind uniformly, compute the lattice mass Z_kind (ffbs.log_Z) for it and for the
        current kind, draw y' from the proposed lattice, and accept with
            [target(kind', y') / target(kind, y)] / [q(kind', y') / q(kind, y)],
        q(kind, y) = (1/n_opt) * score_kind(y) / Z_kind. At beta = 1 with at most two out-edges the
        ratio reduces to evp[kind'] Z_kind' / (evp[kind] Z_kind): the exact collapsed conditional.
        This is what lets borrow/inherit labels move when the form they imply differs from the
        current one (move_joint proposes forms from the generative side only, blind to the children)."""
        self._fixed = None                     # the forms may change: drop the per-sweep option tables
        rng = self.rng
        for v in self.st.f.internal:
            if self.st.f.parent[v] < 0:
                continue
            for c in self._concepts():
                if rng.random() < p_select:
                    self._jffbs_site(v, c)

    def _jffbs_site(self, v, c):
        from .ffbs import draw as ffbs_draw, log_Z as ffbs_logZ
        st, f, rng = self.st, self.st.f, self.rng

        def tables(tab):
            return (tab.log_sub_np, tab.log_ins_np) if tab is not None else None

        cands = st.cands[v]
        n_opt = 2 + len(cands)
        cur, cur_kind, cur_d = st.y[(v, c)], st.event[(v, c)], st.donor.get((v, c))
        i = int(rng.integers(n_opt))
        kind, d = (i, None) if i < 2 else (ev.BORROW, cands[i - 2])
        if kind == cur_kind and d == cur_d:
            return
        m_new, tab_new, x_new, lat = self._ffbs_setup(v, c, kind, d)
        m_cur, tab_cur, x_cur, _ = self._ffbs_setup(v, c)
        lat_t = [(ep.log_sub_np, ep.log_ins_np, z) for ep, z in lat]
        prop, logZ_new = ffbs_draw(rng, m_new, tables(tab_new), x_new, lat_t)
        if not prop or len(prop) > MAX_FORM_LEN:
            return
        logZ_cur = ffbs_logZ(m_cur, tables(tab_cur), x_cur, lat_t)
        logq_new = self._ffbs_score(m_new, tab_new, x_new, lat, prop) - logZ_new
        logq_cur = self._ffbs_score(m_cur, tab_cur, x_cur, lat, cur) - logZ_cur
        edges = st.affected_edges(v, c)
        over = {(v, c): dict(y_child=prop, kind=kind, d=d)}
        for e in edges:
            if e != v:
                over[(e, c)] = dict(y_parent=prop) if f.parent[e] == v else {}
        st.y[(v, c)] = prop
        dterm, new = self._delta_terms([(e, c) for e in edges], over)
        self.acc["jffbs"][1] += 1
        if self._accept(dterm + logq_cur - logq_new):
            self._commit(new)
            st.event[(v, c)] = kind
            if kind == ev.BORROW:
                st.donor[(v, c)] = d
            else:
                st.donor.pop((v, c), None)
            self.acc["jffbs"][0] += 1
        else:
            st.y[(v, c)] = cur

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
        self._fixed = None                     # the forms may change: drop the per-sweep option tables
        st, f, rng = self.st, self.st.f, self.rng
        for v in f.internal:
            if f.parent[v] < 0:
                continue
            cands = st.cands[v]
            n_opt = 2 + len(cands)
            for _ in range(n_prop):
                for c in self._concepts():
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
        self._fixed = None                     # the forms may change: drop the per-sweep option tables
        for e in self.st.f.edges:
            for c in self._concepts():
                self._event_site(e, c)

    def _event_site(self, e, c):
        """Gibbs draw of the event (and donor) on edge e, concept c, given all forms."""
        st, rng = self.st, self.rng
        opts, logw = [], []
        for k in (ev.INHERIT, ev.REPLACE):
            pr, lk = st.edge_terms(e, c, kind=k)
            opts.append((k, None, pr, lk)); logw.append(self.bp * pr + self.beta * lk)
        for d in st.cands[e]:
            pr, lk = st.edge_terms(e, c, kind=ev.BORROW, d=d)
            opts.append((ev.BORROW, d, pr, lk)); logw.append(self.bp * pr + self.beta * lk)
        w = np.array(logw); w = np.exp(w - w.max()); w /= w.sum()
        k, d, pr, lk = opts[int(rng.choice(len(opts), p=w))]
        st.event[(e, c)] = k
        if k == ev.BORROW:
            st.donor[(e, c)] = d
        else:
            st.donor.pop((e, c), None)
        st.t_prior[(e, c)] = pr; st.t_lik[(e, c)] = lk

    # ------------------------------------------------------------------ move 1c: subtree regrow
    def move_regrow(self, p_select=0.05):
        """Independence proposal for ALL internal forms of a subtree of concept c, grown bottom-up: each
        node is drawn exactly from P_in(y) * prod T(child | y) with its (already proposed) inheriting
        children as outputs and, for the subtree root with an INHERIT/BORROW incoming event, the parent
        or donor form as input (the innovation prior otherwise). Nodes below the subtree root cannot see
        their parent (it is proposed after them), so the proposal is leaf-anchored rather than
        conditional; the MH ratio uses the full proposal density (per-node score minus the lattice
        normaliser, both states). Motivation (basin_swap, 3 Nov 2026): under the chain's own theta,
        splicing the truth's latents in concept by concept improves the posterior for 16 of 25 concepts
        by 8-52 nats, yet the single-node and tempered moves never get there - the modes are separated
        by paths through bad intermediate states. A whole-subtree draw can land in the other mode."""
        self._fixed = None                     # the forms may change: drop the per-sweep option tables
        rng = self.rng
        for w in self.st.f.internal:
            for c in self._concepts():
                if rng.random() < p_select:
                    self._regrow_site(w, c)

    def move_root_regrow(self, p_select=1.0):
        """The regrow move restricted to the root node(s) of the forest: the root form is redrawn from the
        innovation prior times its (up to two) inheriting children's transducers and the events on the
        root edges are re-decided (inherit / replace / borrow), the rest of the tree fixed. Targets the
        basin the copy start leaves behind — the joined root a copy of one family's form, the other
        family 'replaced' on its root edge and t_root pushed up to make that cheap (seed 1 at depth
        0.25: copy-start t_root 1.2-1.7 against 0.26-0.30 from the truth, 146 nats apart)."""
        self._fixed = None
        rng = self.rng
        for w in self.st.f.roots:
            if self.st.is_leaf[w]:
                continue
            for c in self._concepts():
                if rng.random() < p_select:
                    self._regrow_site(w, c, max_depth=0)

    def _regrow_site(self, w, c, max_depth=None):
        from .ffbs import draw as ffbs_draw, log_Z as ffbs_logZ
        st, f, rng = self.st, self.st.f, self.rng
        leaf = st.is_leaf
        sub = [v for v in f.topo if not leaf[v] and (v == w or self._below(v, w, max_depth))]
        subset = set(sub)
        cur = {v: st.y[(v, c)] for v in sub}

        def inputs(v):
            if v == w and f.parent[v] >= 0:
                k = st.event[(v, c)]
                if k == ev.INHERIT:
                    return 1, st.ep[v], st.y[(f.parent[v], c)]
                if k == ev.BORROW and st.donor[(v, c)] not in subset:
                    return 1, st.adapt, st.y[(st.donor[(v, c)], c)]
            return 0, None, None

        # child edges whose event is re-decided along with the parent form: inherit, replace, or (with
        # REGROW_BORROW) borrow from each candidate donor outside the subtree. Edges currently borrowing
        # from a donor inside the subtree keep their event (its form is being proposed too). At most two
        # edges enter the lattice as outputs.
        def is_free(ch):
            k = st.event[(ch, c)]
            if k == ev.BORROW:
                return self.REGROW_BORROW and st.donor[(ch, c)] not in subset
            return True
        free = {v: [ch for ch in f.children[v] if is_free(ch)][:2] for v in sub} if self.REGROW_EVENTS \
            else {v: [] for v in sub}
        cur_ev = {ch: (st.event[(ch, c)], st.donor.get((ch, c))) for v in sub for ch in free[v]}
        options = {ch: [(ev.INHERIT, None), (ev.REPLACE, None)] +
                       ([(ev.BORROW, d) for d in st.cands[ch] if d not in subset] if self.REGROW_BORROW else [])
                   for v in sub for ch in free[v]}

        def form_of(ch, forms):
            return forms[ch] if ch in forms else st.y[(ch, c)]

        def outs(v, forms, events):
            """Lattice outputs of v: inheriting children (re-decided ones per `events`, fixed ones as they are)."""
            fr = free[v]
            lat = [(st.ep[ch], form_of(ch, forms)) for ch in fr if events[ch][0] == ev.INHERIT]
            lat += [(st.ep[ch], form_of(ch, forms)) for ch in f.children[v] if ch not in fr and st.event[(ch, c)] == ev.INHERIT]
            return lat[:2]

        def configs(v, forms):
            """(events dict, log weight) for each assignment of v's free child edges: the event prior, the
            innovation / adaptation density of replaced / borrowed children, and the lattice mass of the
            inheriting ones."""
            in_mode, tab, x = inputs(v)
            tabs = (tab.log_sub_np, tab.log_ins_np) if tab is not None else None
            fr = free[v]
            out = []
            lz_cache = {}                     # the lattice mass depends only on which free children inherit
            for combo in itertools.product(*[options[ch] for ch in fr]):
                events = dict(zip(fr, combo))
                lw = 0.0
                for ch in fr:
                    k, d = events[ch]
                    lw += st.evp[ch][c][k]
                    if k == ev.REPLACE:
                        lw += ev.log_G(form_of(ch, forms))
                    elif k == ev.BORROW:
                        lw += -math.log(len(st.cands[ch])) + st.adapt.logprob(st.y[(d, c)], form_of(ch, forms))
                key = tuple(k == ev.INHERIT for k, _ in combo)
                if key not in lz_cache:
                    lat = outs(v, forms, events)
                    lz_cache[key] = (ffbs_logZ(in_mode, tabs, x, [(ep.log_sub_np, ep.log_ins_np, z) for ep, z in lat]), lat)
                lz, lat = lz_cache[key]
                out.append((events, lw + lz, lz, lat))
            return in_mode, tab, x, out

        def logq(forms, events):
            q = 0.0
            for v in reversed(sub):
                in_mode, tab, x, cfgs = configs(v, forms)
                ws = np.array([lw for _, lw, _, _ in cfgs])
                tot = _lse(ws)
                for evs, lw, lz, lat in cfgs:
                    if all(events[ch] == evs[ch] for ch in free[v]):
                        q += lw - tot + self._ffbs_score(in_mode, tab, x, lat, forms[v]) - lz
                        break
            return q

        prop, prop_ev = {}, {}
        for v in reversed(sub):
            in_mode, tab, x, cfgs = configs(v, prop)
            ws = np.array([lw for _, lw, _, _ in cfgs])
            p = np.exp(ws - ws.max()); p /= p.sum()
            evs, lw, lz, lat = cfgs[int(rng.choice(len(cfgs), p=p))]
            prop_ev.update(evs)
            y, _ = ffbs_draw(rng, in_mode, (tab.log_sub_np, tab.log_ins_np) if tab is not None else None, x,
                             [(ep.log_sub_np, ep.log_ins_np, z) for ep, z in lat])
            if not y or len(y) > MAX_FORM_LEN:
                return
            prop[v] = y
        if all(prop[v] == cur[v] for v in sub) and all(prop_ev[ch] == cur_ev[ch] for ch in cur_ev):
            return
        lq = logq(cur, cur_ev) - logq(prop, prop_ev)
        edges = set()
        for v in sub:
            edges.update(st.affected_edges(v, c))
        for v in sub:
            st.y[(v, c)] = prop[v]
        over = {(e, c): (dict(kind=prop_ev[e][0], d=prop_ev[e][1]) if e in prop_ev else {}) for e in edges}
        dterm, new = self._delta_terms([(e, c) for e in edges], over)
        droot = 0.0
        if f.parent[w] < 0:
            droot = self.bp * (ev.log_G(prop[w]) - st.t_prior[("root", w, c)])
        self.acc["regrow"][1] += 1
        if self._accept(dterm + droot + lq):
            self._commit(new)
            for ch, (k, d) in prop_ev.items():
                st.event[(ch, c)] = k
                if k == ev.BORROW:
                    st.donor[(ch, c)] = d
                else:
                    st.donor.pop((ch, c), None)
            if f.parent[w] < 0:
                st.t_prior[("root", w, c)] = ev.log_G(prop[w])
            self.acc["regrow"][0] += 1
        else:
            for v in sub:
                st.y[(v, c)] = cur[v]

    def _below(self, v, w, max_depth=None):
        """Is v a proper descendant of w (within max_depth levels, if given)?"""
        p = self.st.f.parent[v]; d = 1
        while p >= 0:
            if p == w:
                return max_depth is None or d <= max_depth
            p = self.st.f.parent[p]; d += 1
        return False

    # ------------------------------------------------------------------ move 1d: tempered transitions
    def move_tempered(self, concepts, betas=(0.6, 0.35, 0.2, 0.1, 0.05), sweeps_per=1):
        """Neal's (1996) tempered transitions, one concept at a time. Concepts are independent given
        the parameters, so heating only concept c's leaf-likelihood terms (self.beta while moving only
        c's latents) is a tempering of c's conditional. Up the ladder beta_1 > ... > beta_n and back,
        with a reversible kernel at each level (site moves in forward order going up, reverse order
        coming down), the excursion is accepted with
            sum_i (beta_i - beta_{i-1}) ll(x_{i-1})  +  sum_i (beta_{i-1} - beta_i) ll(x^_{i-1})
        where ll is c's leaf log-likelihood. Lets a whole path of ancestral forms reorganise under a
        weak likelihood, which single-site moves cannot do (root_diag, 3 Nov 2026: three starts, three
        basins, 100-170 nats apart in posterior, none left in 800-1500 sweeps)."""
        self._fixed = None                     # the forms may change: drop the per-sweep option tables
        st, f = self.st, self.st.f
        beta0 = self.beta
        ladder = [beta0] + [beta0 * b for b in betas]
        n = len(ladder) - 1
        internal = list(f.internal)
        nonroot = [v for v in internal if f.parent[v] >= 0]
        edges = list(f.edges)
        for c in concepts:
            keys = [(e, c) for e in edges if st.is_leaf[e]]

            def llc():
                return sum(st.t_lik[k] for k in keys)

            def kernel(reverse):
                vs = internal[::-1] if reverse else internal
                es = edges[::-1] if reverse else edges
                if reverse:
                    for e in es:
                        self._event_site(e, c)
                    for v in vs:
                        if f.parent[v] >= 0:
                            self._jffbs_site(v, c)
                        self._ffbs_site(v, c)
                else:
                    for v in vs:
                        self._ffbs_site(v, c)
                        if f.parent[v] >= 0:
                            self._jffbs_site(v, c)
                    for e in es:
                        self._event_site(e, c)

            saved = (dict((v, st.y[(v, c)]) for v in internal), dict((e, st.event[(e, c)]) for e in edges),
                     dict((e, st.donor[(e, c)]) for e in edges if (e, c) in st.donor),
                     dict((e, (st.t_prior[(e, c)], st.t_lik[(e, c)])) for e in edges),
                     dict((r, st.t_prior[("root", r, c)]) for r in f.roots))
            logr = 0.0
            for i in range(1, n + 1):
                logr += (ladder[i] - ladder[i - 1]) * llc()
                self.beta = ladder[i]
                for _ in range(sweeps_per):
                    kernel(False)
            for i in range(n, 0, -1):
                self.beta = ladder[i]
                for _ in range(sweeps_per):
                    kernel(True)
                logr += (ladder[i - 1] - ladder[i]) * llc()
            self.beta = beta0
            self.acc["temper"][1] += 1
            if self._accept(logr):
                self.acc["temper"][0] += 1
            else:
                ys, evs, dns, terms, roots = saved
                for v in internal:
                    st.y[(v, c)] = ys[v]
                for e in edges:
                    st.event[(e, c)] = evs[e]
                    st.donor.pop((e, c), None)
                    st.t_prior[(e, c)], st.t_lik[(e, c)] = terms[e]
                for e, d in dns.items():
                    st.donor[(e, c)] = d
                for r in f.roots:
                    st.t_prior[("root", r, c)] = roots[r]

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
                d += self.bp * (pr - st.t_prior[(e, c)]) + self.beta * (lk - st.t_lik[(e, c)])
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

    def _u_slot_logq(self, slot, present, counts):
        """log q(slot) for a birth over the slots absent from `present`: mixture of uniform and
        count-weighted."""
        n_abs = ev.N_U_SLOTS - len(present)
        tot = sum(counts.values()) - sum(counts.get(sl, 0) for sl in present)
        if tot == 0:
            # the data-informed branch falls back to a uniform draw when there are no mismatches,
            # so the whole mixture is uniform (the old (1 - FRAC)/n understated q by log 2 and
            # accepted births too often; Geweke test, 3 Nov 2026)
            return -math.log(n_abs)
        q = (1 - self.U_DATA_FRAC) / n_abs + self.U_DATA_FRAC * counts.get(slot, 0) / tot
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
        same densities in reverse.

        Cost: the replace/borrow options do not depend on u_e and are taken from the per-sweep option
        tables; a proposal on a substitution/deletion slot (s, o) changes the inherit term only for
        concepts whose parent form contains s (EdgeParams.derive changes row s alone), so only those
        are recomputed."""
        st, f, rng = self.st, self.st.f, self.rng
        M = st.M
        fx = self._fixed_options()                           # replace + borrow options, (M, E, nf) and kinds
        for j, e in enumerate(f.edges):
            t = f.length[e]
            counts = self._u_counts(e)
            nf = 1 + len(st.cands[e])
            Rf, Lf, Kf, donors_f = fx["R"][:, j, :nf], fx["L"][:, j, :nf], fx["K"][j, :nf], fx["donors"][j][:nf]
            evp_e = np.array(st.evp[e])                          # (M, 3)
            wf = self.bp * (Rf + evp_e[:, Kf]) + self.beta * Lf  # fixed option weights, (M, nf)
            mf = wf.max(axis=1); fixed_lse = mf + np.log(np.exp(wf - mf[:, None]).sum(axis=1))
            inh = np.array([st.edge_terms(e, c, kind=ev.INHERIT) for c in range(M)])    # (M, 2): pr, lk
            inh_w = self.bp * inh[:, 0] + self.beta * inh[:, 1]
            logZ = np.logaddexp(fixed_lse, inh_w)
            yp = [st.y[(f.parent[e], c)] for c in range(M)]
            contains = {}
            for _ in range(n_prop):
                u = st.theta.u.setdefault(e, {})
                present = list(u.keys())
                kind = rng.integers(3)
                prop = dict(u)
                logq = 0.0
                if kind == 0:                                  # birth
                    if len(u) >= ev.N_U_SLOTS:
                        continue
                    if rng.random() < self.U_DATA_FRAC:
                        tot = sum(counts.values()) - sum(counts.get(sl, 0) for sl in u)
                        if tot > 0:
                            absent = [sl for sl in counts if sl not in u]
                            wts = np.array([counts[a] for a in absent], dtype=float) / tot
                            slot = absent[int(rng.choice(len(absent), p=wts))]
                        else:
                            slot = _draw_absent(rng, u)
                    else:
                        slot = _draw_absent(rng, u)
                    if rng.random() < self.U_DATA_FRAC:
                        w = float(rng.normal(self.U_DATA_MU, self.U_DATA_SD))
                    else:
                        w = float(rng.normal(0, math.sqrt(t * ev.SIGMA2_U)))
                    prop[slot] = w; delta = w
                    logq = -math.log(len(present) + 1) - (self._u_slot_logq(slot, u, counts) + self._u_w_logq(w, t))
                elif kind == 1:                                # death
                    if not present:
                        continue
                    slot = present[int(rng.integers(len(present)))]
                    w = prop.pop(slot); delta = -w
                    logq = (self._u_slot_logq(slot, prop, counts) + self._u_w_logq(w, t)) - (-math.log(len(present)))
                else:                                          # perturb
                    if not present:
                        continue
                    slot = present[int(rng.integers(len(present)))]
                    delta = float(rng.normal(0, perturb_sd))
                    prop[slot] = u[slot] + delta
                dprior = ev.log_prior_u(prop, t) - ev.log_prior_u(u, t)
                old_ep = st.ep[e]
                st.ep[e] = old_ep.derive(slot, delta)
                if slot[0] == "ins":
                    affected = range(M)
                else:
                    s_ = slot[0]
                    if s_ not in contains:
                        contains[s_] = [c for c in range(M) if s_ in yp[c]]
                    affected = contains[s_]
                inh_new = inh.copy()
                for c in affected:
                    inh_new[c] = st.edge_terms(e, c, kind=ev.INHERIT)
                inh_w_new = self.bp * inh_new[:, 0] + self.beta * inh_new[:, 1]
                logZ_new = np.logaddexp(fixed_lse, inh_w_new)
                d = dprior + logq + float((logZ_new - logZ).sum())
                self.acc["u"][1] += 1
                if self._accept(d):
                    st.theta.u[e] = prop
                    st.t_param += dprior
                    self.acc["u"][0] += 1
                    inh, inh_w, logZ = inh_new, inh_w_new, logZ_new
                    # events on edge e from their conditional under the new u_e
                    wall = np.concatenate([wf, inh_w[:, None]], axis=1)          # options: fixed..., inherit
                    wall = np.exp(wall - wall.max(axis=1, keepdims=True)); cum = np.cumsum(wall, axis=1)
                    uu = rng.random(M) * cum[:, -1]
                    pick = (cum < uu[:, None]).sum(axis=1)
                    for c in range(M):
                        q = min(int(pick[c]), nf)
                        if q == nf:
                            k, dn, pr, lk = ev.INHERIT, None, inh[c, 0], inh[c, 1]
                        else:
                            k, dn = int(Kf[q]), donors_f[q]; pr, lk = Rf[c, q] + evp_e[c, k], Lf[c, q]
                        st.event[(e, c)] = k
                        if k == ev.BORROW:
                            st.donor[(e, c)] = dn
                        else:
                            st.donor.pop((e, c), None)
                        st.t_prior[(e, c)] = float(pr); st.t_lik[(e, c)] = float(lk)
                else:
                    st.ep[e] = old_ep

    # ------------------------------------------------------------------ move 5/6: rates
    def _fixed_options(self):
        """Per-sweep tables of the event options that do not depend on u_e or the rates: replace and
        borrow-from-each-donor, as (R (M, E, nf) = prior part minus evp, L (M, E, nf) = lik part,
        K (E, nf) kinds, donors). Padded with -inf. Built on first use after the latent moves of a
        sweep (they change the forms) and shared by move_u and move_rates."""
        if self._fixed is not None:
            return self._fixed
        st, f = self.st, self.st.f
        M, E = st.M, len(f.edges)
        nf = max(1 + len(st.cands[e]) for e in f.edges)
        R = np.full((M, E, nf), -np.inf); L = np.zeros((M, E, nf)); K = np.zeros((E, nf), dtype=np.int64); donors = []
        for j, e in enumerate(f.edges):
            kinds = [ev.REPLACE] + [ev.BORROW] * len(st.cands[e]); dn = [None] + list(st.cands[e]); donors.append(dn)
            K[j, :len(kinds)] = kinds
            for c in range(M):
                evp = st.evp[e][c]
                for q, (k, d) in enumerate(zip(kinds, dn)):
                    pr, lk = st.edge_terms(e, c, kind=k, d=d)
                    R[c, j, q] = pr - evp[k]; L[c, j, q] = lk
        self._fixed = dict(R=R, L=L, K=K, donors=donors)
        return self._fixed

    def _event_remainders(self, cs):
        """For each edge e and concept c in cs: the event options (kind, donor, pr_minus_evp, lk), i.e. the
        edge terms with the rate-dependent event log-probability removed. These depend on the forms
        and u_e only, so a rate proposal changes just the evp part and the per-(e, c) sum over
        options can be recomputed cheaply. Options are in the same order for every concept (inherit,
        then replace and one borrow per candidate donor from the per-sweep fixed tables) and padded
        to the widest edge with -inf, so the normaliser is one vectorised log-sum-exp over
        (concept, edge, option)."""
        st, f = self.st, self.st.f
        cs = list(cs); E = len(f.edges)
        fx = self._fixed_options()
        nopt = 1 + fx["R"].shape[2]
        R = np.full((len(cs), E, nopt), -np.inf); L = np.zeros((len(cs), E, nopt))
        K = np.zeros((E, nopt), dtype=np.int64); donors = []
        R[:, :, 1:] = fx["R"][cs]; L[:, :, 1:] = fx["L"][cs]; K[:, 1:] = fx["K"]; K[:, 0] = ev.INHERIT
        for j, e in enumerate(f.edges):
            donors.append([None] + fx["donors"][j])
            for i, c in enumerate(cs):
                pr, lk = st.edge_terms(e, c, kind=ev.INHERIT)
                R[i, j, 0] = pr - st.evp[e][c][ev.INHERIT]; L[i, j, 0] = lk
        return dict(cs=cs, R=R, L=L, K=K, donors=donors)

    def _evp_tensor(self, cs):
        """(|cs|, n_edges, 3) array of the current event log-probabilities."""
        st, f = self.st, self.st.f
        return np.array([[st.evp[e][c] for e in f.edges] for c in cs])

    def _rem_weights(self, rem):
        R, L, K = rem["R"], rem["L"], rem["K"]
        evp = self._evp_tensor(rem["cs"])                                    # (|cs|, E, 3)
        evk = np.take_along_axis(evp, np.broadcast_to(K[None], (len(rem["cs"]),) + K.shape), axis=2)
        return self.bp * (R + evk) + self.beta * L                             # -inf at the padding

    def _logZ_sum(self, rem):
        """Sum over (e, c) of log sum over event options of exp(bp * (r + evp) + beta * lk) under the current evp."""
        w = self._rem_weights(rem)
        m = w.max(axis=2)
        return float(np.sum(m + np.log(np.exp(w - m[..., None]).sum(axis=2))))

    def _resample_events(self, rem):
        """Gibbs draw of the events in rem from their conditional under the current evp."""
        st, f, rng = self.st, self.st.f, self.rng
        w = self._rem_weights(rem)
        w = np.exp(w - w.max(axis=2, keepdims=True)); cum = np.cumsum(w, axis=2)
        u = rng.random(cum.shape[:2]) * cum[:, :, -1]
        pick = (cum < u[..., None]).sum(axis=2)
        K, R, L = rem["K"], rem["R"], rem["L"]
        for i, c in enumerate(rem["cs"]):
            for j, e in enumerate(f.edges):
                q = min(int(pick[i, j]), K.shape[1] - 1)
                k = int(K[j, q])
                st.event[(e, c)] = k
                if k == ev.BORROW:
                    st.donor[(e, c)] = rem["donors"][j][q]
                else:
                    st.donor.pop((e, c), None)
                st.t_prior[(e, c)] = R[i, j, q] + st.evp[e][c][k]; st.t_lik[(e, c)] = L[i, j, q]

    def _rem_sub(self, rem, i):
        """The remainder tables restricted to the i-th concept of rem."""
        return dict(cs=[rem["cs"][i]], R=rem["R"][i:i + 1], L=rem["L"][i:i + 1], K=rem["K"], donors=rem["donors"])

    def move_rates(self, step=0.3):
        """Log random walks on rho_c and omega, collapsed over the events they govern.

        Why collapsed: a state with many concepts labelled 'borrow' makes omega large, and a large
        omega makes 'borrow' cheap; with the events held fixed a rate proposal can only ever move
        within that loop (seen as omega 10-20x over truth with the planted rule unfound). The
        proposal is (rate', events') with events' from their exact conditional under rate', so the
        acceptance ratio is the product over (e, c) of Z_ec(rate') / Z_ec(rate).

        The option tables depend on the forms and u only, so they are built once per sweep and
        reused for every rho_c proposal and the omega proposal."""
        st, f, rng = self.st, self.st.f, self.rng
        rem_all = self._event_remainders(range(st.M))
        for c in range(st.M):
            old = st.theta.rho[c]
            prop = old * math.exp(rng.normal(0, step))
            dprior = ev.log_prior_rho(np.array([prop])) - ev.log_prior_rho(np.array([old]))
            rem = self._rem_sub(rem_all, c)
            logZ_old = self._logZ_sum(rem)
            old_evp = {e: st.evp[e][c] for e in f.edges}
            st.theta.rho[c] = prop
            for e in f.edges:
                st.evp[e][c] = ev.event_logprobs(prop, st.theta.omega if st.cands[e] else 0.0, f.length[e])
            logZ_new = self._logZ_sum(rem)
            d = dprior + math.log(prop / old) + logZ_new - logZ_old
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
        logZ_old = self._logZ_sum(rem_all)
        old_evp = {e: list(st.evp[e]) for e in f.edges}
        st.theta.omega = prop
        st.rebuild_event_probs()
        logZ_new = self._logZ_sum(rem_all)
        d = dprior + math.log(prop / old) + logZ_new - logZ_old
        self.acc["omega"][1] += 1
        if self._accept(d):
            st.t_param += dprior; self.acc["omega"][0] += 1
            self._resample_events(rem_all)
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
    FFBS_P = float(__import__("os").environ.get("MASSCOMP_FFBS_P", "0.25"))      # fraction of (node, concept) visited by the lattice FFBS move per sweep (0 = off)
    REGROW_P = float(__import__("os").environ.get("MASSCOMP_REGROW_P", "0"))
    ROOT_REGROW_P = float(__import__("os").environ.get("MASSCOMP_ROOT_REGROW_P", "0"))   # per-concept rate of the root-only regrow   # fraction of (subtree root, concept) regrown per sweep
    TEMPER_P = float(__import__("os").environ.get("MASSCOMP_TEMPER_P", "0"))
    REGROW_EVENTS = __import__("os").environ.get("MASSCOMP_REGROW_EVENTS", "1") == "1"
    REGROW_BORROW = __import__("os").environ.get("MASSCOMP_REGROW_BORROW", "1") == "1"
    INIT = __import__("os").environ.get("MASSCOMP_INIT", "copy")                        # copy | lattice (study start)   # regrow re-decides inherit/replace on subtree edges
    LATENT_MOVES = set(__import__("os").environ.get("MASSCOMP_LATENT_MOVES", "ffbs,jffbs,forms,joint,events").split(","))
    # 'gibbs' (the candidate-set Gibbs move) is off by default since 2026-11-04: the lattice FFBS move is its exact
    # superset and the ESS-per-second grid (results/cloud/diag_41806079603.json) put every mix with it at the bottom
    # on the log-likelihood while costing a third of the sweep.
    PARAM_MOVES = set(__import__("os").environ.get("MASSCOMP_PARAM_MOVES", "lam,u,rates,troot").split(","))   # diagnostics only   # fraction of concepts given a tempered excursion per sweep

    def sweep(self, clamp_latents=False, clamp_u=False, clamp_params=False):
        self._fixed = None                     # per-sweep option tables (see _fixed_options)
        from .ffbs import clear_memo
        clear_memo()                           # lattice results are reused within a sweep only
        if not clamp_latents:
            lm = self.LATENT_MOVES
            if "gibbs" in lm:
                self.inlattice = self.move_forms_gibbs()
            if self.FFBS_P > 0:
                if "ffbs" in lm:
                    self.move_forms_ffbs(self.FFBS_P)
                if "jffbs" in lm:
                    self.move_joint_ffbs(self.FFBS_P)
            if self.REGROW_P > 0:
                self.move_regrow(self.REGROW_P)
            if self.ROOT_REGROW_P > 0:
                self.move_root_regrow(self.ROOT_REGROW_P)
            if self.TEMPER_P > 0:
                cs = [c for c in range(self.st.M) if self.rng.random() < self.TEMPER_P]
                if cs:
                    self.move_tempered(cs)
            if "forms" in lm:
                self.move_forms(n_prop=1)
            if "joint" in lm:
                self.move_joint()
            if "events" in lm:
                self.move_events()
        if clamp_params:
            return
        pm = self.PARAM_MOVES
        if "lam" in pm:
            self.move_lam()
        if not clamp_u and "u" in pm:
            self.move_u()
        if "rates" in pm:
            self.move_rates()
        if "troot" in pm:
            self.move_troot()

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


def _draw_absent(rng, present):
    """A uniform draw over the u slots not in `present` (rejection; present is a few slots of ~870)."""
    while True:
        sl = ev.U_SLOTS[int(rng.integers(ev.N_U_SLOTS))]
        if sl not in present:
            return sl


def _lognorm(w, var):
    return -0.5 * w * w / var - 0.5 * math.log(2 * math.pi * var)


def run_chain(ds, theta0, seed, n_sweeps, burn, thin=10, beta=1.0, init=None, record=True, init_from_prior=False, anneal=0, **sweep_kw):
    """anneal > 0: the first `anneal` sweeps run at beta_i = beta * (i / anneal)^2, i.e. from the prior up
    to the target. A chain started cold from a prior draw at beta = 1 settles into whichever mode the
    first few sweeps happen to find (seen: omega 20x over truth with the planted rule half found,
    while the same chain started at the truth stays there); the ramp lets the forms organise under a
    soft likelihood first. Annealed sweeps are never recorded."""
    rng = np.random.default_rng(seed)
    s = Sampler(ds, theta0.copy(), rng, beta=beta, init=init, init_from_prior=init_from_prior)
    out = []
    for i in range(n_sweeps):
        if anneal:
            s.beta = beta * min(1.0, i / anneal) ** 2
        s.sweep(**sweep_kw)
        if record and i >= burn and (i - burn) % thin == 0:
            out.append(s.snapshot())
    return s, out
