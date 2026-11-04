"""Joint log density of (data, latents, parameters) with a prior/likelihood split for tempering.

Terms, per concept c:
  root:                       log G(y_root)                                  -> prior
  edge e=(p->v), internal v:  log P(event) + log P(y_v | ..., event)         -> prior
  edge e=(p->v), leaf v:      log P(event)                                   -> prior
                              log P(y_v | ..., event)                        -> likelihood
Parameters: log priors on lam, u_e, rho, omega, t_root                       -> prior
"""
import math
import numpy as np
from .segments import N_SEG
from .transducer import EdgeParams
from . import events as ev


class State:
    def __init__(self, ds, theta, y=None, event=None, donor=None):
        self.ds = ds
        self.f = ds.forest
        self.M = ds.M
        self.theta = theta
        self.is_leaf = {v: False for v in range(self.f.n)}
        for l in self.f.leaves:
            self.is_leaf[l] = True
        self.y = dict(y) if y else {}
        for k, v in ds.leaf_forms.items():
            self.y[k] = v
        self.event = dict(event) if event else {}
        self.donor = dict(donor) if donor else {}
        self.cands = {e: self.f.donor_candidates(e) for e in self.f.edges}
        self.joined = len(self.f.roots) == 1 and self.f.family[self.f.roots[0]] == -1
        self.root_edges = list(self.f.children[self.f.roots[0]]) if self.joined else []
        self.root_edge_set = set(self.root_edges)
        # Join temperature (marglik.join_path): the inherit / borrow terms of the root edges are
        # [term]^alpha [G(child)]^(1-alpha). At alpha = 1 this is H1; at alpha = 0 the family roots are
        # ~ G and the joined root, t_root and the root-edge events are decoupled dummies whose priors
        # integrate to 1, so the model is H0. The LBF is log Z(1) - log Z(0) along this path.
        self.alpha = 1.0
        self.rebuild_edge_params()
        self.evp = {}
        self.rebuild_event_probs()
        # term caches
        self.t_prior = {}   # (e,c) -> prior part of edge term; ('root',c) root term
        self.t_lik = {}     # (e,c) -> likelihood part (leaf edges only)
        self.t_param = 0.0
        self.recompute_all()

    # ---- parameter-dependent tables ------------------------------------------------
    def rebuild_edge_params(self):
        self.ep = {e: EdgeParams(self.theta.lam, self.theta.u.get(e, {})) for e in self.f.edges}
        self.adapt = EdgeParams(self.theta.lam, {})

    def rebuild_event_probs(self, edges=None):
        for e in (edges or self.f.edges):
            t = self.f.length[e]
            om = self.theta.omega if self.cands[e] else 0.0
            self.evp[e] = [ev.event_logprobs(self.theta.rho[c], om, t) for c in range(self.M)]

    def log_param_prior(self):
        lp = ev.log_prior_lam(self.theta.lam) + ev.log_prior_rho(self.theta.rho) + ev.log_prior_omega(self.theta.omega)
        for e in self.f.edges:
            lp += ev.log_prior_u(self.theta.u.get(e, {}), self.f.length[e])
        if self.joined:
            lp += ev.log_prior_troot(self.theta.t_root)
        return lp

    # ---- single terms ----------------------------------------------------------------
    def edge_terms(self, e, c, y_child=None, y_parent=None, kind=None, d=None):
        """Return (prior_part, lik_part) for edge e, concept c, optionally with overrides."""
        p = self.f.parent[e]
        yc = y_child if y_child is not None else self.y[(e, c)]
        yp = y_parent if y_parent is not None else self.y[(p, c)]
        k = kind if kind is not None else self.event[(e, c)]
        lp_ev = self.evp[e][c][k]
        if k == ev.INHERIT:
            form = self.ep[e].logprob(yp, yc)
        elif k == ev.REPLACE:
            form = ev.log_G(yc)
        else:
            dd = d if d is not None else self.donor[(e, c)]
            if not self.cands[e]:
                return -1e30, 0.0
            lp_ev += -math.log(len(self.cands[e]))      # donor choice is part of the event prior
            form = self.adapt.logprob(self.y[(dd, c)], yc)
        if self.alpha != 1.0 and k != ev.REPLACE and e in self.root_edge_set:
            form = self.alpha * form + (1.0 - self.alpha) * ev.log_G(yc)
        if self.is_leaf[e]:
            return lp_ev, form
        return lp_ev + form, 0.0

    def join_delta(self):
        """The join path's power variable: sum over the root edges and concepts of log term - log G(child)
        for inherit / borrow events (0 for replace); d/d alpha of the log target."""
        tot = 0.0
        for e in self.root_edges:
            p = self.f.parent[e]
            for c in range(self.M):
                k = self.event[(e, c)]
                yc = self.y[(e, c)]
                if k == ev.INHERIT:
                    tot += self.ep[e].logprob(self.y[(p, c)], yc) - ev.log_G(yc)
                elif k == ev.BORROW:
                    tot += self.adapt.logprob(self.y[(self.donor[(e, c)], c)], yc) - ev.log_G(yc)
        return tot

    def recompute_all(self):
        self.t_param = self.log_param_prior()
        for c in range(self.M):
            for r in self.f.roots:
                self.t_prior[("root", r, c)] = ev.log_G(self.y[(r, c)])
            for e in self.f.edges:
                pr, lk = self.edge_terms(e, c)
                self.t_prior[(e, c)] = pr
                self.t_lik[(e, c)] = lk

    def log_prior(self):
        return self.t_param + sum(self.t_prior.values())

    def log_lik(self):
        return sum(self.t_lik.values())

    def log_target(self, beta=1.0):
        return self.log_prior() + beta * self.log_lik()

    def affected_edges(self, v, c):
        """Edges whose term depends on y[v,c]: incoming, outgoing, and borrowings from v."""
        out = []
        if self.f.parent[v] >= 0:
            out.append(v)
        out.extend(self.f.children[v])
        for e in self.f.edges:
            if self.event.get((e, c)) == ev.BORROW and self.donor.get((e, c)) == v and e not in out:
                out.append(e)
        return out
