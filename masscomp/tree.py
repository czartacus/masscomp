"""Trees and forests (plan §2). Nodes are integers; leaves carry language labels."""
from dataclasses import dataclass, field
import numpy as np

# Benchmark-A family topology: root -> (A, B); A -> (l1, l2); B -> (C, l3); C -> (l4, l5)
_FAMILY_PARENT = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 5, 8: 5}
_FAMILY_LEN = {1: 0.4, 2: 0.4, 3: 0.3, 4: 0.3, 5: 0.3, 6: 0.2, 7: 0.2, 8: 0.2}
_FAMILY_LEAVES = [3, 4, 6, 7, 8]


@dataclass
class Forest:
    parent: list              # parent[v] or -1 for a root
    length: list              # length[v] = length of edge parent[v] -> v (0 for roots)
    family: list              # family index of each node (joined root gets -1)
    leaves: list              # leaf node ids
    roots: list = field(default_factory=list)
    children: dict = field(default_factory=dict)
    names: dict = field(default_factory=dict)
    first: list = field(default_factory=list)   # join-root children put first in the order (put_first)

    def __post_init__(self):
        n = len(self.parent)
        self.roots = [v for v in range(n) if self.parent[v] < 0]
        # the joining root of H1 (family -1), if any; other roots are family roots of separate trees
        # (H0's families, or donor families that are never joined)
        self.join_root = next((v for v in self.roots if self.family[v] < 0), None)
        self.children = {v: [] for v in range(n)}
        for v in range(n):
            if self.parent[v] >= 0:
                self.children[self.parent[v]].append(v)
        if self.first and self.join_root is not None:
            R = self.join_root; fs = set(self.first)
            self.children[R] = [c for c in self.children[R] if c in fs] + [c for c in self.children[R] if c not in fs]
        self.internal = [v for v in range(n) if v not in set(self.leaves)]
        self.edges = [v for v in range(n) if self.parent[v] >= 0]   # edge id == child id
        self.ancestors = {v: self._anc(v) for v in range(n)}
        self.descendants = {v: self._desc(v) for v in range(n)}
        self.topo = self._topo()   # parents before children

    @property
    def n(self):
        return len(self.parent)

    def _anc(self, v):
        out = set(); p = self.parent[v]
        while p >= 0:
            out.add(p); p = self.parent[p]
        return out

    def _desc(self, v):
        out = set(); stack = list(self.children[v])
        while stack:
            c = stack.pop(); out.add(c); stack.extend(self.children[c])
        return out

    def put_first(self, nodes):
        """Move `nodes` (children of the join root: newcomer family roots) to the front of the join root's
        children, so they come first in the topological order and donor_candidates lets the core's root edges
        borrow FROM the newcomer, not the newcomer's root edge from the core (MASSCOMP_CORE_CONTACT=order).
        Returns self."""
        R = self.join_root
        self.first = [c for c in self.children[R] if c in set(nodes)]
        self.children[R] = self.first + [c for c in self.children[R] if c not in set(nodes)]
        self.topo = self._topo()
        return self

    def _topo(self):
        order = []; stack = list(self.roots)
        while stack:
            v = stack.pop(0); order.append(v); stack.extend(self.children[v])
        return order

    def donor_candidates(self, recipient):
        """Nodes a borrowing into `recipient` may copy from: nodes strictly earlier in the
        (breadth-first) topological order that are not ancestors of the recipient. Restricting
        donors to earlier nodes makes the borrowing graph acyclic, so the joint density of all
        forms is a proper (normalised) distribution; mutual borrowing A<->B would not be."""
        bad = self.ancestors[recipient] | {recipient}
        pos = self.topo.index(recipient)
        out = [v for v in self.topo[:pos] if v not in bad]
        if CROSS_FAMILY_ONLY:
            # Within-family donors make the model degenerate: borrowing from the parent's sibling
            # has the same likelihood as inheriting (same transducer, no branch rule), so only the
            # prior on omega separates the two, and the chain drifts to "everything borrowed, omega
            # large" (planted-rule trace, 2026-11-02). Cross-family borrowing is the thing the test
            # needs omega for, so donors are restricted to nodes of other families (the joining
            # root under H1 belongs to no family and is never a donor).
            out = [v for v in out if self.family[v] != self.family[recipient] and self.family[v] >= 0]
        return out


CROSS_FAMILY_ONLY = True   # see Forest.donor_candidates


def benchmark_a(related=False, t_root=1.5):
    """Two benchmark-A families. If related, add a joining root R with edges of length t_root."""
    parent, length, family, leaves, names = [], [], [], [], {}
    for fam in range(2):
        off = len(parent)
        for v in range(9):
            parent.append(off + _FAMILY_PARENT[v] if v in _FAMILY_PARENT else -1)
            length.append(_FAMILY_LEN.get(v, 0.0))
            family.append(fam)
        for i, lv in enumerate(_FAMILY_LEAVES):
            leaves.append(off + lv); names[off + lv] = f"F{fam}L{i+1}"
    if related:
        R = len(parent)
        parent.append(-1); length.append(0.0); family.append(-1)
        for fam_root in (0, 9):
            parent[fam_root] = R; length[fam_root] = t_root
    return Forest(parent, length, family, leaves, names=names)


def benchmark_n(n_fams=3, joined=(0, 1, 2), t_root=1.5):
    """n_fams benchmark-A families (nodes 9k .. 9k+8 for family k); the families in `joined` hang under one
    joining root R (appended last) with root edges of length t_root, the others are separate trees. Node
    ids do not depend on `joined`, so datasets made on one such forest can be fitted on another."""
    parent, length, family, leaves, names = [], [], [], [], {}
    for fam in range(n_fams):
        off = len(parent)
        for v in range(9):
            parent.append(off + _FAMILY_PARENT[v] if v in _FAMILY_PARENT else -1)
            length.append(_FAMILY_LEN.get(v, 0.0))
            family.append(fam)
        for i, lv in enumerate(_FAMILY_LEAVES):
            leaves.append(off + lv); names[off + lv] = f"F{fam}L{i+1}"
        names[off] = f"F{fam}"
    if joined:
        R = len(parent)
        parent.append(-1); length.append(0.0); family.append(-1)
        for fam in joined:
            parent[9 * fam] = R; length[9 * fam] = t_root
    return Forest(parent, length, family, leaves, names=names)


def with_root_length(forest, t_root):
    """Return a copy of a joined forest with the root edges set to t_root."""
    length = list(forest.length)
    R = forest.join_root
    for c in forest.children[R]:
        length[c] = t_root
    return Forest(list(forest.parent), length, list(forest.family), list(forest.leaves), names=dict(forest.names),
                  first=list(forest.first))   # keep a put_first order across t_root moves
