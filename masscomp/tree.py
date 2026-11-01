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

    def __post_init__(self):
        n = len(self.parent)
        self.roots = [v for v in range(n) if self.parent[v] < 0]
        self.children = {v: [] for v in range(n)}
        for v in range(n):
            if self.parent[v] >= 0:
                self.children[self.parent[v]].append(v)
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
        return [v for v in self.topo[:pos] if v not in bad]


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


def with_root_length(forest, t_root):
    """Return a copy of a joined forest with the root edges set to t_root."""
    length = list(forest.length)
    R = forest.roots[0]
    for c in forest.children[R]:
        length[c] = t_root
    return Forest(list(forest.parent), length, list(forest.family), list(forest.leaves), names=dict(forest.names))
