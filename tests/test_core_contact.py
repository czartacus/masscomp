"""State.is_contact: with MASSCOMP_CORE_CONTACT on, a newcomer root edge borrowing from a core family is
contact (not tempered by the join path); in the all-mixed form nothing changes."""
from masscomp import likelihood
from masscomp.tree import benchmark_n


class _S:                                             # just the attributes is_contact reads
    is_contact = likelihood.State.is_contact

    def __init__(self):
        self.f = benchmark_n(3, (0, 1, 2), t_root=0.25)
        self.root_edges = list(self.f.children[self.f.join_root]); self.root_edge_set = set(self.root_edges)
        self.external = set(); self._fam_root = {self.f.family[e]: e for e in self.root_edges}


def _state():
    return _S()


def test_core_contact(monkeypatch):
    st = _state()
    monkeypatch.setattr(likelihood, "CORE_CONTACT", False)
    st.mix_edges = {18}
    assert not st.is_contact(0) and not st.is_contact(9)
    monkeypatch.setattr(likelihood, "CORE_CONTACT", True)
    assert st.is_contact(0) and st.is_contact(9) and st.is_contact(1)
    assert not st.is_contact(19)                      # the newcomer's own nodes are not core
    st.mix_edges = set(st.root_edges)                 # pair / all-mixed form: unchanged
    assert not st.is_contact(0)


def test_put_first_survives_root_length():
    from masscomp.tree import with_root_length
    f = benchmark_n(3, (0, 1, 2), t_root=0.25).put_first([18])
    g = with_root_length(f, 0.7)
    assert g.topo == f.topo and g.topo[1] == 18
    assert g.donor_candidates(0) == [18] and g.donor_candidates(18) == []
