"""The three-family forest and the simulated newcomer datasets (masscomp.tree.benchmark_n, simulate_newcomer)."""
from masscomp.tree import benchmark_n
from masscomp.simulate import simulate_newcomer


def test_benchmark_n_shapes():
    f3 = benchmark_n(3, (0, 1, 2), t_root=0.7)
    f2 = benchmark_n(3, (0, 1), t_root=0.7)
    assert f3.n == f2.n == 28 and f3.join_root == f2.join_root == 27
    assert sorted(f3.children[27]) == [0, 9, 18] and sorted(f2.children[27]) == [0, 9]
    assert 18 in f2.roots and f2.length[18] == 0.0 and f3.length[18] == 0.7
    assert len(f3.leaves) == 15 and f3.names[18] == "F2"


def test_simulate_newcomer_matched():
    ins, out = simulate_newcomer(1.0, 10, 3)
    assert ins.M == out.M == 10 and set(ins.leaf_forms) == set(out.leaf_forms)
    # families 0 and 1 share the root in both datasets; only family 2's history differs
    assert ins.forest.join_root == out.forest.join_root
