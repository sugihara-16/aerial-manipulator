"""Verify coverage and legality independently of the search ordering."""
from collections import Counter
import numpy as np
from amsrr_det.model import DockEdge, validate_graph
from amsrr_det.structure_groups import skeleton_groups, sample_assignment
from build.audit_structure_diversity import tree_key


def test_all_degree_four_tree_skeletons_and_free_contact_counts_are_present():
    groups = skeleton_groups()
    assert Counter(g['modules'] for g in groups) == dict(enumerate([1,1,1,2,3,5,9,18], 1))
    rng = np.random.default_rng(1022)
    for group in groups:
        n = group['modules']
        for k in range(1, 2*n+3):
            edges, surfaces = sample_assignment(group, rng, contact_count=k)
            validate_graph(n, [DockEdge(**edge) for edge in edges])
            occupied = {(e['a'],e['port_a']) for e in edges} | {(e['b'],e['port_b']) for e in edges}
            assert len(set(surfaces)) == k
            assert not occupied.intersection(surfaces)


def test_small_skeleton_sampling_reaches_independently_counted_port_topologies():
    rng = np.random.default_rng(322)
    for n, expected in [(1,1),(2,4),(3,24)]:
        group = next(g for g in skeleton_groups() if g['modules']==n)
        keys = set()
        for _ in range(2000):
            edges, _ = sample_assignment(group, rng)
            keys.add(tree_key(n, edges, ports=True))
        assert len(keys) == expected
