"""All legal tree skeletons and task-time random port/contact assignments.

Only the 40 unlabeled skeletons and port-count combinatorics are cached, not
finished structures or postures. Every legal assignment has positive support.
Sampling coverage alone is not evidence of physical or optimization quality.
"""
from functools import lru_cache
from itertools import product
from math import prod
import networkx as nx
import numpy as np


@lru_cache(maxsize=1)
def skeleton_groups():
    groups = []
    for n in range(1, 9):
        trees = [nx.empty_graph(1)] if n == 1 else nx.nonisomorphic_trees(n)
        index = 0
        for tree in trees:
            if max(dict(tree.degree).values()) > 4:
                continue
            edges = tuple(tree.edges())
            assignments = []
            for signs in product((0, 1), repeat=len(edges)):
                pitch = [0]*n
                for (a, b), sign in zip(edges, signs):
                    pitch[a] += sign; pitch[b] += 1-sign
                yaw = [tree.degree[i]-pitch[i] for i in range(n)]
                if max(pitch+yaw) > 2:
                    continue
                weight = prod((2 if p else 1)*(2 if y else 1)
                              for p, y in zip(pitch, yaw))
                assignments.append((signs, weight))
            groups.append(dict(id=f'{n}:{index}', modules=n, skeleton=edges,
                               orientations=tuple(assignments)))
            index += 1
    return tuple(groups)


def sample_assignment(group, rng, *, contact_count=None):
    """Uniform port assignment within a skeleton, then a uniform k-subset.

    When k is unspecified, its distribution is uniform over 1..2N+2. This is
    intentionally NOT uniform over all nonempty subsets, which would heavily
    favor middle contact counts. The distribution is explicit and task-free.
    """
    orientations = group['orientations']
    weights = np.array([item[1] for item in orientations], dtype=float)
    signs = orientations[int(rng.choice(len(weights), p=weights/weights.sum()))][0]
    available = {}
    for module in range(group['modules']):
        for kind in ('pitch', 'yaw'):
            available[module, kind] = [f'{kind}{i}' for i in rng.permutation([1, 2])]
    edges = []
    for (a, b), sign in zip(group['skeleton'], signs):
        kinds = ('pitch', 'yaw') if sign else ('yaw', 'pitch')
        edges.append(dict(a=a, b=b, port_a=available[a, kinds[0]].pop(),
                          port_b=available[b, kinds[1]].pop()))
    free = [(module, port) for (module, _), ports in available.items() for port in ports]
    k = int(rng.integers(1, len(free)+1)) if contact_count is None else contact_count
    if not isinstance(k, int) or not 1 <= k <= len(free):
        raise ValueError('contact count outside available free pads')
    selected = sorted(rng.choice(len(free), size=k, replace=False))
    return edges, [free[i] for i in selected]
