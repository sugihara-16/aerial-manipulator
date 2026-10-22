"""Cached geometry must preserve authored bounds and reject stale inputs."""
import json
import numpy as np
from amsrr_det import part_bounds_cache as cache
from amsrr_det.grasp_geometry import part_geometry


def test_authored_bounds_equal_original():
    saved = cache.read_bounds()
    assert saved is not None
    original = [(link, T.act((shape.aabb_local.min_ + shape.aabb_local.max_) / 2),
                 abs(T.rotation) @ ((shape.aabb_local.max_ - shape.aabb_local.min_) / 2))
                for link, T, shape in part_geometry().entries]
    assert len(saved) == len(original)
    for (a, ac, ah), (b, bc, bh) in zip(saved, original):
        assert a == b
        np.testing.assert_array_equal(ac, bc)
        np.testing.assert_array_equal(ah, bh)


def test_stale_or_invalid_bounds_not_used(tmp_path, monkeypatch):
    raw = json.loads(cache.CACHE.read_text())
    path = tmp_path / 'bounds.json'
    monkeypatch.setattr(cache, 'CACHE', path)
    assert cache.read_bounds() is None
    raw['inputs']['versions']['coal'] = 'different-version'
    path.write_text(json.dumps(raw))
    assert cache.read_bounds() is None
    raw['inputs'] = cache.identity()
    raw['bounds'][0]['half'][0] = -1
    path.write_text(json.dumps(raw))
    assert cache.read_bounds() is None
    path.write_text('{invalid')
    assert cache.read_bounds() is None
