"""Prepare immutable single-part geometry once; stale caches are not used."""
import json
from amsrr_det.grasp_geometry import part_geometry
from amsrr_det.part_bounds_cache import CACHE, identity


def main():
    bounds = [dict(link=link, center=T.act((shape.aabb_local.min_ + shape.aabb_local.max_) / 2).tolist(),
                   half=(abs(T.rotation) @ ((shape.aabb_local.max_ - shape.aabb_local.min_) / 2)).tolist())
              for link, T, shape in part_geometry().entries]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(inputs=identity(), bounds=bounds), indent=2))
    temporary.replace(CACHE)
    print(f'Saved {len(bounds)} authored-part bounds to {CACHE}')


if __name__ == '__main__':
    main()
