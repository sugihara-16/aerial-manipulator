"""Saved replay must preserve controller settings and endpoint coordinates."""
import json
from pathlib import Path

import numpy as np

from scripts.replay_panel import CONTROL_FLAGS, replay_arguments


def saved_case(tmp_path):
    case = tmp_path / 'saved'
    (case / 'generated_plan').mkdir(parents=True)
    (case / 'generated_plan/scene.xml').write_text('''<mujoco><worldbody>
      <body pos="0 0 1"><freejoint/><geom size=".1"/></body>
      <body pos="1 0 1"><freejoint/><geom size=".1"/></body>
    </worldbody></mujoco>''')
    settings = dict.fromkeys(CONTROL_FLAGS, True)
    settings.update(steps=80, duration=110., global_structure_search=True)
    (case / 'experiment.json').write_text(json.dumps(dict(execution_configuration=settings)))
    q = np.array([0., 0., .3, 1., 0., 0., 0., 1., 0., .4, 1., 0., 0., 0.])
    np.savez(case / 'endpoint.npz', q=q)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    return case, scratch, q


def test_replay_preserves_endpoint_and_disables_new_search(tmp_path):
    case, scratch, q = saved_case(tmp_path)
    original = (case / 'endpoint.npz').read_bytes()
    args = replay_arguments(case, scratch)
    with np.load(Path(args[args.index('--goal') + 1])) as data:
        lowered = data['q'].copy()
    lowered[[2, 9]] -= .15  # The controller's existing airborne-goal conversion.
    np.testing.assert_allclose(lowered, q, atol=1e-15)
    assert (case / 'endpoint.npz').read_bytes() == original
    assert '--global-structure-search' not in args
    assert '--request' not in args
    assert all('--' + flag.replace('_', '-') in args for flag in CONTROL_FLAGS)
    assert args[args.index('--steps') + 1] == '80'
    assert args[args.index('--duration') + 1] == '110.0'


def test_preview_only_shortens_duration(tmp_path):
    case, scratch, _ = saved_case(tmp_path)
    normal = replay_arguments(case, scratch)
    preview = replay_arguments(case, scratch, .1)
    duration = normal.index('--duration') + 1
    assert preview[duration] == '0.1'
    preview[duration] = normal[duration]
    assert preview == normal
