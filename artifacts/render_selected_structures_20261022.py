"""Display recorded successful structures, never evaluate or select a plan."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from build.load_saved_plan import load_saved_plan
from scripts.report_structure_selection import signature


def render_case(root, output):
    plan, request = load_saved_plan(root / 'generated_plan')
    model = plan.plant.mj
    model.vis.global_.offwidth = 800
    model.vis.global_.offheight = 600
    model.vis.headlight.ambient[:] = .5
    model.vis.headlight.diffuse[:] = .8
    data = mujoco.MjData(model)
    data.qpos[:] = np.load(root / 'entry_CARRY.npz')['q']
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    robot = [i for i in range(model.ngeom)
             if model.geom(i).name.startswith('m') or model.geom(i).name == 'object']
    positions = data.geom_xpos[robot]
    camera.lookat[:] = (positions.min(axis=0) + positions.max(axis=0)) / 2
    camera.distance = max(1.3, 2.1 * np.max(np.ptp(positions, axis=0)))
    camera.azimuth = 135
    camera.elevation = -35
    with mujoco.Renderer(model, height=600, width=800, max_geom=30000) as renderer:
        renderer.update_scene(data, camera)
        frame = Image.fromarray(renderer.render())
    frame.save(output)
    return frame, request


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    cases = []
    for path in sorted(args.root.glob('*/result.json')):
        result = json.loads(path.read_text())
        if not result.get('task_success') or not (path.parent / 'entry_CARRY.npz').exists():
            continue
        plan = json.loads((path.parent / 'generated_plan/plan.json').read_text())
        modules = plan['model']['modules']
        skeleton = signature(modules, plan['model']['edges'], ports=False)
        pitch = sum(port.startswith('pitch') for _, port in plan['surfaces'])
        cases.append((path.parent, modules, len(plan['surfaces']), skeleton, pitch))
    selected, used = [], set()
    ordered = sorted(cases, key=lambda c: (-c[4], -c[1], str(c[0])))
    # Show the range of actually successful module counts before filling
    # remaining cells with different ports/skeletons. Display selection only.
    representatives = [next(c for c in ordered if c[1] == count)
                       for count in sorted({c[1] for c in cases}, reverse=True)]
    for case in representatives + ordered:
        key = (case[1], case[2], case[3], bool(case[4]))
        if key not in used:
            selected.append(case)
            used.add(key)
        if len(selected) == args.limit:
            break
    if not selected:
        raise ValueError('no successful recorded CARRY pose')
    gallery = Image.new('RGB', (1600, 650 * ((len(selected) + 1) // 2)), 'white')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    manifest = []
    for index, (root, modules, contacts, skeleton, pitch) in enumerate(selected):
        frame, request = render_case(root, args.output / (root.name + '.png'))
        x, y = (index % 2) * 800, (index // 2) * 650
        gallery.paste(frame, (x, y + 50))
        text = f'{root.name}: {modules} modules, {contacts} pads ({pitch} pitch)'
        ImageDraw.Draw(gallery).text((x + 12, y + 14), text, fill='black', font=font)
        manifest.append(dict(case=root.name, modules=modules, contacts=contacts,
                             pitch_pads=pitch, skeleton=skeleton, source=str(root),
                             observation='entry_CARRY', evaluation='full task succeeded'))
    gallery.save(args.output / 'structures.jpg')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(args.output / 'structures.jpg')
