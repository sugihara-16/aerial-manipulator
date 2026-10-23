"""Share immutable geometry on Linux before any numerical worker starts."""
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np


def _generate_trial_candidates(request,modules,index,seed,options):
    from build.frozen_structure_trial import generate_candidate
    rng=np.random.default_rng(np.random.SeedSequence([seed,modules,index]))
    return [generate_candidate(request,modules,rng,generator=options['generator'],
        common_initialization=options['common_initialization'],retain_task_pose=options['retain_task_pose'])
        for _ in range(options['starts'])]


class _TrialFuture:
    """Keep a trial indivisible as evidence, but distribute its physical screens."""
    def __init__(self,pool,args,options):
        self.pool,self.args,self.options=pool,args,options;self.started=time.perf_counter()
        self.generation=pool.apply_async(_generate_trial_candidates,(*args,options));self.screens=None

    def ready(self):
        if self.screens is None:
            if not self.generation.ready():return False
            from build.frozen_structure_trial import screen_candidate
            candidates=self.generation.get()
            self.screens=[self.pool.apply_async(screen_candidate,(self.args[0],c),
                dict(task_wrench=self.options['task_wrench'])) for c in candidates]
        return all(f.ready() for f in self.screens)

    def get(self):
        if not self.ready():raise ValueError('independent trial is not complete')
        rows=[];best=None;prepared=None
        for start,f in enumerate(self.screens):
            row,plan,current=f.get();row['start']=start;rows.append(row)
            if current is not None and (prepared is None or current['burden']<prepared['burden']):best,prepared=plan,current
        return dict(modules=self.args[1],trial=self.args[2],seed=self.args[3],rows=rows,
            burden=prepared['burden'] if prepared is not None else float('inf'),
            plan=best,prepared=prepared,wall_s=time.perf_counter()-self.started)


class TrialPool:
    def __init__(self,context,**kwargs):self.pool=context.Pool(**kwargs)
    def apply_async(self,fn,args,kwds=None):
        from build.frozen_structure_trial import trial
        if fn is trial:return _TrialFuture(self.pool,args,kwds)
        return self.pool.apply_async(fn,args,kwds or {})
    def terminate(self):self.pool.terminate()
    def join(self):self.pool.join()


def planning_context():
    # Fork only a single-threaded parent. After control/solver threads exist,
    # use fresh interpreters so their locks cannot be inherited into a child.
    if sys.platform!='linux' or len(list(Path('/proc/self/task').iterdir()))!=1:
        return mp.get_context('spawn')
    from .structure_groups import skeleton_groups
    from .grasp_geometry import part_bounds
    from .contact_geometry import material_triangles
    from .mujoco_model import CONTACT_CACHE,_part_templates
    from .grasp_control import _surface_source
    from .model import PORTS
    skeleton_groups();part_bounds();_part_templates(str(CONTACT_CACHE.resolve()))
    for port in PORTS:material_triangles(port,str(CONTACT_CACHE.resolve()))
    for link in ('pitch_dock_mech1','pitch_dock_mech2','yaw_dock_mech1','yaw_dock_mech2'):_surface_source(link)
    return mp.get_context('fork')
