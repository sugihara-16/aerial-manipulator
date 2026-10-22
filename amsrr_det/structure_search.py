"""Grow grasp branches from a small seed; solve geometry for each new task.

No task-indexed finished graph/posture is loaded. This bounded search is not
complete: failure is 'not found', never proof of physical impossibility.
"""
import time
from collections import deque
import numpy as np
import pinocchio as pin
from .model import build_model
from .morphology import grow_tree
from .grasp_geometry import fit_contacts,free_surfaces
from .grasp_task import plan_seed,refine_material_posture,finish_plan,support_obstacles
from .mujoco_model import make_plant,CONTACT_CACHE


class SearchBudgetExpired(ValueError):
    pass


def check_search_deadline(deadline):
    # Stop between numerical solves; an in-progress solve is not preempted.
    if deadline is not None and time.perf_counter()>=deadline:
        raise SearchBudgetExpired('additional structure search time exhausted')


def copy_posture(old_model,old_meta,old_q,model,meta,source):
    q=pin.neutral(model);q[:7]=old_q[:7];q[meta['object']['q']:]=source
    values={a['name']:old_q[a['q']] for a in old_meta['actuators']}
    for a in meta['actuators']:
        if a['name'] in values:q[a['q']]=values[a['name']]
    return q


def branch_postures(draft,shape,source,destination,support_top, *, maximum_modules=5):
    """One/two-link branches plus an independent third material contact."""
    base=draft['plant'];n=base.meta['modules'];occupied=set(draft['surfaces'])
    R=pin.Quaternion(source[3:]).matrix();top,normal=shape.ray_surface(R.T@np.array([0.,0.,1.]))
    points=np.vstack([draft['points'],top]);normals=np.vstack([draft['normals'],-normal])
    obstacles=support_obstacles(shape,source,destination,support_top)
    first=[edges for edges in grow_tree(n,base.meta['edges'])
           if (edges[-1]['a'],edges[-1]['port_a']) not in occupied]
    # A branch attaches to the seed's root first. Other attachment sites remain
    # future search extensions, not a claim of infeasibility when this fails.
    first=[edges for edges in first if edges[-1]['a']==0]
    levels=[(n+1,first)]
    if maximum_modules>=n+2:
        levels.append((n+2,[edges for first_edges in first for edges in grow_tree(n+1,first_edges)
                             if edges[-1]['a']==n]))
    for count,graphs in levels:
        if count>maximum_modules:continue
        pool=[]
        for edges in graphs:
            m,meta=build_model(count,edges,shape.payload())
            q=copy_posture(base.pin,base.meta,draft['closed'],m,meta,source)
            incoming=edges[-1]['port_b']
            # Opposite port on the incoming joint axis is a first orientation
            # seed. It includes both yaw and pitch through the connection rule.
            outgoing=incoming[:-1]+('2' if incoming[-1]=='1' else '1')
            surfaces=draft['surfaces']+[(count-1,outgoing)]
            closed,fit=fit_contacts(m,meta,q,surfaces,points@R.T+source[:3]+.001*(normals@R.T),normals@R.T,
                                   floor_clearance=.03,base_rotation=True,obstacles=obstacles,max_evaluations=65)
            reserve=min(min(closed[a['q']]-a['lower'],a['upper']-closed[a['q']]) for a in meta['actuators'] if a['role']=='dock')
            score=fit['position_error_m']+.15*fit['normal_error']+.01*max(0.,.12-reserve)
            pool.append((score,edges,closed,surfaces,fit))
        for score,edges,q,surfaces,fit in sorted(pool,key=lambda item:item[0]):
            yield dict(modules=count,edges=edges,q=q,surfaces=surfaces,points=points.copy(),normals=normals.copy(),fit=fit,score=score)


def plan_three_contacts(shape,source,destination,support_top, *, mode='yaw',axis_world=None,maximum_modules=5):
    """Development entry: conservative three-pad grasp, generated branching."""
    start=time.perf_counter();source=np.asarray(source);destination=np.asarray(destination)
    draft=plan_seed(shape,source,destination,support_top,mode=mode,axis_world=axis_world,geometry_only=True)
    log=[]
    for candidate in branch_postures(draft,shape,source,destination,support_top,maximum_modules=maximum_modules):
        row={k:candidate[k] for k in ['modules','edges','surfaces','fit','score']};log.append(row)
        if candidate['fit']['position_error_m']>.015 or candidate['fit']['normal_error']>.15:
            row['rejected']='coarse_reach';continue
        p=make_plant(candidate['modules'],candidate['edges'],shape.payload(),mesh_cache=CONTACT_CACHE,floor=True)
        try:
            opened,closed,fit=refine_material_posture(p,shape,source,candidate['q'],candidate['surfaces'],candidate['points'],candidate['normals'],
                obstacles=support_obstacles(shape,source,destination,support_top))
            plan=finish_plan(p,shape,source,destination,support_top,opened,closed,candidate['surfaces'],candidate['points'],candidate['normals'],fit,start)
            plan.fit['structure_search']=log;plan.fit['structure_scope']='seed plus root-attached one/two-link branches'
            return plan
        except ValueError as error:row['rejected']=str(error)
    error=ValueError('no three-contact candidate found in this search scope');error.evidence=dict(structure_search=log)
    raise error



def side_inclinations(shape,source,axis,support_top):
    """A lower-side seed inside the available point-to-table clearance band.

    This necessary point clearance is not a robot-clearance guarantee; the
    ordinary full geometry checks remain required after posture fitting.
    """
    R=pin.Quaternion(source[3:]).matrix()
    target=(source[2]+support_top+.03)/2
    if target>=source[2]:return (0.,np.pi/6)
    lower,upper=-np.pi/2+1e-6,0.
    for _ in range(24):
        angle=(lower+upper)/2;points,_=shape.opposed_sites(source,axis,angle)
        height=float(np.min((points@R.T)[:,2]+source[2]))
        if height<target:lower=angle
        else:upper=angle
    return (0.,(lower+upper)/2,np.pi/6)


def grasp_candidates(shape,source,destination,support_top, *, preferred_mode='yaw',maximum_modules=8,contact_counts=(2,3,4),contact_tilts=None,normal_moment_refinement=False,deadline=None):
    """Yield newly solved candidates so physical rejection can resume search.

    A yielded candidate passed geometry/static screens only. The caller records
    physical execution before deciding whether to accept it or request another.
    """
    from .expanded_search import expanded_postures
    requested_counts=set(contact_counts)
    if not requested_counts or not requested_counts<={2,3,4}:raise ValueError('supported contact counts are 2, 3, 4')
    scheduled_counts=requested_counts|({3} if 4 in requested_counts else set())
    source=np.asarray(source);destination=np.asarray(destination)
    R=pin.Quaternion(source[3:]).matrix()
    preferred=max([R[:,i] for i in range(3)],key=lambda n:np.linalg.norm(n[:2]) if abs(n[1])>.1 else 0.).copy()
    preferred[2]=0;preferred/=np.linalg.norm(preferred)
    axes=[preferred]
    for direction in [*R.T,np.cross([0.,0.,1.],destination[:3]-source[:3])]:
        axis=direction.copy();axis[2]=0
        if np.linalg.norm(axis)<1e-8:continue
        axis/=np.linalg.norm(axis)
        if all(abs(axis@previous)<1-1e-7 for previous in axes):axes.append(axis)
    attempts=[];planning_s=0.;pending=deque()
    modes=[preferred_mode,'pitch' if preferred_mode=='yaw' else 'yaw']
    for axis in axes:
        tilts=side_inclinations(shape,source,axis,support_top) if contact_tilts is None else contact_tilts
        for tilt in tilts:
            seed_states={mode:{} for mode in modes}
            # Try both grasp axes at a contact count before growing more arms.
            # Each mode keeps its own task-local seed, shared across counts.
            for count in sorted(scheduled_counts):
                for mode in modes:
                    seed_state=seed_states[mode]
                    postures=expanded_postures(shape,source,destination,support_top,mode=mode,axis_world=axis,
                        maximum_modules=maximum_modules,contact_counts=(count,),screen_parents=True,contact_tilt=tilt,deadline=deadline,seed_state=seed_state)
                    pending.append((mode,axis,tilt,count,postures))
    # Visit another approach/mode before exhausting one branch's extensions.
    # Iterators remain lazy; the common first accepted seed adds no fit work.
    while pending:
        try:check_search_deadline(deadline)
        except SearchBudgetExpired as error:
            error.evidence=dict(task_level_search=attempts,planning_s=planning_s);raise
        mode,axis,tilt,count,postures=pending.popleft();before=time.perf_counter()
        row=dict(mode=mode,axis_world=axis.tolist(),contact_tilt_rad=float(tilt),requested_contact_count=count)
        try:
            draft=next(postures)
        except StopIteration:
            planning_s+=time.perf_counter()-before;continue
        except SearchBudgetExpired as error:
            row.update(status='search_budget_exhausted',wall_s=time.perf_counter()-before)
            attempts.append(row);planning_s+=row['wall_s']
            error.evidence=dict(task_level_search=attempts,planning_s=planning_s);raise
        except ValueError as error:
            planning_s+=time.perf_counter()-before
            attempts.append(dict(row,status='seed_not_found',reason=str(error)));continue
        pending.append((mode,axis,tilt,count,postures));meta=draft['plant'].meta
        row.update(modules=meta['modules'],edges=meta['edges'],surfaces=draft['surfaces'],
            contact_count=len(draft['surfaces']),coarse_fit=draft['fit']);attempts.append(row)
        if draft.get('scheduler_pending') or count not in requested_counts:
            row.update(status='parent_pending' if draft.get('scheduler_pending') else 'intermediate_parent',wall_s=time.perf_counter()-before)
            planning_s+=row['wall_s'];continue
        # Every contact count faces the same material, load and path screens.
        if 'rejection'in draft:
            row.update(status='rejected',reason=draft['rejection'],wall_s=time.perf_counter()-before)
            planning_s+=row['wall_s'];continue
        plan=draft['plan']
        if normal_moment_refinement:
            from .grasp_refinement import refine_normal_moments
            refined,report=refine_normal_moments(plan)
            if refined is not None:plan=refined
            plan.fit['contact_geometry_refinement']=report
            row['contact_geometry_refinement']=dict(static_candidate=report['static_candidate'],wall_s=report['wall_s'],reason=report.get('reason'))
        row.update(status='static_candidate',wall_s=time.perf_counter()-before)
        planning_s+=row['wall_s'];plan.wall_s=planning_s
        plan.fit.update(task_level_search=[dict(a) for a in attempts],
            eligible_contact_counts=list(contact_counts),maximum_modules=maximum_modules,
            selection_rule='first fully screened candidate; round robin over contact count, approach and mode, bounded branch growth; physical execution unverified')
        yield plan
    error=ValueError('no grasp found within generated contact-direction/branch search')
    error.evidence=dict(task_level_search=attempts);raise error


def static_load_burden(plan):
    """Worst final allocated motor burden across the same fixed-pose screens.

    This measures static reserve, not realizable contact forces or task success.
    """
    return max([row['max_normalized_joint_effort'] for row in plan.trims]+
               [row['max_normalized_joint_effort'] for row in plan.fit['restoring_load_screen']['directions']])


def select_static_candidate(candidates,limit=None,*,deadline=None):
    """Compare generated candidates completed before the common deadline.

    An ongoing numerical solve is not preempted. Its late result is recorded,
    but cannot displace a candidate that finished within the search budget.
    """
    if limit is None and deadline is None:raise ValueError('a count or time budget is required')
    if limit is not None and (not isinstance(limit,int) or limit<1):raise ValueError('invalid candidate limit')
    start=time.perf_counter();plans=[];rows=[];search_error=None;latest_search=[]
    while limit is None or len(rows)<limit:
        before=time.perf_counter()
        try:
            check_search_deadline(deadline)
            plan=next(candidates)
        except (ValueError,StopIteration) as error:
            if not plans:raise
            search_error=dict(reason=str(error),evidence=getattr(error,'evidence',None));break
        completed=deadline is None or time.perf_counter()<=deadline
        latest_search=plan.fit['task_level_search']
        rows.append(dict(candidate=len(rows)+1,static_burden=static_load_burden(plan),
            modules=plan.plant.meta['modules'],edges=plan.plant.meta['edges'],surfaces=plan.surfaces,
            completed_within_budget=completed,
            generation_s=time.perf_counter()-before,closed_q=plan.closed_q.tolist(),
            open_q=plan.open_q.tolist(),points=plan.points.tolist(),normals=plan.normals.tolist(),
            trim_final_effort_ratios=[row['max_normalized_joint_effort'] for row in plan.trims],
            restoring_load_screen=plan.fit['restoring_load_screen']))
        if completed:plans.append(plan)
        else:
            search_error=dict(reason='candidate completed after search deadline',evidence=dict(candidate=len(rows)))
            break
    if not plans:
        error=SearchBudgetExpired('no candidate completed within search budget')
        error.evidence=dict(rows=rows,task_level_search=latest_search);raise error
    selected=min(range(len(plans)),key=lambda i:(rows[i]['static_burden'],i))
    plan=plans[selected];plan.wall_s=time.perf_counter()-start
    plan.fit['task_level_search']=latest_search
    rule='minimum worst final allocated static motor burden; retain first on ties; no motion certificate'
    plan.fit.update(selection_rule=rule,candidate_selection=dict(limit=limit,rows=rows,
        selected_candidate=selected+1,search_error=search_error,wall_s=plan.wall_s,rule=rule))
    return plan


def plan_grasp(shape,source,destination,support_top, *, preferred_mode='yaw',maximum_modules=8,contact_counts=(2,3,4),contact_tilts=None,normal_moment_refinement=False,candidate_limit=None,search_seconds=30.):
    """Generate throughout a bounded search, then choose a static candidate.

    Two, three and four contacts are eligible. Neither the first-pass rule nor
    the static-burden comparison guarantees motion or exhaustive search.
    """
    if not np.isfinite(search_seconds) or search_seconds<=0:raise ValueError('invalid search duration')
    deadline=time.perf_counter()+search_seconds
    candidates=grasp_candidates(shape,source,destination,support_top,
        preferred_mode=preferred_mode,maximum_modules=maximum_modules,contact_counts=contact_counts,contact_tilts=contact_tilts,normal_moment_refinement=normal_moment_refinement,deadline=deadline)
    selected=select_static_candidate(candidates,candidate_limit,deadline=deadline)
    selected.fit['candidate_selection']['search_target_s']=search_seconds
    return selected
