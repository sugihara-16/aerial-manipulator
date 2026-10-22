"""Bounded contact-count/attachment search, generated from legal port rules.

The two-pad posture is an initial guess, not the only eligible final structure.
Search failure is not physical impossibility. Every output still needs the
same material-contact, load, clearance and physical-execution checks.
"""
from types import SimpleNamespace
import time
import numpy as np
import pinocchio as pin
from .model import build_model
from .morphology import grow_tree
from .grasp_geometry import free_surfaces,grasp_frame,fit_contacts
from .structure_search import copy_posture,check_search_deadline,SearchBudgetExpired
from .grasp_task import plan_seed,refine_material_posture,finish_plan,support_obstacles
from .mujoco_model import make_plant,CONTACT_CACHE


def contact_extensions(n,edges,surfaces,maximum_modules=8):
    """Use an existing free pad, or grow one/two links from ANY free port."""
    reserved=set(map(tuple,surfaces))
    for surface in free_surfaces(dict(modules=n,edges=edges)):
        if surface not in reserved:yield n,edges,surface
    first=[g for g in grow_tree(n,edges) if (g[-1]['a'],g[-1]['port_a']) not in reserved]
    levels=[(n+1,first)]
    if n+2<=maximum_modules:
        levels.append((n+2,[g for parent in first for g in grow_tree(n+1,parent) if g[-1]['a']==n]))
    for count,graphs in levels:
        if count>maximum_modules:continue
        for graph in graphs:
            for surface in free_surfaces(dict(modules=count,edges=graph)):
                if surface[0]==count-1:yield count,graph,surface


def contact_replacements(n,edges,surfaces,maximum_modules=8):
    """Lengthen one grasp branch while keeping the contact count unchanged."""
    if n>=maximum_modules:return
    reserved=set(map(tuple,surfaces))
    first=[graph for graph in grow_tree(n,edges)
           if (graph[-1]['a'],graph[-1]['port_a']) in reserved]
    levels=[(n+1,first)]
    if n+2<=maximum_modules:
        levels.append((n+2,[graph for parent in first for graph in grow_tree(n+1,parent)
                            if graph[-1]['a']==n]))
    for count,graphs in levels:
        if count>maximum_modules:continue
        for graph in graphs:
            consumed=(graph[n-1]['a'],graph[n-1]['port_a'])
            replaced=list(map(tuple,surfaces)).index(consumed)
            for surface in free_surfaces(dict(modules=count,edges=graph)):
                if surface[0]==count-1:yield count,graph,surface,replaced


def additional_sites(shape,source,points):
    """Sample actual surfaces, including upper, lower and lateral locations."""
    R=pin.Quaternion(source[3:]).matrix();sites=[]
    directions=[np.array([0.,0.,1.]),np.array([0.,0.,-1.]),*R.T,-R[:,0],-R[:,1],-R[:,2]]
    # Inclined side seeds include supports below the COM. The unchanged full
    # clearance screens, not a hemisphere exclusion, reject table interference.
    for axis in R.T:
        horizontal=axis.copy();horizontal[2]=0.
        if np.linalg.norm(horizontal)<1e-8:continue
        horizontal/=np.linalg.norm(horizontal)
        for sign in [-1.,1.]:
            for height in [-.5,.5]:directions.append(sign*np.sqrt(.75)*horizontal+np.array([0.,0.,height]))
    for direction in directions:
        point,normal=shape.ray_surface(R.T@direction)
        # Distinct material targets only; this is not a force-feasibility test.
        if min(np.linalg.norm(np.asarray(points)-point,axis=1))<.02:continue
        if any(np.linalg.norm(point-p)<1e-8 for p,_ in sites):continue
        sites.append((point,-normal))
    return sites


def extend_posture(draft,shape,source,destination,support_top, *, maximum_modules=8,fit_budget=12,replace_existing=False,deadline=None):
    """Rank geometry cheaply, then fit a bounded, attachment-diverse subset."""
    base=draft['plant'];n=base.meta['modules'];R=pin.Quaternion(source[3:]).matrix()
    sites=additional_sites(shape,source,draft['points']);pool=[];cache={}
    if replace_existing:
        options=((count,edges,surface,[(draft['points'][i],draft['normals'][i])],i)
            for count,edges,surface,i in contact_replacements(n,base.meta['edges'],draft['surfaces'],maximum_modules))
    else:
        options=((count,edges,surface,sites,None)
            for count,edges,surface in contact_extensions(n,base.meta['edges'],draft['surfaces'],maximum_modules))
    for count,edges,surface,targets,replaced in options:
        check_search_deadline(deadline)
        key=tuple((e['a'],e['port_a'],e['b'],e['port_b']) for e in edges)
        if key not in cache:
            model,meta=build_model(count,edges,shape.payload())
            q=copy_posture(base.pin,base.meta,draft['closed'],model,meta,source)
            data=model.createData();pin.framesForwardKinematics(model,data,q)
            cache[key]=(q,{s:data.oMf[meta['frames'][grasp_frame(*s)]].copy() for s in free_surfaces(meta)})
        q,frames=cache[key];T=frames[surface]
        for point,normal in targets:
            score=np.linalg.norm(T.translation-(R@point+source[:3]))+.15*np.linalg.norm(T.rotation[:,0]-R@normal)
            group=(count,edges[n-1]['a'] if count>n else surface[0],surface[1][:3])
            if replaced is not None:group=(*group,replaced)
            pool.append((float(score),group,edges,surface,point,normal,q,replaced))
    # Round robin across attachment location and pad axis before spending a
    # second fit on one group. Cheap geometric ranking does not prove rejection.
    buckets={}
    for row in sorted(pool,key=lambda row:row[0]):buckets.setdefault(row[1],[]).append(row)
    selected=[]
    while buckets and len(selected)<fit_budget:
        for group in list(buckets):
            selected.append(buckets[group].pop(0))
            if not buckets[group]:del buckets[group]
            if len(selected)>=fit_budget:break
    obstacles=support_obstacles(shape,source,destination,support_top)
    for _,group,edges,surface,point,normal,q,replaced in selected:
        check_search_deadline(deadline)
        model,meta=build_model(group[0],edges,shape.payload())
        if replaced is None:
            surfaces=draft['surfaces']+[surface];points=np.vstack([draft['points'],point]);normals=np.vstack([draft['normals'],normal])
        else:
            surfaces=list(draft['surfaces']);surfaces[replaced]=surface
            points=np.asarray(draft['points']).copy();normals=np.asarray(draft['normals']).copy()
        closed,fit=fit_contacts(model,meta,q,surfaces,points@R.T+source[:3]+.001*(normals@R.T),normals@R.T,
            floor_clearance=.03,base_rotation=True,obstacles=obstacles,max_evaluations=65)
        score=fit['position_error_m']+.15*fit['normal_error']
        yield dict(plant=SimpleNamespace(pin=model,meta=meta),closed=closed,surfaces=surfaces,
            points=points,normals=normals,fit=fit,score=score,search=dict(enumerated=len(pool),fitted_budget=len(selected),attachment=group,replaced_contact=replaced))


def lengthened_postures(parents,shape,source,destination,support_top,*,maximum_modules,fit_budget,beam,deadline=None):
    """Spend one bounded fit budget across successive branch extensions.

    Extending one side can leave the other side unreachable. Keep growing the
    best current poses instead of requiring a short branch to be final.
    """
    frontier=list(parents)[:beam];remaining=fit_budget
    while frontier and remaining:
        frontier=[p for p in frontier if p['plant'].meta['modules']<maximum_modules]
        if not frontier:break
        levels=maximum_modules-min(p['plant'].meta['modules']for p in frontier)
        quota=max(1,int(np.ceil(remaining/levels)));candidates=[]
        for index,parent in enumerate(frontier):
            budget=quota//len(frontier)+(index<quota%len(frontier))
            if budget:
                for candidate in extend_posture(parent,shape,source,destination,support_top,
                        maximum_modules=maximum_modules,fit_budget=budget,replace_existing=True,deadline=deadline):
                    candidates.append(candidate)
                    yield candidate
        remaining-=quota
        candidates.sort(key=lambda row:row['score'])
        # Caller may have fully screened these while consuming the iterator.
        passed=[c for c in candidates if 'plan'in c]
        frontier=(passed or candidates)[:beam]
        frontier=[dict(c,plant=c['plan'].plant,closed=c['plan'].closed_q,
                       surfaces=c['plan'].surfaces,points=c['plan'].points,normals=c['plan'].normals)
                  if 'plan'in c else c for c in frontier]


def expanded_postures(shape,source,destination,support_top, *, mode='yaw',axis_world=None,
                      contact_counts=(2,3,4),maximum_modules=8,beam=2,fit_budget=12,screen_parents=False,contact_tilt=0.,deadline=None,seed_state=None):
    """Advance one fitted branch at a time, sharing task-local parents.

    Four contacts can grow as soon as one parent is available. A four-only
    search still generates its three-contact parents, without selecting them.
    """
    counts=tuple(sorted(set(contact_counts)))
    if not counts or min(counts)<2 or max(counts)>4:raise ValueError('supported contact counts are 2, 3, 4')
    if not 3<=maximum_modules<=8 or beam<1 or fit_budget<1:raise ValueError('invalid search budget')
    if len(counts)>1 or (counts==(4,) and seed_state is None):
        from collections import deque
        state={} if seed_state is None else seed_state
        intermediate=set(counts)|({3} if 4 in counts else set())
        pending=deque((count,expanded_postures(shape,source,destination,support_top,
            mode=mode,axis_world=axis_world,contact_counts=(count,),maximum_modules=maximum_modules,
            beam=beam,fit_budget=fit_budget,screen_parents=screen_parents,contact_tilt=contact_tilt,
            deadline=deadline,seed_state=state)) for count in sorted(intermediate))
        while pending:
            count,stream=pending.popleft()
            try:draft=next(stream)
            except StopIteration:continue
            pending.append((count,stream))
            if count in counts:yield draft
        return
    count,=counts
    state={} if seed_state is None else seed_state
    def screen(draft):
        check_search_deadline(deadline)
        if not screen_parents:return
        if draft['fit']['position_error_m']>.015 or draft['fit']['normal_error']>.15:
            draft['rejection']='coarse_reach';return
        try:
            draft['plan']=finish_expanded_posture(draft,shape,source,destination,support_top)
        except SearchBudgetExpired:raise
        except ValueError as error:draft['rejection']=str(error)
    def grown_seed(plan):
        return dict(plant=plan.plant,closed=plan.closed_q,surfaces=plan.surfaces,
                    points=plan.points,normals=plan.normals)
    check_search_deadline(deadline)
    # A single task/approach has the same two-pad seed for every count stream.
    # Branch fitting copies its posture; no completed-structure library is read.
    # The caller creates a separate cache for each task/axis/tilt/mode.
    if 'draft' in state:
        draft=state['draft']
    else:
        draft=plan_seed(shape,source,destination,support_top,mode=mode,axis_world=axis_world,geometry_only=True,contact_tilt=contact_tilt)
        screen(draft)
        state['draft']=draft
    if count==2:yield draft
    parents=[grown_seed(draft['plan']) if 'plan' in draft else draft]
    if count==2:
        for candidate in lengthened_postures(parents,shape,source,destination,support_top,
                maximum_modules=maximum_modules,fit_budget=fit_budget,beam=beam,deadline=deadline):
            screen(candidate)
            yield candidate
        return
    candidates=[]
    if count==3:
        state['three']=candidates;state['three_done']=False
        try:
            for candidate in extend_posture(parents[0],shape,source,destination,support_top,
                    maximum_modules=maximum_modules,fit_budget=fit_budget,deadline=deadline):
                screen(candidate);candidates.append(candidate);yield candidate
        finally:state['three_done']=True
    else:
        used=[]
        while len(used)<beam:
            check_search_deadline(deadline)
            three=state.get('three',[]);done=state.get('three_done',False)
            # A partial structure need not carry the payload: another contact
            # or branch can change its load capacity. Use reachability to grow
            # it; all final material/load/path screens still apply to children.
            available=[c for c in three if c['fit']['position_error_m']<=.015
                       and c['fit']['normal_error']<=.15]
            if done and not available:available=sorted(three,key=lambda c:c['score'])
            remaining=[c for c in available if all(c is not old for old in used)]
            if not remaining:
                if done:break
                yield dict(draft,scheduler_pending=True)
                continue
            parent=remaining[0];used.append(parent)
            parent=grown_seed(parent['plan']) if 'plan' in parent else parent
            for candidate in extend_posture(parent,shape,source,destination,support_top,
                    maximum_modules=maximum_modules,fit_budget=fit_budget,deadline=deadline):
                screen(candidate);candidates.append(candidate);yield candidate
    passed=[c for c in candidates if 'plan' in c]
    parents=(passed or sorted(candidates,key=lambda c:c['score']))[:beam]
    parents=[grown_seed(c['plan']) if 'plan' in c else c for c in parents]
    for candidate in lengthened_postures(parents,shape,source,destination,support_top,
            maximum_modules=maximum_modules,fit_budget=fit_budget,beam=beam,deadline=deadline):
        screen(candidate);yield candidate


def finish_expanded_posture(draft,shape,source,destination,support_top):
    """Apply exactly the existing full material/load/path screens."""
    start=time.perf_counter();meta=draft['plant'].meta
    p=make_plant(meta['modules'],meta['edges'],shape.payload(),mesh_cache=CONTACT_CACHE,floor=True)
    points=draft['points'].copy();normals=draft['normals'].copy()
    opened,closed,fit=refine_material_posture(p,shape,source,draft['closed'].copy(),draft['surfaces'],points,normals,
        obstacles=support_obstacles(shape,source,destination,support_top))
    plan=finish_plan(p,shape,source,destination,support_top,opened,closed,draft['surfaces'],points,normals,fit,start)
    plan.fit.update(structure_scope='two-pad seed; branch lengthening at fixed contact count; free pads or legal one/two-link additions at any free port',
                    contact_count=len(plan.surfaces),expanded_search=draft.get('search',{}))
    return plan
