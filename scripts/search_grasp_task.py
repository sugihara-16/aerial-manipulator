"""Development model rollouts that reject a candidate and resume generation.

These runs begin with a prepared structure. They select a plan for subsequent
continuous assembly/task validation; they are not autonomous assembly results.
"""
import json
from pathlib import Path
import time
from amsrr_det.evidence import save_sources
from amsrr_det.structure_search import grasp_candidates
from scripts.run_grasp_task import execute


def search(shape,source,destination,support_top,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);save_sources(output)
    request=dict(payload=shape.payload(),source=source.tolist(),destination=destination.tolist(),
                 support_top=support_top,planner='generated_with_physical_rejection',timestep=.001)
    (output/'request.json').write_text(json.dumps(request,indent=2))
    started=time.perf_counter();trials=[];selected=None;last=None;failure=None
    try:
        for plan in grasp_candidates(shape,source,destination,support_top):
            folder=output/f'candidate_{len(trials):02d}';folder.mkdir(exist_ok=True)
            (folder/'request.json').write_text(json.dumps(request,indent=2))
            last=execute(plan,folder,request)
            trials.append(dict(directory=folder.name,task_success=last['task_success'],reason=last['reason'],
                               phase=last['phase'],simulation_s=last['simulation_s'],wall_s=last['wall_s'],
                               cumulative_geometry_planning_s=plan.wall_s))
            (output/'trials.json').write_text(json.dumps(trials,indent=2))
            print(json.dumps(trials[-1]),flush=True)
            if last['task_success']:selected=folder.name;break
    except ValueError as error:
        failure=dict(reason=str(error),evidence=getattr(error,'evidence',None))
    result=dict(last or {},task_success=selected is not None,
                scope='candidate_selection_by_prepared_structure_rollouts_no_assembly',
                selected_candidate=selected,trials=trials,planning_failure=failure,
                total_selection_wall_s=time.perf_counter()-started)
    if selected is None:result['reason']='no_physically_accepted_candidate'
    (output/'result.json').write_text(json.dumps(result,indent=2))
    return result
