"""Guarded external fixed joints between independent module articulations."""
import numpy as np
import pinocchio as pin
from pxr import UsdPhysics,Gf,Sdf
from amsrr_det.model import PORTS
from amsrr_det.assembly import mechanism
from .physx_tree import quat


def author_seams(stage,paths,plant,edges):
    joints=[];mate=pin.rpy.rpyToMatrix(0,0,np.pi)
    for index,edge in enumerate(edges):
        first,second=[plant.pin.frames[plant.meta['frames'][f"m{edge[module]}/{PORTS[edge[port]]}"]] for module,port in [('a','port_a'),('b','port_b')]]
        joint=UsdPhysics.FixedJoint.Define(stage,f'/World/Seams/seam_{index}')
        joint.CreateBody0Rel().SetTargets([paths[first.parentJoint]]);joint.CreateBody1Rel().SetTargets([paths[second.parentJoint]])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float,first.placement.translation)));joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*map(float,second.placement.translation)))
        joint.CreateLocalRot0Attr().Set(quat(first.placement.rotation));joint.CreateLocalRot1Attr().Set(quat(second.placement.rotation@mate))
        joint.CreateExcludeFromArticulationAttr().Set(True);joint.CreateJointEnabledAttr().Set(False)
        # Pair exclusions are controlled separately, not by disabling collision
        # for every descendant of these joints.
        joint.CreateCollisionEnabledAttr().Set(True);joints.append(joint)
    return joints


def observe_connected(sim,view):
    """Controller coordinates over an unchanged physical forest state."""
    physical_q,physical_v=sim.observation();q=pin.neutral(view.pin);v=np.zeros(view.pin.nv)
    bases={b['module']:b for b in sim.meta['bases']};actuators={a['name']:a for a in sim.meta['actuators']}
    for base in view.meta['bases']:
        source=bases[base['module']];q[base['q']:base['q']+7]=physical_q[source['q']:source['q']+7];v[base['v']:base['v']+6]=physical_v[source['v']:source['v']+6]
    for a in view.meta['actuators']:
        source=actuators[a['name']];q[a['q']]=physical_q[source['q']];v[a['v']]=physical_v[source['v']]
    if view.meta['object']:
        obj=view.meta['object'];source=sim.meta['object'];q[obj['q']:obj['q']+7]=physical_q[source['q']:source['q']+7];v[obj['v']:obj['v']+6]=physical_v[source['v']:source['v']+6]
    return q,v


def actuator_order(source,target):
    names=[a['name'] for a in source['actuators']]
    return np.array([names.index(a['name']) for a in target['actuators']])


class SeamFilters:
    """Allow only the currently approached pair and already attached pairs."""
    def __init__(self,stage,plant,edges):
        self.pairs=[];self.groups={}
        for edge in edges:
            pair=[plant.mj.body(mechanism(edge[key],edge[port])).id for key,port in [('a','port_a'),('b','port_b')]]
            self.pairs.append(pair)
            for body in pair:self.groups[body]=UsdPhysics.CollisionGroup.Get(stage,f'/World/Groups/body_{body}')
        self.baseline={body:set(group.GetFilteredGroupsRel().GetTargets()) for body,group in self.groups.items()}
        for first,second in self.pairs:
            self.baseline[first].discard(self.groups[second].GetPath());self.baseline[second].discard(self.groups[first].GetPath())
        # Group relations are initialized once. In this PhysX version their
        # runtime edits leave collision behavior unchanged; explicit pair
        # filtering was independently checked to update during simulation.
        for body,paths in self.baseline.items():self.groups[body].GetFilteredGroupsRel().SetTargets(sorted(paths,key=str))
        self.body_paths={};self.relations={}
        for body in self.groups:
            frame=plant.pin.frames[plant.meta['frames'][plant.mj.body(body).name]]
            path=f'/World/Robot/link_{frame.parentJoint}';self.body_paths[body]=path
            self.relations[body]=UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(path)).CreateFilteredPairsRel()
        self.current={body:None for body in self.groups}

    def apply(self,indices):
        targets={body:set() for body in self.groups}
        for index in indices:
            first,second=self.pairs[index]
            targets[first].add(self.body_paths[second]);targets[second].add(self.body_paths[first])
        # Updating unchanged USD relations can trigger costly physics rebuilds.
        # Publish the changed pair relations together, with identical targets.
        with Sdf.ChangeBlock():
            for body,paths in targets.items():
                value=tuple(sorted(paths))
                if value!=self.current[body]:
                    self.relations[body].SetTargets(list(value));self.current[body]=value
