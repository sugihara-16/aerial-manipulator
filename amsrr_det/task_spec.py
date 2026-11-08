"""Versioned task contract shared by the console and the execution adapters.

The supported profiles are explicit: unknown fields and unsupported overrides
are errors, never silent requests that the current controller would ignore.
"""
import copy
import math


def field(key,label,default,minimum=None,maximum=None,step=None,choices=None):
    return dict(key=key,label=label,default=default,min=minimum,max=maximum,step=step,choices=choices)


COMMON=[field('seed','Random seed',1,0,2147483647,1)]
CATALOG={
 'grasp':dict(label='Grasp and transport',fields=[
  field('shape','Object shape','box',choices=['box','elongated_box','cylinder','ellipsoid','triangular_prism','tapered_box']),
  field('sx','Size X (m)',.3,.05,2.,.01),field('sy','Size Y (m)',.3,.05,2.,.01),field('sz','Size Z (m)',.2,.05,2.,.01),
  field('mass','Mass (kg)',.6,.05,20.,.05),field('x','Start X (m)',0.,-5.,5.,.05),field('y','Start Y (m)',0.,-5.,5.,.05),
  field('yaw','Initial yaw (°)',0.,-180.,180.,5.),field('roll','Initial roll (°)',0.,-180.,180.,5.),
  field('height','Support height (m)',.2,.05,1.,.05),field('dx','Transport X (m)',.4,-3.,3.,.05),field('dy','Transport Y (m)',0.,-3.,3.,.05)]),
 'valve':dict(label='Valve rotation',fields=[
  field('torque','Resisting torque (Nm)',1.5,.1,60.,.5),field('tilt','Axis tilt from vertical (°)',0.,0.,180.,15.),
  field('azimuth','Axis azimuth (°)',0.,-180.,180.,15.),field('angle','Initial bar angle (°)',0.,-180.,180.,15.),
  field('height','Valve center height (m)',1.5,1.,3.,.1)]),
 'walking':dict(label='Airborne start, landing, and walking',fields=[
  field('modules','Modules',4,3,8,1),field('speed','Walking speed (m/s)',.22,.2,.5,.02),
  field('distance','Travel distance (m)',5.,5.,20.,.5),field('curvature','Curvature (1/m; positive turns left)',0.,-.5,.5,.05)])}


def defaults(kind):
    if kind not in CATALOG:raise ValueError('Unsupported task')
    return {f['key']:f['default'] for f in CATALOG[kind]['fields']+COMMON}


def check_values(kind,values):
    expected=defaults(kind)
    if not isinstance(values,dict) or set(values)!=set(expected):raise ValueError('Missing or unsupported input fields')
    result={}
    for f in CATALOG[kind]['fields']+COMMON:
        x=values[f['key']]
        if f['choices']:
            if x not in f['choices']:raise ValueError(f"{f['label']}: Invalid selection")
        else:
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):raise ValueError(f"{f['label']}: A finite number is required")
            if not f['min']<=x<=f['max']:raise ValueError(f"{f['label']}: Enter a value between {f['min']} and {f['max']}")
            if f['key'] in ('seed','modules'):
                if int(x)!=x:raise ValueError(f"{f['label']}: An integer is required")
                x=int(x)
        result[f['key']]=x
    if kind=='grasp':
        if result['shape']=='cylinder' and abs(result['sx']-result['sy'])>1e-10:raise ValueError('Cylinder sizes X and Y must be equal')
        if math.hypot(result['dx'],result['dy'])<.05:raise ValueError('Transport distance must be at least 5 cm')
    return result


def make_spec(kind,values,mode='task'):
    v=check_values(kind,values)
    if mode not in ('task','assembly','integrated'):raise ValueError('Unsupported execution mode')
    spec=dict(version=1,task=kind,scene=dict(frame='world',length_unit='m',angle_unit='rad',gravity_m_s2=[0.,0.,-9.80665],ground='horizontal_rigid'),
        robot=dict(model='amsrr',module_count=dict(min=1,max=8)),contacts={},goal={},
        objectives=dict(priority=['required_physical_margins','minimize_rotor_load_proxy'],profile='current_v1'),
        acceptance=dict(profile=kind+'_v1'),execution=dict(mode=mode,initial_state='assembled_airborne',seed=v['seed'],planning_budget_s=15.,control_period_s=.01,delay='measured',wall_limit_s=600.))
    if kind=='grasp':
        spec['scene']['object']=dict(shape=v['shape'],size_m=[v['sx'],v['sy'],v['sz']],mass_kg=v['mass'],support_height_m=v['height'],initial=dict(xy_m=[v['x'],v['y']],rpy_rad=[math.radians(v['roll']),0.,math.radians(v['yaw'])],height='resting_on_support'))
        spec['contacts']=dict(robot_surface='anchor',target='object',allowed_region='object_surface',model='unilateral_friction',count=dict(min=1,max=8))
        spec['goal']=dict(quantity='object_pose',frame='world',translation_m=[v['dx'],v['dy'],0.],orientation='preserve_initial',minimum_transport_clearance_m=.10,release=True)
    elif kind=='valve':
        spec['scene']['object']=dict(model='valve_with_bar_v1',center_m=[0.,0.,v['height']],axis=dict(tilt_rad=math.radians(v['tilt']),azimuth_rad=math.radians(v['azimuth'])),initial_angle_rad=math.radians(v['angle']),resistance_torque_nm=v['torque'])
        spec['contacts']=dict(robot_surface='anchor',target='object.bar',allowed_region=['local_y_positive','local_y_negative'],model='unilateral_friction',count=dict(min=1,max=4))
        spec['goal']=dict(quantity='joint_angle',target='object.hinge',change_rad=math.pi,release=True)
    else:
        spec['robot']['module_count']=dict(fixed=v['modules'])
        spec['contacts']=dict(robot_surface='dock',target='ground',allowed_region='upper_surface',model='unilateral_friction',count=dict(min=2,max=8),support='alternating_feet')
        spec['goal']=dict(quantity='body_path',frame='world',speed_m_s=v['speed'],distance_m=v['distance'],curvature_m_inv=v['curvature'],initial_xy_m=[0.,0.],initial_heading_rad=0.)
        spec['objectives']['energy_interval']='walking'
    if mode in ('assembly','integrated'):
        spec['execution']['initial_state']='individual_ground_random'
        spec['execution']['assembly']=dict(nearest_distance_m=[1.,2.],deadline_after_hover_s=30.,final_hover_s=1.)
        if mode=='assembly':spec['acceptance']['profile']='assembly_v1'
    return spec


def validate_spec(spec):
    """Decode and round-trip the supported contract to reject ignored fields."""
    try:
        if not isinstance(spec,dict) or spec['version']!=1:raise ValueError('Task specification version must be 1')
        kind=spec['task'];v=defaults(kind);v['seed']=spec['execution']['seed']
        goal=spec['goal']
        if kind=='grasp':
            o=spec['scene']['object'];p=o['initial']
            v.update(shape=o['shape'],sx=o['size_m'][0],sy=o['size_m'][1],sz=o['size_m'][2],mass=o['mass_kg'],height=o['support_height_m'],x=p['xy_m'][0],y=p['xy_m'][1],roll=math.degrees(p['rpy_rad'][0]),yaw=math.degrees(p['rpy_rad'][2]),dx=goal['translation_m'][0],dy=goal['translation_m'][1])
        elif kind=='valve':
            o=spec['scene']['object'];v.update(torque=o['resistance_torque_nm'],height=o['center_m'][2],tilt=math.degrees(o['axis']['tilt_rad']),azimuth=math.degrees(o['axis']['azimuth_rad']),angle=math.degrees(o['initial_angle_rad']))
        else:v.update(modules=spec['robot']['module_count']['fixed'],speed=goal['speed_m_s'],distance=goal['distance_m'],curvature=goal['curvature_m_inv'])
        expected=make_spec(kind,v,spec['execution']['mode'])
        def same(a,b,path='task'):
            if isinstance(b,dict):
                if not isinstance(a,dict) or a.keys()!=b.keys():raise ValueError(f'{path}: Missing or unsupported fields')
                for k in b:same(a[k],b[k],path+'.'+k)
            elif isinstance(b,list):
                if not isinstance(a,list) or len(a)!=len(b):raise ValueError(f'{path}: Invalid number of elements')
                for i in range(len(b)):same(a[i],b[i],path+f'[{i}]')
            elif isinstance(b,(int,float)) and not isinstance(b,bool):
                if isinstance(a,bool) or not isinstance(a,(int,float)) or not math.isfinite(a) or not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12):raise ValueError(f'{path}: Unsupported value')
            elif type(a)!=type(b) or a!=b:raise ValueError(f'{path}: Unsupported value')
        same(spec,expected)
        return expected
    except (KeyError,IndexError,TypeError,OverflowError) as e:raise ValueError('Invalid task specification structure or value: '+str(e)) from e


def form_values(spec):
    """Recover GUI values through validated common fields, not saved form data."""
    spec=validate_spec(spec);kind=spec['task'];v=defaults(kind);v['seed']=spec['execution']['seed'];g=spec['goal']
    if kind=='grasp':
        o=spec['scene']['object'];p=o['initial'];v.update(shape=o['shape'],sx=o['size_m'][0],sy=o['size_m'][1],sz=o['size_m'][2],mass=o['mass_kg'],height=o['support_height_m'],x=p['xy_m'][0],y=p['xy_m'][1],roll=math.degrees(p['rpy_rad'][0]),yaw=math.degrees(p['rpy_rad'][2]),dx=g['translation_m'][0],dy=g['translation_m'][1])
    elif kind=='valve':
        o=spec['scene']['object'];v.update(torque=o['resistance_torque_nm'],height=o['center_m'][2],tilt=math.degrees(o['axis']['tilt_rad']),azimuth=math.degrees(o['axis']['azimuth_rad']),angle=math.degrees(o['initial_angle_rad']))
    else:v.update(modules=spec['robot']['module_count']['fixed'],speed=g['speed_m_s'],distance=g['distance_m'],curvature=g['curvature_m_inv'])
    return v


def requirements(spec):
    """Shared requirement record, consumed by the per-task model adapters."""
    s=validate_spec(spec)
    stages={'grasp':['approach','grasp','lift','transport','place','release'],
            'valve':['approach','load','rotate','release'],
            'walking':['hover','morph','land','walk']}[s['task']]
    return dict(scene=copy.deepcopy(s['scene']),resources=copy.deepcopy(s['robot']),contact_domain=copy.deepcopy(s['contacts']),
        motion=copy.deepcopy(s['goal']),objectives=copy.deepcopy(s['objectives']),acceptance=copy.deepcopy(s['acceptance']),
        stages=(['takeoff','assemble','stable_hover'] if s['execution']['mode']=='assembly' else ['takeoff','assemble','stable_hover','transfer']+stages if s['execution']['mode']=='integrated' else stages),
        design_stages=stages,timing=copy.deepcopy(s['execution']))
