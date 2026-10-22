"""Freeze varied task inputs before testing; contains no robot structures."""
import argparse,json
from pathlib import Path
import numpy as np
import pinocchio as pin
import yaml
from amsrr_det.objects import ObjectShape
from amsrr_det.evaluation_inputs import resting_pose


def generate(config,seed):
    rng=np.random.default_rng(seed);cases=[]
    for family in config['families']:
        for yaw_bin,(low,high) in enumerate(config['yaw_bins_rad']):
            for repeat in range(config['cases_per_family_yaw']):
                size=rng.uniform(*config['object_dimensions_m'],size=3)
                if family=='cylinder':size[1]=size[0]
                elif family=='elongated_box':size[0]=rng.uniform(.40,.50);size[1:]=rng.uniform(.18,.22,size=2)
                mass=float(rng.uniform(*config['object_mass_kg']));shape=ObjectShape(family,size,mass)
                yaw=rng.uniform(low,high);roll=0. if repeat%2==0 else np.pi/2
                seed_pose=np.r_[np.zeros(3),pin.Quaternion(pin.rpy.rpyToMatrix(roll,0,0)).coeffs()]
                source,rest=resting_pose(shape,seed_pose);top=float(rng.uniform(*config['support_height_m']))
                source[3:]=pin.Quaternion(pin.rpy.rpyToMatrix(0,0,yaw)@pin.Quaternion(source[3:]).matrix()).coeffs()
                rest['applied_world_yaw_rad']=float(yaw)
                source[:2]=rng.uniform(*config['source_xy_m'],size=2);source[2]+=top
                direction=rng.uniform(-np.pi,np.pi);distance=rng.uniform(*config['transport_distance_m'])
                destination=source.copy();destination[:2]+=distance*np.array([np.cos(direction),np.sin(direction)])
                # Classify the resulting orientation, not the possibly unstable
                # seed from which the supported-COM search started.
                vertical=abs(pin.Quaternion(source[3:]).matrix()[2]);axis=int(np.argmax(vertical))
                if family=='ellipsoid':resting='curved_stable'
                elif family=='cylinder':resting='standing' if axis==2 else 'lying'
                else:resting='large_face' if axis==int(np.argmin(size)) else 'small_or_sloped_face'
                cases.append(dict(id=f'{family}_y{yaw_bin}_{repeat}',payload=shape.payload(),source=source.tolist(),destination=destination.tolist(),support_top=top,
                                  strata=dict(family=family,yaw_bin=yaw_bin,family_x_yaw_bin=f'{family}:{yaw_bin}',resting_orientation=resting,
                                              position_bin=f'{int(source[0]>=0)}:{int(source[1]>=0)}',size_bin='large' if max(size)>=.34 else 'small',mass_bin='heavy' if mass>=.7 else 'light'),
                                  resting_pose_generation=rest))
    return cases


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--development',action='store_true');args=parser.parse_args()
    config_path=Path('configs/evaluation/grasp_transport.yaml');config=yaml.safe_load(config_path.read_text());seed=config['seed_development' if args.development else 'seed_holdout']
    cases=generate(config,seed)
    if len(cases)!=config['holdout_cases']:raise ValueError('panel count mismatch')
    # Never silently replace an already frozen panel.
    with args.output.open('x') as stream:json.dump(dict(role='development' if args.development else 'heldout',seed=seed,configuration=config,cases=cases),stream,indent=2)
    print(json.dumps(dict(output=str(args.output),cases=len(cases),role='development' if args.development else 'heldout')))
