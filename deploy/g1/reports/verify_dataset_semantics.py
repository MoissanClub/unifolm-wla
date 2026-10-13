"""Numerical convention checks against the released BrainCo dataset."""
import argparse
import json
from pathlib import Path
import numpy as np
import polars as pl
import pinocchio as pin
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--parquet', type=Path, required=True)
parser.add_argument('--urdf', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--fk-samples', type=int, default=1000)
args = parser.parse_args()
df = pl.read_parquet(args.parquet)
def values(key): return np.array(df[key].to_list())
def poses(v, quat=False):
    T = np.broadcast_to(np.eye(4), (len(v), 4, 4)).copy()
    T[:, :3, 3] = v[:, :3]
    T[:, :3, :3] = (Rotation.from_quat(v[:, 3:]) if quat else Rotation.from_euler('xyz', v[:, 3:])).as_matrix()
    return T
def summary(v):return dict(zip(['median','p95','max'], np.percentile(v,[50,95,100]).tolist()))
report={'rows':len(df),'episodes':df['episode_index'].n_unique(),'checks':{}}
for kind,hand,base,torso in [
    ('observation.state','hand_state','state_base_pose','state_torso'),
    ('action','hand_cmd','action_base_pose','action_torso')]:
    raw=values(f'{kind}.{hand}')
    fig=np.concatenate([values(f'{kind}.left_fig6d'),values(f'{kind}.right_fig6d')],axis=1)
    report['checks'][kind+'_fig6d_minus_1_minus_raw_maxabs']=float(np.max(np.abs(fig-(1-raw))))
    Bt=poses(values(f'{kind}.{torso}'))
    for side in ('left','right'):
        Ee=poses(values(f'{kind}.{side}_ee_pose_gripper_base'))
        Te=poses(values(f'{kind}.{side}_ee_pose_gripper_torso'))
        pred=Bt@Te
        report['checks'][kind+'_'+side+'_torso_composition_position_m']=summary(np.linalg.norm(Ee[:,:3,3]-pred[:,:3,3],axis=1))
        report['checks'][kind+'_'+side+'_torso_composition_rotation_rad']=summary(Rotation.from_matrix(np.swapaxes(Ee[:,:3,:3],1,2)@pred[:,:3,:3]).magnitude())
gravity=Rotation.from_quat(values('observation.state.state_base_pose')[:,3:]).inv().apply([0,0,-1])
reported=values('observation.state.state_base_rot')
report['checks']['gravity_matches_base_pose']=summary(np.linalg.norm(gravity-reported[:,:3],axis=1))
report['checks']['base_rot_last3_absmax']=float(np.max(np.abs(reported[:,3:])))

# Fit the yaw-joint origin difference separately from a rigid end-tool offset.
model=pin.buildModelFromUrdf(str(args.urdf))
data=model.createData()
jointidx={name:model.joints[model.getJointId(name)].idx_q for name in model.names[1:]}
names=['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']
sample=df[np.unique(np.linspace(0,len(df)-1,args.fk_samples,dtype=int))].to_dicts()
for side in ('left','right'):
    diffs=[];angles=[]
    for row in sample:
        q=pin.neutral(model)
        for name,v in zip(['yaw','roll','pitch'],row['observation.state.waist_state_joint']):q[jointidx[f'waist_{name}_joint']]=v
        for name,v in zip(names,row[f'observation.state.{side}_arm']):q[jointidx[f'{side}_{name}_joint']]=v
        pin.forwardKinematics(model,data,q)
        wrist=data.oMi[model.getJointId(f'{side}_wrist_yaw_joint')].homogeneous.copy()
        base=poses(np.array([row['observation.state.state_base_pose']]),quat=True)[0]
        target=poses(np.array([row[f'observation.state.{side}_ee_pose_gripper_base']]))[0]
        diff=np.linalg.inv(base@wrist)@target
        diffs.append(diff[:3,3]);angles.append(row[f'observation.state.{side}_arm'][-1])
    diffs=np.asarray(diffs);angles=np.asarray(angles)
    matrices=[]
    for angle in angles:matrices.append(np.concatenate([np.eye(3),Rotation.from_euler('z',-angle).as_matrix()],axis=1))
    A=np.concatenate(matrices,axis=0);y=diffs.reshape(-1)
    # z offsets are not individually identifiable; x/y are.
    sol=np.linalg.lstsq(A,y,rcond=None)[0]
    error=(A@sol-y).reshape(-1,3)
    report['checks'][side+'_tool_offset_and_yaw_origin_difference_m']=sol.tolist()
    report['checks'][side+'_two_offset_fit_residual_m']=summary(np.linalg.norm(error,axis=1))

rotations=Rotation.from_quat(values('observation.state.state_base_pose')[:,3:])
angular=(rotations[:-1].inv()*rotations[1:]).as_rotvec()*30
within_episode=np.diff(df['episode_index'].to_numpy())==0
report['checks']['raw_body_angular_velocity_max_abs_error_rad_s']=float(np.max(np.abs(reported[1:,3:][within_episode]-angular[within_episode])))
print(json.dumps(report,indent=2))
args.out.write_text(json.dumps(report,indent=2))
