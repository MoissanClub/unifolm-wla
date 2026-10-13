"""Read-only frame recovery from official WBT joint/pose sample pairs."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
import polars as pl
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--parquet', type=Path, required=True)
parser.add_argument('--urdf', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--samples', type=int, default=500)
args = parser.parse_args()
URDF = str(args.urdf)
df = pl.read_parquet(args.parquet)
df = df.filter(pl.col('observation.state.recomputed_ee_valid') == 1)
ix = np.unique(np.linspace(0, len(df) - 1, args.samples, dtype=int))
rows = df[ix].to_dicts()
model = pin.buildModelFromUrdf(URDF)
data = model.createData()
joints = {name: model.joints[model.getJointId(name)].idx_q for name in model.names[1:]}
arm_prefix = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow']


def recover(side, wrist_names, waist_names, kind='observation.state'):
    offsets = []
    arm_names = arm_prefix + list(wrist_names)
    waist_key = f'{kind}.waist_state_joint' if kind == 'observation.state' else 'action.waist_action_joint'
    for row in rows:
        q = pin.neutral(model)
        for name, angle in zip(waist_names, row[waist_key]):
            q[joints[f'waist_{name}_joint']] = angle
        for name, angle in zip(arm_names, row[f'{kind}.{side}_arm']):
            q[joints[f'{side}_{name}_joint']] = angle
        pin.forwardKinematics(model, data, q)
        wrist = data.oMi[model.getJointId(f'{side}_wrist_yaw_joint')].homogeneous.copy()
        base_pose = row['observation.state.state_base_pose' if kind == 'observation.state' else 'action.action_base_pose']
        base = np.eye(4)
        base[:3, :3] = Rotation.from_quat(base_pose[3:]).as_matrix()
        base[:3, 3] = base_pose[:3]
        wrist = base @ wrist
        pose = row[f'{kind}.{side}_ee_pose_gripper_base']
        target = np.eye(4)
        target[:3, :3] = Rotation.from_euler('xyz', pose[3:]).as_matrix()
        target[:3, 3] = pose[:3]
        offsets.append(np.linalg.inv(wrist) @ target)
    offsets = np.asarray(offsets)
    R = Rotation.from_matrix(offsets[:, :3, :3]).mean()
    p = offsets[:, :3, 3].mean(axis=0)
    pe = np.linalg.norm(offsets[:, :3, 3] - p, axis=1)
    re = (R.inv() * Rotation.from_matrix(offsets[:, :3, :3])).magnitude()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R.as_matrix(), p
    return {
        'side': side, 'kind': kind, 'wrist_order': wrist_names, 'waist_order': waist_names,
        'wrist_yaw_T_policy': T.tolist(),
        'translation_residual_m': dict(zip(['median', 'p95', 'max'], np.percentile(pe, [50, 95, 100]).tolist())),
        'rotation_residual_rad': dict(zip(['median', 'p95', 'max'], np.percentile(re, [50, 95, 100]).tolist())),
    }


results = []
for side in ['left', 'right']:
    side_results = []
    for wrist_names in itertools.permutations(['wrist_roll', 'wrist_pitch', 'wrist_yaw']):
        for waist_names in itertools.permutations(['yaw', 'roll', 'pitch']):
            side_results.append(recover(side, wrist_names, waist_names))
    side_results.sort(key=lambda r: r['translation_residual_m']['p95'] + r['rotation_residual_rad']['p95'])
    results += side_results[:3]
    print(json.dumps(side_results[:3], indent=2), flush=True)
    best = side_results[0]
    action = recover(side, best['wrist_order'], best['waist_order'], 'action')
    results.append(action)
    print(json.dumps(action, indent=2), flush=True)
args.out.write_text(json.dumps({'urdf': URDF, 'samples': len(rows), 'results': results}, indent=2))
