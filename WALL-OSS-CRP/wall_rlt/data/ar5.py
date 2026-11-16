"""Versioned target definitions for the recorded AR5 + OmniPicker episodes.

See 数据集格式说明.md, sections 3, 4 and 7. Raw ``action[t, :6]``
describes measured flange motion from t-1 to t in the AR5 base frame. It
is not a future command. Raw ``state[:, :7]`` is measured j1..j7 in radians,
including the two approximately constant joints j3 and j5; ``state[:, 7]``
is measured gripper opening. The binary gripper command is ``action[:, 6]``.

No robot control, kinematic conversion, normalization or slot mapping occurs
here. The joint target variant learns the next measured joint configuration,
not an unavailable logged joint command.

The EE variant follows the public Wall-OSS-0.5 LeRobot layout (wall-x
``x2robot2lerobot.py``: state = frame i, action = frame i+1, both measured
follower poses): rows are base-frame flange position, 6D rotation (first two
rows of the rotation matrix, row-major) and gripper. Windows are made relative
to the observation exactly like wall-x ``LerobotDataset._to_relative_action``:
position difference, ``R_rel = R_abs @ R_state^T`` as 6D, gripper unchanged.
The gripper target stays the recorded binary command (user decision 2026-11-02).
"""
from __future__ import annotations

import math
import numpy as np


JOINT_NEXT = 'ar5_joint_next_v1'
EE_DELTA_NEXT = 'ar5_ee_delta_next_v1'
EE_REL6D_NEXT = 'ar5_ee_rel6d_next_v1'
TARGET_MODES = (JOINT_NEXT, EE_DELTA_NEXT, EE_REL6D_NEXT)
EE_ROW_DIM = 10   # position 3 + rotation 6D + gripper 1


def _matrix(value, width, name):
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f'{name} must have shape [T,{width}]')
    if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite numeric values')
    return array


def _raw_action(action):
    action = _matrix(action, 7, 'action')
    if not np.isin(action[:, 6], (0, 1)).all():
        raise ValueError('action gripper command must be binary 0/1, not measured opening')
    return action


def aligned_targets(state, action, mode=JOINT_NEXT, *, ee_pose=None):
    """Return ``(targets, observations)`` with exactly T-1 float32 rows.

    ``observations[t] = state[t]``. Joint targets contain j1..j7 from
    ``state[t+1]`` followed by the *command* ``action[t+1,6]``. The optional
    EE delta variant contains ``action[t+1,:7]`` (base-frame flange translation
    in meters, left-multiplied rotation vector in radians, and binary gripper).
    The EE 6D variant requires ``ee_pose`` and returns absolute rows
    ``[pos, rot6d, gripper]``: observations use ``ee_pose[t]`` and the measured
    opening ``state[t,7]``; targets use ``ee_pose[t+1]`` and the command
    ``action[t+1,6]``. Window-relative conversion happens later, per window.

    The final observation has no next-frame target and is excluded. Window
    construction may pad later, but must mark those padded labels invalid.
    Both returned arrays own their data and do not mutate source recordings.
    """
    if mode not in TARGET_MODES:
        raise ValueError(f'unsupported AR5 target mode: {mode}')
    state = _matrix(state, 8, 'state')
    action = _raw_action(action)
    if len(state) != len(action):
        raise ValueError('state/action frame counts must match')
    if len(state) < 2:
        raise ValueError('at least two frames are required for a next-frame target')
    if mode == EE_REL6D_NEXT:
        if ee_pose is None:
            raise ValueError(f'{mode} requires the recorded ee_pose')
        ee_pose = _matrix(ee_pose, 6, 'ee_pose').astype(np.float64, copy=False)
        if len(ee_pose) != len(state):
            raise ValueError('ee_pose/state frame counts must match')
        pose = np.concatenate((ee_pose[:, :3], rotvec_to_rot6d(ee_pose[:, 3:])), axis=1)
        targets = np.concatenate((pose[1:], action[1:, 6:7]), axis=1)
        observations = np.concatenate((pose[:-1], state[:-1, 7:8]), axis=1)
    elif ee_pose is not None:
        raise ValueError(f'{mode} does not use ee_pose')
    elif mode == JOINT_NEXT:
        targets = np.concatenate((state[1:, :7], action[1:, 6:7]), axis=1)
        observations = state[:-1]
    else:
        targets = action[1:]
        observations = state[:-1]
    targets = np.array(targets, dtype=np.float32, copy=True)
    observations = np.array(observations, dtype=np.float32, copy=True)
    if not np.isfinite(targets).all() or not np.isfinite(observations).all():
        raise ValueError('values are outside the finite float32 range')
    return targets, observations


def validate_pose_deltas(ee_pose, action, *, atol=1e-6):
    """Verify the recorded previous-to-current flange displacement on SO(3).

    Returns maximum componentwise position error (meters), maximum relative
    rotation angle error (radians), and maximum absolute initial delta.
    Raises ValueError on disagreement. Rotation-vector components are never
    subtracted to form a rotation delta, and equivalent axis-angle branches
    are accepted by comparing the rotations themselves.
    """
    from scipy.spatial.transform import Rotation

    if not math.isfinite(atol) or atol <= 0:
        raise ValueError('atol must be finite and positive')
    ee_pose = _matrix(ee_pose, 6, 'ee_pose').astype(np.float64, copy=False)
    action = _raw_action(action).astype(np.float64, copy=False)
    if len(ee_pose) != len(action) or len(ee_pose) < 2:
        raise ValueError('ee_pose/action require matching frame counts of at least two')
    initial = float(np.abs(action[0, :6]).max())
    position_error = float(np.abs(np.diff(ee_pose[:, :3], axis=0) - action[1:, :3]).max())
    rotations = Rotation.from_rotvec(ee_pose[:, 3:])
    expected = rotations[1:] * rotations[:-1].inv()
    recorded = Rotation.from_rotvec(action[1:, 3:6])
    rotation_error = float((recorded * expected.inv()).magnitude().max())
    result = {'max_position_error':position_error,
              'max_rotation_error':rotation_error,
              'initial_delta_max':initial}
    if any(error > atol for error in result.values()):
        raise ValueError(f'AR5 previous-to-current pose deltas disagree (atol={atol}): {result}')
    return result


def rotvec_to_rot6d(rotvec):
    """[N,3] rotation vectors -> [N,6] first two rows of the rotation matrix."""
    from scipy.spatial.transform import Rotation

    rotvec = _matrix(rotvec, 3, 'rotvec').astype(np.float64, copy=False)
    return Rotation.from_rotvec(rotvec).as_matrix()[:, :2, :].reshape(-1, 6)


def rot6d_to_matrix(rot6d):
    """[N,6] -> [N,3,3] by Gram-Schmidt, as wall-x ``so3_to_matrix_batch_nb``."""
    rot6d = _matrix(rot6d, 6, 'rot6d').astype(np.float64, copy=False)
    eps = 1e-12
    r1 = rot6d[:, :3] / (np.linalg.norm(rot6d[:, :3], axis=1, keepdims=True) + eps)
    r2 = rot6d[:, 3:] - np.sum(r1 * rot6d[:, 3:], axis=1, keepdims=True) * r1
    r2 = r2 / (np.linalg.norm(r2, axis=1, keepdims=True) + eps)
    return np.stack((r1, r2, np.cross(r1, r2)), axis=1)


def _ee_rows(rows, name):
    rows = _matrix(rows, EE_ROW_DIM, name).astype(np.float64, copy=False)
    if name == 'targets' and not np.isin(rows[:, 9], (0, 1)).all():
        raise ValueError('EE target gripper must be the binary 0/1 command')
    return rows


def ee_relative_window(targets, state):
    """Absolute ``[K,10]`` next poses -> rows relative to the ``[10]`` observation.

    Pass only valid (unpadded) rows. Matches wall-x ``_to_relative_action``.
    """
    targets = _ee_rows(targets, 'targets')
    state = _ee_rows(np.asarray(state).reshape(1, -1), 'state')[0]
    r_state = rot6d_to_matrix(state[None, 3:9])[0]
    r_rel = np.matmul(rot6d_to_matrix(targets[:, 3:9]), r_state.T)
    return np.concatenate((targets[:, :3] - state[None, :3],
                           r_rel[:, :2, :].reshape(-1, 6), targets[:, 9:10]), axis=1)


def ee_absolute_window(relative, state):
    """Inverse for inference: ``[K,10]`` relative rows -> ``[K,7]`` flange targets.

    Returns base-frame position (m), rotation vector (rad, ``R_abs = R_rel @
    R_state``) and the predicted gripper value, still continuous.
    """
    from scipy.spatial.transform import Rotation

    relative = _ee_rows(relative, 'relative')
    state = _ee_rows(np.asarray(state).reshape(1, -1), 'state')[0]
    r_abs = np.matmul(rot6d_to_matrix(relative[:, 3:9]), rot6d_to_matrix(state[None, 3:9])[0])
    return np.concatenate((relative[:, :3] + state[None, :3],
                           Rotation.from_matrix(r_abs).as_rotvec(), relative[:, 9:10]), axis=1)
