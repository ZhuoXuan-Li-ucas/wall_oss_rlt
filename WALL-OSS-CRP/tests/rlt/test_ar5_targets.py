import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wall_rlt.data.ar5 import aligned_targets, validate_pose_deltas, JOINT_NEXT, EE_DELTA_NEXT


def episode():
    state = np.arange(32, dtype=np.float32).reshape(4, 8) / 100
    state[:, 2] = np.float32(np.deg2rad(7))
    state[:, 4] = np.float32(np.deg2rad(-93.2))
    # Measured aperture is deliberately different from the binary command.
    state[:, 7] = [1, .9, .386, .386]
    action = np.zeros((4, 7), dtype=np.float32)
    action[:, 6] = [1, 1, 0, 0]
    action[1:, :6] = np.arange(18).reshape(3, 6) / 1000
    return state, action


def test_joint_target_shifts_once_and_uses_command_not_opening():
    state, action = episode()
    targets, observations = aligned_targets(state, action)
    np.testing.assert_array_equal(observations, state[:-1])
    np.testing.assert_array_equal(targets[:, :7], state[1:, :7])
    np.testing.assert_array_equal(targets[:, 7], [1, 0, 0])
    assert targets.shape == (3, 8)
    assert observations.shape == (3, 8)
    # j3 and j5 remain present, at their recorded values, with no rescaling.
    np.testing.assert_array_equal(targets[:, [2, 4]], state[1:, [2, 4]])


def test_ee_target_shifts_from_previous_to_next_frame():
    state, action = episode()
    targets, observations = aligned_targets(state, action, EE_DELTA_NEXT)
    np.testing.assert_array_equal(targets, action[1:])
    np.testing.assert_array_equal(observations, state[:-1])
    assert targets.shape == (3, 7)
    assert np.any(targets[0, :6])  # Raw action[0] zero label is not reused.


@pytest.mark.parametrize('mode', [JOINT_NEXT, EE_DELTA_NEXT])
def test_two_frames_have_exactly_one_real_target_and_no_source_mutation(mode):
    state, action = episode()
    state, action = state[:2], action[:2]
    state_before, action_before = state.copy(), action.copy()
    targets, observations = aligned_targets(state, action, mode)
    assert len(targets) == len(observations) == 1
    assert targets.dtype == observations.dtype == np.float32
    targets.fill(17)
    observations.fill(-5)
    np.testing.assert_array_equal(state, state_before)
    np.testing.assert_array_equal(action, action_before)


@pytest.mark.parametrize('mode', [JOINT_NEXT, EE_DELTA_NEXT])
@pytest.mark.parametrize('length', [0, 1])
def test_no_label_is_fabricated_for_short_episode(mode, length):
    state, action = episode()
    with pytest.raises(ValueError, match='at least two frames'):
        aligned_targets(state[:length], action[:length], mode)


@pytest.mark.parametrize('fault', ['state_shape', 'action_shape', 'length', 'nan', 'inf', 'gripper', 'mode'])
def test_invalid_recording_fails(fault):
    state, action = episode()
    mode = JOINT_NEXT
    if fault == 'state_shape':
        state = state[:, :7]
    elif fault == 'action_shape':
        action = action[:, :6]
    elif fault == 'length':
        action = action[:-1]
    elif fault == 'nan':
        state[0, 0] = np.nan
    elif fault == 'inf':
        action[0, 0] = np.inf
    elif fault == 'gripper':
        action[2, 6] = .386
    elif fault == 'mode':
        mode = 'guess'
    with pytest.raises(ValueError):
        aligned_targets(state, action, mode)


def pose_episode():
    ee = np.array([[.1, .2, .5, .6, -.4, .2],
                   [.11, .19, .51, .62, -.38, .21],
                   [.13, .2, .52, .64, -.39, .23]], dtype=np.float32)
    rotations = Rotation.from_rotvec(ee[:, 3:].astype(np.float64))
    action = np.zeros((len(ee), 7), dtype=np.float32)
    action[:, 6] = [1, 0, 0]
    action[1:, :3] = np.diff(ee[:, :3], axis=0)
    action[1:, 3:6] = (rotations[1:] * rotations[:-1].inv()).as_rotvec()
    return ee, action


def test_pose_delta_verifies_noncommuting_base_frame_rotations():
    ee, action = pose_episode()
    report = validate_pose_deltas(ee, action)
    assert report['max_position_error'] < 1e-6
    assert report['max_rotation_error'] < 1e-6
    assert report['initial_delta_max'] == 0


@pytest.mark.parametrize('fault', ['component_subtraction', 'body_frame', 'position', 'first_frame'])
def test_pose_delta_rejects_wrong_rotation_math_and_time_origin(fault):
    ee, action = pose_episode()
    if fault == 'component_subtraction':
        action[1:, 3:6] = np.diff(ee[:, 3:], axis=0)
    elif fault == 'body_frame':
        rotations = Rotation.from_rotvec(ee[:, 3:].astype(np.float64))
        action[1:, 3:6] = (rotations[:-1].inv() * rotations[1:]).as_rotvec()
    elif fault == 'position':
        action[1, 0] += .01
    else:
        action[0, 0] = .01
    with pytest.raises(ValueError, match='pose deltas disagree'):
        validate_pose_deltas(ee, action)


def test_pose_delta_handles_axis_angle_branch_crossing():
    ee = np.zeros((2, 6))
    ee[:, 5] = [np.pi-.01, -np.pi+.01]
    action = np.zeros((2, 7))
    action[:, 6] = 1
    action[1, 5] = .02
    assert validate_pose_deltas(ee, action)['max_rotation_error'] < 1e-12


@pytest.mark.parametrize('atol', [0, -1, np.inf, np.nan])
def test_pose_delta_requires_valid_tolerance(atol):
    ee, action = pose_episode()
    with pytest.raises(ValueError, match='atol'):
        validate_pose_deltas(ee, action, atol=atol)
