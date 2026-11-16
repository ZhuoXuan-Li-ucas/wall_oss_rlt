"""RTC request contract and the numerical helpers; the guided sampler itself is
checked on the target GPU against the vanilla path (zero guidance is bit-exact)."""
import numpy as np
import pytest
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from wall_rlt.data.ar5 import ee_absolute_window, rotvec_to_rot6d
from wall_rlt.inference.protocol import (CAMERA_KEYS, ProtocolError, decode_request, json_bytes,
                                         make_request, strict_json_loads)
from wall_rlt.inference.runtime import (RTC_MAX_GUIDANCE, ee_relative_targets, rtc_guidance_weight,
                                        rtc_prefix_weights)


def ee_observation():
    return {'instruction': 'insert the bolt into the sleeve', 'state': [0.3, -0.2, 0.4] + [1., 0., 0., 0., 1., 0.] + [0.8],
            'images': {key: Image.new('RGB', (28, 28), (30, 60, 90)) for key in CAMERA_KEYS}}


def rtc(steps=12, delay=5):
    return {'prefix_targets': [[0.3, -0.2, 0.4, 0.01, 0.02, 0.03, 0.9]] * steps, 'inference_delay': delay}


def test_rtc_roundtrips_and_is_omitted_when_absent():
    value = make_request(ee_observation(), request_id='r', rtc=rtc())
    request = decode_request(strict_json_loads(json_bytes(value)), state_dim=10)
    assert request.rtc == {**rtc(), 'method': 'inpaint'}                  # default method
    guided = make_request(ee_observation(), request_id='g', rtc={**rtc(), 'method': 'guidance'})
    assert decode_request(guided, state_dim=10).rtc['method'] == 'guidance'
    plain = make_request(ee_observation(), request_id='p')
    assert 'rtc' not in plain and decode_request(plain, state_dim=10).rtc is None


@pytest.mark.parametrize('fault', [
    {'prefix_targets': [], 'inference_delay': 0},
    {'prefix_targets': [[0.] * 7] * 33, 'inference_delay': 0},
    {'prefix_targets': [[0.] * 6], 'inference_delay': 0},
    {'prefix_targets': [[0.] * 7], 'inference_delay': -1},
    {'prefix_targets': [[0.] * 7], 'inference_delay': 33},
    {'prefix_targets': [[0.] * 7], 'inference_delay': 1.0},
    {'prefix_targets': [[0.] * 7]},
    {'prefix_targets': [[0.] * 7], 'inference_delay': 0, 'extra': 1},
    {'prefix_targets': [[0.] * 7], 'inference_delay': 0, 'method': 'hard'},
])
def test_invalid_rtc_is_rejected(fault):
    value = make_request(ee_observation(), request_id='r')
    value['rtc'] = fault
    with pytest.raises(ProtocolError):
        decode_request(value, state_dim=10)


def test_rtc_requires_the_ee_contract():
    value = make_request(ee_observation(), request_id='r')
    value['rtc'], value['state'] = rtc(), [0.] * 8
    with pytest.raises(ProtocolError, match='EE'):
        decode_request(value, state_dim=8)


def test_prefix_weights_match_lerobot_linear_schedule():
    # LeRobot get_prefix_weights(LINEAR): ones, linspace(1,0,n+2)[1:-1], zeros
    assert rtc_prefix_weights(3, 7, 10).tolist() == pytest.approx(
        [1, 1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0])
    assert rtc_prefix_weights(9, 7, 10).tolist() == [1.] * 7 + [0.] * 3   # delay clipped to horizon
    assert rtc_prefix_weights(0, 32, 32)[0] < 1 and rtc_prefix_weights(0, 32, 32)[-1] > 0


def test_guidance_weight_is_clipped_at_both_ends():
    assert rtc_guidance_weight(0.0) == rtc_guidance_weight(0.999) == RTC_MAX_GUIDANCE
    assert rtc_guidance_weight(0.5) == pytest.approx(2.0)                # (1-t)/t / r2 at t=1/2


def test_relative_targets_invert_absolute_window():
    rng = np.random.default_rng(0)
    rotvec = Rotation.random(random_state=1).as_rotvec()
    state = np.concatenate(([0.3, -0.2, 0.4], rotvec_to_rot6d(rotvec[None])[0], [0.7]))
    absolute = np.concatenate((rng.normal(0, 0.05, (32, 3)) + state[:3],
                               (Rotation.from_rotvec(rng.normal(0, 0.2, (32, 3))) *
                                Rotation.from_rotvec(rotvec)).as_rotvec(),
                               rng.uniform(0, 1, (32, 1))), axis=1)
    back = ee_absolute_window(ee_relative_targets(absolute, state), state)
    assert np.abs(back[:, :3] - absolute[:, :3]).max() < 1e-12
    assert (Rotation.from_rotvec(back[:, 3:6]).inv() * Rotation.from_rotvec(absolute[:, 3:6])).magnitude().max() < 1e-9
    assert np.array_equal(back[:, 6], absolute[:, 6])                    # gripper stays continuous
