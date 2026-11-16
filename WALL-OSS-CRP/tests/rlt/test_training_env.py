"""Guard environment isolation, offline-wheel integrity, and the real logger path."""
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prepare_wall_rlt_training_env',
                                            ROOT / 'scripts/prepare_wall_rlt_training_env.py')
env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env)


def test_overlay_refuses_existing_destination_before_touching_it(tmp_path, monkeypatch):
    marker = tmp_path / 'keep'
    marker.write_text('existing environment')
    monkeypatch.setattr(env, 'require_d0', lambda: pytest.fail('must refuse before inspecting base'))
    with pytest.raises(ValueError, match='must be new'):
        env.install_overlay(tmp_path, tmp_path / 'wheels')
    assert marker.read_text() == 'existing environment'


def test_overlay_refuses_wrong_base_before_creating_directory(tmp_path, monkeypatch):
    def reject():
        raise ValueError('wrong CUDA runtime')
    monkeypatch.setattr(env, 'require_d0', reject)
    destination = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='wrong CUDA'):
        env.install_overlay(destination, tmp_path / 'wheels')
    assert not destination.exists()


def test_wheelhouse_requires_checksums_for_every_wheel(tmp_path):
    wheel = tmp_path / 'a-1.0-py3-none-any.whl'
    wheel.write_bytes(b'wheel fixture')
    checksum = hashlib.sha256(wheel.read_bytes()).hexdigest()
    (tmp_path / 'SHA256SUMS').write_text(f'{checksum}  {wheel.name}\n')
    assert env.verify_wheelhouse(tmp_path) == {wheel.name: checksum}
    (tmp_path / 'unexpected-1.0-py3-none-any.whl').write_bytes(b'unverified')
    with pytest.raises(ValueError, match='unverified'):
        env.verify_wheelhouse(tmp_path)


@pytest.mark.parametrize('entry', ['../escape.whl', '/tmp/escape.whl'])
def test_wheelhouse_rejects_manifest_traversal(tmp_path, entry):
    (tmp_path / 'SHA256SUMS').write_text('0' * 64 + '  ' + entry + '\n')
    with pytest.raises(ValueError, match='Unsafe'):
        env.verify_wheelhouse(tmp_path)


def test_wheelhouse_rejects_tampering_and_symlinks(tmp_path):
    wheel = tmp_path / 'a-1.0-py3-none-any.whl'
    original = tmp_path / 'payload'
    original.write_bytes(b'original')
    wheel.symlink_to(original)
    checksum = hashlib.sha256(original.read_bytes()).hexdigest()
    (tmp_path / 'SHA256SUMS').write_text(f'{checksum}  {wheel.name}\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        env.verify_wheelhouse(tmp_path)
    wheel.unlink()
    wheel.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        env.verify_wheelhouse(tmp_path)


def test_native_copy_rejects_host_binary_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(env, 'require_d0', lambda: None)
    host_binary = tmp_path / '_cuda_ext_bin.cpython-310-x86_64-linux-gnu.so'
    host_binary.write_bytes(b'wrong platform binary')
    destination = tmp_path / 'source'
    with pytest.raises(ValueError, match='not the validated D0'):
        env.copy_wall_native(host_binary, destination, None)
    assert not destination.exists()


def test_actual_rlinf_tensorboard_logger_writes_and_closes_event(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / '.cache/rlinf-rlt'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    result = env.logger_smoke()
    assert result['status'] == 'PASS'
    assert result['events_written'] >= 1
    assert result['gpu_used'] is False
