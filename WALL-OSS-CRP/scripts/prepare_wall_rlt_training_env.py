#!/usr/bin/env python3
"""Inspect the D0-derived training runtime without GPU use; install only by opt-in.

Default --check-only never installs packages. --install-overlay creates a NEW
system-site-packages venv from D0 and installs verified local wheels offline.
No command in this tool changes D0 or repairs its known declaration exceptions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CORE_PINS = {
    'torch': '2.10.0+cu128', 'torchvision': '0.25.0+cu128',
    'torchaudio': '2.10.0+cu128', 'flash-attn': '2.8.3',
    'transformers': '5.2.0', 'numpy': '1.26.4',
    'huggingface-hub': '1.33.0', 'packaging': '25.0',
}
TRAINING_PINS = {
    'hydra-core': '1.3.2', 'omegaconf': '2.3.1',
    'antlr4-python3-runtime': '4.9.3', 'torchdata': '0.11.0',
    'ray': '2.47.0', 'sortedcontainers': '2.4.0',
    'tensorboard': '2.20.0', 'pytest': '8.4.2',
    'virtualenv': '21.12.0',
}
NATIVE_SHA256 = {
    'flash_attn_2_cuda': '09c5dea022d0e88a1de3cd2563e676c59a3e0836f691467240fb7f63031d8f9f',
    'wall_x.model.core.ops._cuda_ext_bin': '2c2a4ae238834e1adc7e6777dc266f81da8f3ccb77a607d2d9bd445c72c044eb',
}
KNOWN_D0_DECLARATION_EXCEPTIONS = {
    'missing': ['cmake', 'deepdiff', 'diffusers', 'gymnasium', 'imageio',
                'jsonlines', 'pynput', 'pyserial', 'rerun-sdk', 'torchcodec'],
    'incompatible': ['lerobot 0.4.4 declares huggingface-hub[cli,hf-transfer]>=0.34.2,<0.36.0; D0 has 1.33.0'],
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(data)
    return value.hexdigest()


def versions(pins):
    result = {}
    for name, expected in pins.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        result[name] = {'expected': expected, 'actual': actual,
                        'status': 'PASS' if actual == expected else 'FAIL'}
    return result


def base_pins():
    result = {}
    for line in (ROOT / 'docker/runtime-pins.txt').read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            name, version = line.split('==', 1)
            result[name] = version
    result.update(CORE_PINS)
    return result


def require_d0():
    checks = versions(base_pins())
    errors = {name: check for name, check in checks.items() if check['status'] != 'PASS'}
    if sys.version_info[:2] != (3, 10):
        errors['python'] = {'expected': '3.10', 'actual': sys.version.split()[0]}
    import torch
    if torch.version.cuda != '12.8':
        errors['torch.version.cuda'] = {'expected': '12.8', 'actual': torch.version.cuda}
    if errors:
        raise ValueError(f'Current interpreter is not the locked D0 runtime: {errors}')


def verify_wheelhouse(directory):
    """Every wheel must appear in SHA256SUMS; reject traversal and symlinks."""
    directory = Path(directory).resolve()
    listed = {}
    for line in (directory / 'SHA256SUMS').read_text().splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'([0-9a-fA-F]{64}) [ *](.+)', line)
        if not match:
            raise ValueError('Invalid SHA256SUMS line')
        expected, filename = match.groups()
        relative = Path(filename)
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name in listed:
            raise ValueError('Unsafe or duplicate wheelhouse manifest entry')
        path = directory / relative
        if path.is_symlink() or not path.is_file() or digest(path) != expected.lower():
            raise ValueError(f'Wheelhouse checksum mismatch: {relative}')
        listed[relative.name] = expected.lower()
    wheels = {path.name for path in directory.glob('*.whl')}
    if not wheels or not wheels.issubset(listed):
        raise ValueError('Wheelhouse contains no wheels or contains unverified wheels')
    return listed


def install_overlay(destination, wheelhouse):
    """Create an isolated overlay, refusing an existing destination or wrong base."""
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError('Overlay destination must be new; existing environments are never altered')
    require_d0()
    verified = verify_wheelhouse(wheelhouse)
    # Inherit D0's audited pip/setuptools, instead of seeding older ensurepip wheels.
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages',
                    '--without-pip', str(destination)], check=True)
    python = destination / 'bin/python'
    command = [str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
               '--no-index', '--no-cache-dir', '--only-binary=:all:',
               '--find-links', str(Path(wheelhouse).resolve()),
               '-c', str(ROOT / 'configs/rlt/constraints-training.txt'),
               '-r', str(ROOT / 'configs/rlt/requirements-training.txt')]
    subprocess.run(command, check=True)
    return {'path': str(destination), 'wheels_verified': len(verified), 'command': command}


def native_tools():
    spec = importlib.util.spec_from_file_location('wall_rlt_native_arch_check',
                                                ROOT / 'docker/verify_native_architectures.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def locate_cuobjdump(value):
    found = shutil.which(value)
    if not found and value == 'cuobjdump':
        candidate = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')) / 'bin/cuobjdump'
        if candidate.is_file() and os.access(candidate, os.X_OK):
            found = str(candidate)
    return found


def copy_wall_native(source, wall_root, cuobjdump):
    """Copy ONLY the validated D0 binary into an explicitly selected source tree."""
    require_d0()
    source, wall_root = Path(source).resolve(), Path(wall_root).resolve()
    if digest(source) != NATIVE_SHA256['wall_x.model.core.ops._cuda_ext_bin']:
        raise ValueError('Source is not the validated D0 WALL native extension')
    expected = '_cuda_ext_bin' + sysconfig.get_config_var('EXT_SUFFIX')
    if source.name != expected:
        raise ValueError('Native extension filename does not match this Python ABI')
    if not cuobjdump:
        raise ValueError('cuobjdump is required before copying a native extension')
    run = subprocess.run([cuobjdump, '--all-fatbin', '--list-elf', str(source)],
                         capture_output=True, text=True, timeout=60, check=True)
    if 'sm_90' not in set(re.findall(r'\bsm_\d+[a-z]?\b', run.stdout)):
        raise ValueError('Native extension does not contain sm_90 cubin')
    target = wall_root / 'wall_x/model/core/ops' / source.name
    if not target.parent.is_dir() or not (target.parent / '_cuda_ext.py').is_file():
        raise ValueError('Destination must be an existing patched WALL source tree')
    if wall_root == (ROOT / 'wall-x').resolve():
        raise ValueError('Native copy requires an isolated patched source tree, not the base checkout')
    locked_commit = 'ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f'
    actual_commit = subprocess.run(['git', '-C', str(wall_root), 'rev-parse', 'HEAD'],
                                   capture_output=True, text=True, check=True).stdout.strip()
    actual_patch = subprocess.run(['git', '-C', str(wall_root), 'diff', '--binary', 'HEAD', '--'],
                                  capture_output=True, check=True).stdout
    if actual_commit != locked_commit or actual_patch != (ROOT / 'patches/wall-x-rlt-prefix.patch').read_bytes():
        raise ValueError('WALL destination does not match the locked base commit and RLT patch')
    # Resolve symlinked parents before enforcing that no installed package is altered.
    resolved_target = target.resolve()
    if not resolved_target.is_relative_to(wall_root):
        raise ValueError('Native destination escapes the selected source tree')
    site_dirs = [Path(p).resolve() for p in sys.path if 'site-packages' in p]
    if any(resolved_target.is_relative_to(directory) for directory in site_dirs):
        raise ValueError('Never overwrite a base/system site-packages native extension')
    if target.is_symlink():
        raise ValueError('Refusing a symlink native destination')
    if target.exists():
        if digest(target) != digest(source):
            raise ValueError('Different native binary already exists; no replacement performed')
        return {'status': 'UNCHANGED', 'path': str(target), 'sha256': digest(target)}
    with source.open('rb') as origin, target.open('xb') as output:
        shutil.copyfileobj(origin, output, length=4 * 1024**2)
    if digest(target) != NATIVE_SHA256['wall_x.model.core.ops._cuda_ext_bin']:
        raise ValueError('Copied native binary failed SHA256 validation')
    return {'status': 'COPIED', 'path': str(target), 'sha256': digest(target)}


def logger_smoke():
    """Exercise the actual selected RLinf logger, including event-file creation."""
    from omegaconf import OmegaConf
    from rlinf.utils.metric_logger import MetricLogger
    with tempfile.TemporaryDirectory(prefix='wall-rlt-logger-') as scratch:
        cfg = OmegaConf.create({'runner': {'logger': {
            'log_path': scratch, 'project_name': 'runtime-check',
            'experiment_name': 'cpu-smoke', 'logger_backends': ['tensorboard']}}})
        logger = MetricLogger(cfg)
        try:
            logger.log({'runtime_check/scalar': 1.0}, step=0)
        finally:
            logger.finish()
        files = list(Path(scratch).rglob('events.out.tfevents.*'))
        if not files or not all(path.stat().st_size > 0 for path in files):
            raise RuntimeError('TensorBoard logger did not create a nonempty event file')
        return {'status': 'PASS', 'events_written': len(files),
                'temporary_logs_removed': True, 'gpu_used': False}


def check_environment(wall_root, rlinf_root, cuobjdump):
    # Prevent device use and telemetry during imports. Do not initialize Ray.
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', WANDB_MODE='offline')
    for path in (ROOT, rlinf_root, wall_root):
        sys.path.insert(0, str(Path(path).resolve()))
    report = {
        'schema_version': 1, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'CPU runtime/import/logger/native architecture check; no training or GPU kernels',
        'python': sys.executable, 'python_version': sys.version.split()[0],
        'python_abi': {'actual': f'{sys.version_info.major}.{sys.version_info.minor}',
                       'expected': '3.10', 'status': 'PASS' if sys.version_info[:2] == (3, 10) else 'FAIL'},
        'sources': {'wall': str(wall_root), 'rlinf': str(rlinf_root)},
        'base_versions': versions(base_pins()), 'training_versions': versions(TRAINING_PINS),
        'imports': {}, 'native_extensions': [], 'gpu_used': False,
    }
    modules = ('torch', 'hydra', 'ray', 'torchdata.stateful_dataloader',
               'rlinf.scheduler.cluster.utils', 'rlinf.workers.sft.fsdp_vla_sft_worker',
               'wall_rlt.workers.sft')
    for name in modules:
        try:
            module = importlib.import_module(name)
            report['imports'][name] = {'status': 'PASS', 'path': getattr(module, '__file__', None)}
        except Exception as error:
            report['imports'][name] = {'status': 'FAIL', 'error': f'{type(error).__name__}: {error}'}
    try:
        import torch
        report['torch_cuda'] = {'actual': torch.version.cuda, 'expected': '12.8',
                                'status': 'PASS' if torch.version.cuda == '12.8' else 'FAIL'}
    except Exception as error:
        report['torch_cuda'] = {'status': 'FAIL', 'error': str(error)}
    try:
        report['tensorboard_logger'] = logger_smoke()
    except Exception as error:
        report['tensorboard_logger'] = {'status': 'FAIL', 'error': f'{type(error).__name__}: {error}'}
    for name, expected in NATIVE_SHA256.items():
        if cuobjdump:
            item = native_tools().inspect_extension(name, cuobjdump, ['sm_90'])
            if item.get('sha256') != expected:
                item.update(status='FAIL', expected_d0_sha256=expected,
                            integrity_error='Active native binary is not the validated D0 binary')
        else:
            item = {'module': name, 'status': 'FAIL', 'error': 'cuobjdump unavailable; architecture unverified'}
        try:
            importlib.import_module(name)
            item['import_status'] = 'PASS'
        except Exception as error:
            item.update(status='FAIL', import_status='FAIL', import_error=f'{type(error).__name__}: {error}')
        report['native_extensions'].append(item)
    run = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True, timeout=60)
    report['package_declarations'] = {
        'status': 'PASS' if run.returncode == 0 else 'FAIL',
        'returncode': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr,
        'known_d0_exceptions_not_repaired': KNOWN_D0_DECLARATION_EXCEPTIONS,
        'note': 'D0 has 11 known declaration exceptions; runtime checks do not waive pip check failures',
    }
    checks = [*report['base_versions'].values(), *report['training_versions'].values(),
              *report['imports'].values(), report['python_abi'], report['torch_cuda'], report['tensorboard_logger'],
              *report['native_extensions']]
    report['runtime_checks_status'] = 'PASS' if all(check['status'] == 'PASS' for check in checks) else 'FAIL'
    # A declaration failure must never become an overall PASS, even if narrowed runtime checks pass.
    report['status'] = 'PASS' if (report['runtime_checks_status'] == 'PASS' and
                                  report['package_declarations']['status'] == 'PASS') else 'BLOCKED'
    report['training_or_gpu_validation_completed'] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true', help='Default; no package installation')
    mode.add_argument('--install-overlay', type=Path, metavar='NEW_VENV')
    parser.add_argument('--wheelhouse', type=Path)
    parser.add_argument('--wall-root', type=Path, default=ROOT / '.cache/wall-x-rlt')
    parser.add_argument('--rlinf-root', type=Path, default=ROOT / '.cache/rlinf-rlt')
    parser.add_argument('--copy-wall-native-from', type=Path,
                        help='Explicitly copy the trusted D0 .so into --wall-root; never overwrite a different binary')
    parser.add_argument('--cuobjdump', default='cuobjdump')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.install_overlay:
            if not args.wheelhouse:
                parser.error('--install-overlay requires --wheelhouse with SHA256SUMS')
            installed = install_overlay(args.install_overlay, args.wheelhouse)
            command = [str(args.install_overlay.absolute() / 'bin/python'), str(Path(__file__).resolve()),
                       '--check-only', '--wall-root', str(args.wall_root),
                       '--rlinf-root', str(args.rlinf_root), '--cuobjdump', args.cuobjdump]
            if args.copy_wall_native_from:
                command += ['--copy-wall-native-from', str(args.copy_wall_native_from)]
            if args.output:
                command += ['--output', str(args.output)]
            print(json.dumps({'overlay_installation': installed}), flush=True)
            return subprocess.run(command).returncode
        cuobjdump = locate_cuobjdump(args.cuobjdump)
        copied = None
        if args.copy_wall_native_from:
            copied = copy_wall_native(args.copy_wall_native_from, args.wall_root, cuobjdump)
        report = check_environment(args.wall_root, args.rlinf_root, cuobjdump)
        if copied:
            report['explicit_native_copy'] = copied
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return 0 if report['status'] == 'PASS' else 1
    except Exception as error:
        print(json.dumps({'status': 'BLOCKED', 'error': f'{type(error).__name__}: {error}'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
