#!/usr/bin/env python3
"""Check the Ubuntu RTX 5070 Ti inference runtime without loading model weights.

Default checks read nvidia-smi/nvcc metadata, import the selected runtime and
inspect sm_120 cubins. Only explicit --run-kernels performs small CUDA/BF16 and
FlashAttention operations; it never loads a full model or accesses a robot.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TARGET = {'gpu_model': 'NVIDIA GeForce RTX 5070 Ti', 'compute_capability': '12.0',
          'required_cubin': 'sm_120', 'user_reported_memory_mib': 16303,
          'user_reported_driver': '595.91.07', 'user_reported_driver_cuda_capability': '13.2',
          'user_project_directory': '/home/crp-5070ti-01/yuhang_workspace/wall_oss_rlt'}
IMPORTS = {'torch': ('torch', '2.10.0'), 'transformers': ('transformers', '5.2.0'),
           'accelerate': ('accelerate', None), 'safetensors': ('safetensors', None),
           'av': ('av', None), 'PIL': ('Pillow', None), 'flash_attn': ('flash-attn', '2.8.3')}
SUPPORTED_COMPILED_CUDA = {'12.8', '13.0'}
NATIVE_MODULES = ('flash_attn_2_cuda', 'wall_x.model.core.ops._cuda_ext_bin')


def command_output(command, timeout=20):
    run = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if run.returncode:
        raise RuntimeError(f'{command[0]} failed with exit {run.returncode}: {run.stderr.strip()}')
    return run.stdout


def query_gpus():
    command = ['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,memory.free,driver_version,compute_cap',
               '--format=csv,noheader,nounits']
    records = []
    for row in csv.reader(command_output(command).splitlines()):
        if len(row) != 7:
            raise ValueError('nvidia-smi returned an unexpected GPU inventory format')
        index, name, uuid, total, free, driver, capability = [item.strip() for item in row]
        records.append({'index': int(index), 'name': name, 'uuid': uuid,
                        'memory_total_mib': int(total), 'memory_free_mib': int(free),
                        'driver_version': driver, 'compute_capability': capability})
    if not records or len({row['index'] for row in records}) != len(records):
        raise ValueError('nvidia-smi returned no GPUs or duplicate GPU indices')
    return records


def driver_cuda_capability():
    """This describes driver compatibility, not an installed CUDA toolkit."""
    raw = command_output(['nvidia-smi'])
    found = re.search(r'CUDA Version:\s*([0-9]+\.[0-9]+)', raw)
    return {'value': found.group(1) if found else None,
            'meaning': 'maximum CUDA version advertised by the driver; not nvcc or torch build CUDA',
            'used_as_nvcc_version': False}


def select_target_gpu(records, device_index):
    if not isinstance(records, list):
        raise ValueError('GPU inventory is unavailable; no target device can be selected')
    selected = [row for row in records if row['index'] == device_index]
    if len(selected) != 1:
        raise ValueError('requested physical GPU index was not found uniquely')
    result = dict(selected[0])
    if not re.search(r'\bRTX\s*5070\s*Ti\b', result['name'], re.IGNORECASE):
        raise ValueError(f"selected GPU is {result['name']}; this entry targets RTX 5070 Ti")
    if result['compute_capability'] != TARGET['compute_capability']:
        raise ValueError('selected GPU does not report compute capability 12.0')
    if not result['uuid'].startswith('GPU-'):
        raise ValueError('selected GPU must have a full GPU UUID')
    return result


def inspect_nvcc():
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home:
        path = Path(cuda_home) / 'bin/nvcc'
        source = 'CUDA_HOME'
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError('CUDA_HOME does not contain an executable bin/nvcc')
    else:
        found = shutil.which('nvcc')
        if not found:
            raise FileNotFoundError('nvcc is unavailable; nvidia-smi CUDA Version is not a toolkit')
        path, source = Path(found), 'PATH'
    raw = command_output([str(path), '--version'])
    found = re.search(r'\brelease\s+(\d+\.\d+)', raw)
    if not found:
        raise ValueError('cannot determine nvcc major.minor release')
    return {'status': 'PASS', 'path': str(path.resolve()), 'selected_from': source,
            'version': found.group(1), 'version_output': raw,
            'path_nvcc': shutil.which('nvcc'), 'cuda_home': cuda_home}


def inspect_imports():
    reports, modules = {}, {}
    for name, (distribution, expected) in IMPORTS.items():
        item = {'expected_version': expected}
        try:
            module = importlib.import_module(name)
            actual = importlib.metadata.version(distribution)
            item.update(status='PASS', version=actual, path=getattr(module, '__file__', None))
            if expected and actual.split('+', 1)[0] != expected:
                item.update(status='FAIL', error=f'expected {distribution} {expected}')
            modules[name] = module
        except Exception as error:
            item.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        reports[name] = item
    return reports, modules


def select_wall_source(explicit=None):
    """Prefer an explicit/cache checkout; installed patched wall_x is supported."""
    cache = ROOT / '.cache/wall-x-rlt'
    source = Path(explicit).resolve() if explicit is not None else (
        cache if (cache / 'wall_x/__init__.py').is_file() else None)
    mode = 'explicit' if explicit is not None else ('project_cache' if source else 'installed')
    if source:
        if not (source / 'wall_x/__init__.py').is_file():
            raise FileNotFoundError('selected --wall-source does not contain wall_x/__init__.py')
        sys.path.insert(0, str(source))
        importlib.invalidate_caches()
    module = importlib.import_module('wall_x')
    module_path = Path(module.__file__).resolve()
    if source and not module_path.is_relative_to(source / 'wall_x'):
        raise RuntimeError('active wall_x came from a different source than the selected checkout')
    return {'status': 'PASS', 'mode': mode, 'requested_source': str(source) if source else None,
            'active_package': str(module_path.parent), 'module_path': str(module_path)}


def check_toolchain(compiled_cuda, nvcc, driver_capability):
    result = {'status': 'FAIL', 'torch_compiled_cuda': compiled_cuda,
              'nvcc_cuda': nvcc.get('version'), 'supported_cuda_families': sorted(SUPPORTED_COMPILED_CUDA),
              'driver_cuda_capability': driver_capability.get('value'),
              'driver_capability_is_toolkit_version': False}
    if compiled_cuda not in SUPPORTED_COMPILED_CUDA:
        result['error'] = 'use the matched torch/native toolkit family CUDA 12.8 or CUDA 13.0'
    elif nvcc.get('status') != 'PASS' or nvcc.get('version') != compiled_cuda:
        result['error'] = 'nvcc major.minor must match the CUDA version compiled into torch'
    elif driver_capability.get('value') and tuple(map(int, driver_capability['value'].split('.'))) < tuple(map(int, compiled_cuda.split('.'))):
        result['error'] = 'driver advertises a CUDA capability below the selected torch runtime'
    else:
        result['status'] = 'PASS'
    return result


def native_helper():
    path = ROOT / 'docker/verify_native_architectures.py'
    spec = importlib.util.spec_from_file_location('inference_native_architectures', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_native(nvcc, wall_source):
    candidates = []
    if nvcc.get('path'):
        candidates.append(Path(nvcc['path']).parent / 'cuobjdump')
    if os.environ.get('CUDA_HOME'):
        candidates.append(Path(os.environ['CUDA_HOME']) / 'bin/cuobjdump')
    found = shutil.which('cuobjdump')
    if found:
        candidates.append(Path(found))
    tool = next((str(path) for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)
    helper = native_helper()
    result = []
    for name in NATIVE_MODULES:
        if name.startswith('wall_x') and wall_source.get('status') != 'PASS':
            result.append({'module': name, 'status': 'FAIL', 'import_status': 'NOT_RUN',
                           'error': 'the selected WALL source was not resolved'})
            continue
        if tool:
            item = helper.inspect_extension(name, tool, ['sm_120'])
        else:
            item = {'module': name, 'status': 'FAIL', 'error': 'cuobjdump unavailable; sm_120 coverage unverified'}
        try:
            module = importlib.import_module(name)
            item.update(import_status='PASS', imported_path=getattr(module, '__file__', None))
            if item.get('path') and Path(module.__file__).resolve() != Path(item['path']).resolve():
                item.update(status='FAIL', error='imported native binary differs from inspected binary')
        except Exception as error:
            item.update(status='FAIL', import_status='FAIL', import_error=f'{type(error).__name__}: {error}')
        item['required_cubin'] = 'sm_120'
        result.append(item)
    return {'cuobjdump': tool, 'extensions': result,
            'status': 'PASS' if all(item['status'] == 'PASS' for item in result) else 'FAIL'}


def _normalized_uuid(value):
    return str(value).lower().removeprefix('gpu-')


def run_kernel_checks(torch, selected):
    """The only function that queries Torch CUDA devices or allocates GPU tensors."""
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('kernel check must see exactly the explicitly selected GPU')
    props = torch.cuda.get_device_properties(0)
    if _normalized_uuid(getattr(props, 'uuid', '')) != _normalized_uuid(selected['uuid']):
        raise RuntimeError('Torch device 0 UUID differs from selected physical GPU')
    if (props.major, props.minor) != (12, 0):
        raise RuntimeError('Torch device 0 must have compute capability 12.0')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('selected GPU runtime does not support BF16')
    torch.cuda.set_device(0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        matrix = torch.ones((64, 64), device='cuda:0', dtype=torch.bfloat16)
        product = matrix @ matrix
        if not torch.isfinite(product).all().item() or not (product == 64).all().item():
            raise RuntimeError('BF16 matrix multiplication returned incorrect values')
        from flash_attn import flash_attn_func
        qkv = [torch.ones((1, 32, 2, 64), device='cuda:0', dtype=torch.bfloat16) for _ in range(3)]
        attention = flash_attn_func(*qkv, dropout_p=0., causal=False)
        if tuple(attention.shape) != (1, 32, 2, 64) or not torch.isfinite(attention).all().item():
            raise RuntimeError('FlashAttention returned invalid shape or nonfinite values')
        if not torch.allclose(attention.float(), torch.ones_like(attention, dtype=torch.float32), atol=.02, rtol=.02):
            raise RuntimeError('FlashAttention ones-input result is outside tolerance')
        torch.cuda.synchronize()
    return {'status': 'PASS', 'executed': True, 'logical_device': 0,
            'physical_device_index': selected['index'], 'uuid': selected['uuid'],
            'compute_capability': '12.0', 'bf16_matmul': 'PASS', 'flash_attention': 'PASS',
            'elapsed_seconds': time.perf_counter() - start,
            'full_model_loaded': False, 'wall_native_kernel_executed': False}


def check_host(*, wall_source=None, device_index=0, run_kernels=False):
    if type(device_index) is not int or device_index < 0:
        raise ValueError('device_index must be a nonnegative physical GPU index')
    # Configure visibility before importing torch or an extension that imports it.
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_index) if run_kernels else ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    report = {'schema_version': 1, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'RTX 5070 Ti runtime imports, toolchain and sm_120 coverage; optional small kernels only',
        'target': dict(TARGET), 'device_index': device_index, 'requested_kernel_checks': run_kernels,
        'python': {'executable': sys.executable, 'version': sys.version.split()[0], 'expected': '3.10',
                   'status': 'PASS' if sys.version_info[:2] == (3, 10) else 'FAIL'},
        'full_model_loaded': False, 'full_model_inference': 'NOT_RUN', 'robot_accessed': False,
        'system_modified': False, 'native_architecture_source': 'cuobjdump cubins, not driver CUDA capability'}
    def inspect(name, function):
        try:
            report[name] = function()
        except Exception as error:
            report[name] = {'status': 'FAIL', 'error': f'{type(error).__name__}: {error}'}
    inspect('gpu_inventory', query_gpus)
    inspect('driver_cuda_capability', driver_cuda_capability)
    inspect('selected_gpu', lambda: dict(select_target_gpu(report['gpu_inventory'], device_index), status='PASS'))
    inspect('nvcc', inspect_nvcc)
    report['imports'], modules = inspect_imports()
    inspect('wall_source', lambda: select_wall_source(wall_source))
    compiled_cuda = getattr(getattr(modules.get('torch'), 'version', None), 'cuda', None)
    report['cuda_toolchain'] = check_toolchain(compiled_cuda, report['nvcc'], report['driver_cuda_capability'])
    inspect('native_extensions', lambda: inspect_native(report['nvcc'], report['wall_source']))
    failures = [name for name in ('python', 'selected_gpu', 'nvcc', 'wall_source', 'cuda_toolchain', 'native_extensions')
                if report[name].get('status') != 'PASS']
    failures += ['imports.' + name for name, item in report['imports'].items() if item['status'] != 'PASS']
    report['static_checks_status'] = 'PASS' if not failures else 'FAIL'
    report['kernels'] = {'status': 'NOT_RUN', 'executed': False,
                         'reason': 'explicit --run-kernels was not requested'}
    if run_kernels:
        if failures:
            report['kernels']['reason'] = 'static compatibility checks must pass before any GPU kernel runs'
        else:
            try:
                report['kernels'] = run_kernel_checks(modules['torch'], report['selected_gpu'])
            except Exception as error:
                report['kernels'] = {'status': 'FAIL', 'executed': None,
                    'error': f'{type(error).__name__}: {error}',
                    'note': 'an explicitly requested kernel attempt may have initialized CUDA'}
                failures.append('kernels')
    report['status'] = 'PASS' if not failures else 'BLOCKED'
    report['failed_checks'] = failures
    report['readiness_scope'] = 'SMALL_KERNELS_ONLY' if report['kernels']['status'] == 'PASS' else 'STATIC_ONLY'
    report['next_steps'] = []
    if 'selected_gpu' in failures:
        report['next_steps'].append('run this check on the confirmed RTX 5070 Ti inference host')
    if any(name in failures for name in ('python', 'nvcc', 'cuda_toolchain')) or any(name.startswith('imports.') for name in failures):
        report['next_steps'].append('prepare Python 3.10, torch 2.10.0, transformers 5.2.0 and one matching CUDA 12.8 or 13.0 toolchain')
    if 'native_extensions' in failures or 'wall_source' in failures:
        report['next_steps'].append('build/install the selected WALL and FlashAttention extensions for sm_120 with the same torch/CUDA toolchain')
    report['next_steps'].append('full-model inference and memory acceptance require a separate explicit model check')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall-source', type=Path,
                        help='optional WALL checkout; otherwise use project cache when present, or installed wall_x')
    parser.add_argument('--device-index', type=int, default=0, help='physical nvidia-smi index, default 0')
    parser.add_argument('--run-kernels', action='store_true', help='explicitly execute small BF16/FlashAttention checks')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.device_index < 0:
        parser.error('--device-index must be nonnegative')
    report = check_host(wall_source=args.wall_source, device_index=args.device_index, run_kernels=args.run_kernels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'status': report['status'], 'report': str(args.output),
                      'static_checks_status': report['static_checks_status'],
                      'kernels': report['kernels']['status'], 'failed_checks': report['failed_checks'],
                      'full_model_inference': 'NOT_RUN'}, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
