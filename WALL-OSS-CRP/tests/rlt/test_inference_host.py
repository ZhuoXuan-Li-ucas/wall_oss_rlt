"""Host readiness checks with simulated metadata; no target GPU is accessed."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('wall_rlt_inference_host_check',
                                             ROOT / 'scripts/check_wall_rlt_inference_host.py')
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


def gpu():
    return {'index': 0, 'name': 'NVIDIA GeForce RTX 5070 Ti', 'uuid': 'GPU-target',
            'memory_total_mib': 16303, 'memory_free_mib': 15000,
            'driver_version': '595.91.07', 'compute_capability': '12.0'}


@pytest.fixture
def static_runtime(monkeypatch):
    # Register the environment changes with monkeypatch before check_host writes
    # to os.environ, so surrounding tests retain their CUDA-disabled environment.
    for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'HF_HUB_OFFLINE'):
        monkeypatch.setenv(key, host.os.environ.get(key, ''))
    monkeypatch.setattr(host, 'query_gpus', lambda: [gpu()])
    monkeypatch.setattr(host, 'driver_cuda_capability', lambda: {'value': '13.2', 'used_as_nvcc_version': False})
    monkeypatch.setattr(host, 'inspect_nvcc', lambda: {'status': 'PASS', 'version': '12.8', 'path': '/fake/nvcc'})
    class NoCUDA:
        version = SimpleNamespace(cuda='12.8')
        @property
        def cuda(self):
            pytest.fail('default/static checks must not inspect or initialize Torch CUDA')
    modules = {'torch': NoCUDA()}
    monkeypatch.setattr(host, 'inspect_imports', lambda: (
        {name: {'status': 'PASS'} for name in host.IMPORTS}, modules))
    monkeypatch.setattr(host, 'select_wall_source', lambda explicit=None: {'status': 'PASS', 'mode': 'installed'})
    monkeypatch.setattr(host, 'inspect_native', lambda nvcc, wall: {'status': 'PASS', 'extensions': []})
    monkeypatch.setattr(host, 'run_kernel_checks', lambda *args: pytest.fail('kernels were not authorized'))
    return modules


def test_default_static_check_does_not_call_cuda_or_kernels(static_runtime):
    result = host.check_host()
    assert result['status'] == 'PASS' and result['static_checks_status'] == 'PASS'
    assert result['kernels']['status'] == 'NOT_RUN' and result['kernels']['executed'] is False
    assert result['readiness_scope'] == 'STATIC_ONLY'
    assert host.os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert result['full_model_loaded'] is False and result['robot_accessed'] is False


def test_explicit_kernel_check_runs_only_after_static_checks_pass(static_runtime, monkeypatch):
    calls = []
    def kernels(torch, selected):
        assert host.os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
        assert host.os.environ['CUDA_VISIBLE_DEVICES'] == '0'
        calls.append(selected['uuid'])
        return {'status': 'PASS', 'executed': True, 'test_origin': 'simulated_kernel_probe'}
    monkeypatch.setattr(host, 'run_kernel_checks', kernels)
    report = host.check_host(run_kernels=True)
    assert report['status'] == 'PASS' and calls == ['GPU-target']
    assert report['readiness_scope'] == 'SMALL_KERNELS_ONLY'


def test_other_gpu_is_blocked_without_a_kernel_attempt(static_runtime, monkeypatch):
    monkeypatch.setattr(host, 'query_gpus', lambda: [{**gpu(), 'name': 'NVIDIA GeForce RTX 3060 Ti',
                                                    'compute_capability': '8.6'}])
    result = host.check_host(run_kernels=True)
    assert result['status'] == 'BLOCKED' and 'selected_gpu' in result['failed_checks']
    assert '3060 Ti' in result['selected_gpu']['error']
    assert result['kernels']['executed'] is False


def test_missing_native_sm120_blocks_even_explicit_kernels(static_runtime, monkeypatch):
    monkeypatch.setattr(host, 'inspect_native', lambda *args: {
        'status': 'FAIL', 'extensions': [{'status': 'FAIL', 'missing_architectures': ['sm_120']}]})
    report = host.check_host(run_kernels=True)
    assert report['status'] == 'BLOCKED' and report['kernels']['executed'] is False
    assert 'native_extensions' in report['failed_checks']


@pytest.mark.parametrize('compiled,nvcc,driver,passed', [
    ('12.8', '12.8', '13.2', True), ('13.0', '13.0', '13.2', True),
    ('12.8', '13.2', '13.2', False), ('13.0', '12.8', '13.2', False),
    ('13.2', '13.2', '13.2', False), (None, '12.8', '13.2', False),
    ('13.0', '13.0', '12.8', False),
])
def test_driver_cuda_capability_is_not_a_substitute_for_nvcc(compiled, nvcc, driver, passed):
    report = host.check_toolchain(compiled, {'status': 'PASS', 'version': nvcc}, {'value': driver})
    assert (report['status'] == 'PASS') is passed
    assert report['driver_capability_is_toolkit_version'] is False


def test_nvidia_smi_inventory_and_capability_are_parsed_separately(monkeypatch):
    def output(command, timeout=20):
        if len(command) == 1:
            return '| NVIDIA-SMI 595.91.07 Driver Version: 595.91.07 CUDA Version: 13.2 |'
        return '0, NVIDIA GeForce RTX 5070 Ti, GPU-target, 16303, 15000, 595.91.07, 12.0\n'
    monkeypatch.setattr(host, 'command_output', output)
    records = host.query_gpus()
    assert records == [gpu()]
    assert host.select_target_gpu(records, 0)['uuid'] == 'GPU-target'
    capability = host.driver_cuda_capability()
    assert capability['value'] == '13.2' and capability['used_as_nvcc_version'] is False


@pytest.mark.parametrize('records,index', [([], 0), ([gpu()], 1),
    ([{**gpu(), 'compute_capability': '9.0'}], 0), ([{**gpu(), 'uuid': 'MIG-unknown'}], 0),
    ({'status': 'FAIL'}, 0)])
def test_ambiguous_missing_or_wrong_gpu_metadata_is_rejected(records, index):
    with pytest.raises(ValueError):
        host.select_target_gpu(records, index)


def test_nvcc_uses_actual_cuda_home_binary_and_not_driver_report(tmp_path, monkeypatch):
    cuda = tmp_path / 'cuda'; (cuda / 'bin').mkdir(parents=True)
    nvcc = cuda / 'bin/nvcc'; nvcc.write_text('fixture'); nvcc.chmod(0o755)
    monkeypatch.setenv('CUDA_HOME', str(cuda))
    calls = []
    def output(command, timeout=20):
        calls.append(command)
        return 'Cuda compilation tools, release 12.8, V12.8.93\n'
    monkeypatch.setattr(host, 'command_output', output)
    result = host.inspect_nvcc()
    assert result['version'] == '12.8' and result['selected_from'] == 'CUDA_HOME'
    assert calls == [[str(nvcc), '--version']]


def test_invalid_cuda_home_does_not_silently_fall_back_to_other_nvcc(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_HOME', str(tmp_path / 'missing'))
    with pytest.raises(FileNotFoundError, match='CUDA_HOME'):
        host.inspect_nvcc()


def test_no_worktree_can_use_installed_wall_package(tmp_path, monkeypatch):
    monkeypatch.setattr(host, 'ROOT', tmp_path)
    package = tmp_path / 'site-packages/wall_x/__init__.py'
    monkeypatch.setattr(host.importlib, 'import_module', lambda name: SimpleNamespace(__file__=str(package)))
    report = host.select_wall_source()
    assert report['status'] == 'PASS' and report['mode'] == 'installed'
    assert report['active_package'] == str(package.parent)


def test_explicit_source_refuses_missing_or_already_loaded_wrong_package(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        host.select_wall_source(tmp_path / 'missing')
    source = tmp_path / 'checkout'; (source / 'wall_x').mkdir(parents=True)
    (source / 'wall_x/__init__.py').write_text('')
    monkeypatch.setattr(host.sys, 'path', host.sys.path.copy())
    monkeypatch.setattr(host.importlib, 'import_module', lambda name: SimpleNamespace(__file__='/wrong/wall_x/__init__.py'))
    with pytest.raises(RuntimeError, match='different source'):
        host.select_wall_source(source)


def test_native_checker_requires_sm120_and_imports_same_binary(tmp_path, monkeypatch):
    tool = tmp_path / 'cuobjdump'; tool.write_text('fixture'); tool.chmod(0o755)
    binaries = {name: tmp_path / (name + '.so') for name in host.NATIVE_MODULES}
    for path in binaries.values(): path.write_bytes(b'fixture native bytes')
    calls = []
    def inspect(name, cuobjdump, required):
        calls.append((name, cuobjdump, required))
        return {'module': name, 'status': 'PASS', 'path': str(binaries[name]), 'elf_architectures': ['sm_120']}
    monkeypatch.setattr(host, 'native_helper', lambda: SimpleNamespace(inspect_extension=inspect))
    monkeypatch.setattr(host.importlib, 'import_module', lambda name: SimpleNamespace(__file__=str(binaries[name])))
    report = host.inspect_native({'path': str(tmp_path / 'nvcc')}, {'status': 'PASS'})
    assert report['status'] == 'PASS'
    assert len(calls) == 2 and all(call[2] == ['sm_120'] for call in calls)
    assert all(row['import_status'] == 'PASS' for row in report['extensions'])


def test_kernel_probe_checks_uuid_before_allocating_any_tensor():
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                           get_device_properties=lambda index: SimpleNamespace(uuid='GPU-wrong', major=12, minor=0))
    fake = SimpleNamespace(cuda=cuda)
    with pytest.raises(RuntimeError, match='UUID'):
        host.run_kernel_checks(fake, gpu())


def test_kernel_failure_is_recorded_without_claiming_full_model_pass(static_runtime, monkeypatch):
    def fail(*args):
        raise RuntimeError('compiled kernel cannot execute')
    monkeypatch.setattr(host, 'run_kernel_checks', fail)
    result = host.check_host(run_kernels=True)
    assert result['status'] == 'BLOCKED' and result['kernels']['status'] == 'FAIL'
    assert result['full_model_inference'] == 'NOT_RUN' and result['full_model_loaded'] is False


def test_cli_writes_report_and_nonzero_status_for_mismatch(tmp_path, static_runtime, monkeypatch):
    monkeypatch.setattr(host, 'query_gpus', lambda: [{**gpu(), 'name': 'NVIDIA RTX 3060 Ti'}])
    output = tmp_path / 'reports/host.json'
    assert host.main(['--output', str(output)]) == 1
    report = json.loads(output.read_text())
    assert report['status'] == 'BLOCKED' and report['target']['required_cubin'] == 'sm_120'
    assert report['requested_kernel_checks'] is False
    assert not output.with_name('host.json.tmp').exists()
