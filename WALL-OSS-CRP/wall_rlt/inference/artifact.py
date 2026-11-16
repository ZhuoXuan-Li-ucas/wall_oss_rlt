"""Export a completed Stage 1 checkpoint for model-only inference.

The source is RLinf's consolidated ``model_state_dict/full_weights.pt`` plus
the project's checkpoint sidecar. No random initialization, shard merging or
optimizer deserialization is performed. A content hash establishes artifact
identity, not model quality or authorship of the source checkpoint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from wall_rlt.config import WallRLTConfig
from wall_rlt.data.contract import DataContract, canonical_hash
from wall_rlt._vendor.rlt_token_transformer import RLTTokenEncoder
from wall_rlt.model.ar5_adapter import variant_layouts
from wall_rlt.model.builder import SUPPORTED_ADAPTATIONS


KIND = 'wall-rlt-stage1-inference'
MANIFEST = 'manifest.json'
REQUIRED_ASSETS = {
    'adapted_wall_config.json', 'rlt_config.json', 'data_contract.json',
    'normalization.json', 'episodes.json', 'processor/config.json',
    'processor/preprocessor_config.json', 'processor/tokenizer_config.json',
    'wall_train_config.yml', 'source_hashes.json', 'training_config.json',
    'build_report.json',
}
FP32_WALL_MARKERS = (
    'input_layernorm', 'post_attention_layernorm', 'model.norm',
    'action_preprocessor', 'action_processor',
)


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def _relative(name):
    if not isinstance(name, str):
        raise ValueError('artifact path must be a string')
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or any(p in ('.', '..') for p in path.parts)
            or path.as_posix() != name or '\\' in name or ':' in name):
        raise ValueError('unsafe artifact path')
    return path


def _regular(root, name):
    path = Path(root) / _relative(name)
    if (not path.is_file() or path.is_symlink()
            or not path.resolve().is_relative_to(Path(root).resolve())
            or any(p.is_symlink() for p in path.parents if p != Path(root)
                   and p.is_relative_to(Path(root)))):
        raise ValueError(f'missing or linked artifact file: {name}')
    return path


def _inventory(root):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        if path.is_symlink():
            raise ValueError('artifact must not contain symlinks')
        if path.is_file():
            name = path.relative_to(root).as_posix()
            _relative(name)
            result[name] = path
    if len({name.casefold() for name in result}) != len(result):
        raise ValueError('case-colliding artifact files')
    return result


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _bound_contract(assets, expected_hash):
    contract = DataContract.load(assets / 'data_contract.json')
    # Checkpoint assets must be self-contained before relocation. No original
    # dataset is required on the inference host, but the frozen provenance is.
    if Path(contract.normalization_path) != (assets / 'normalization.json').resolve():
        raise ValueError('contract normalization must refer to the saved asset')
    if Path(contract.manifest_path) != (assets / 'episodes.json').resolve():
        raise ValueError('contract manifest must refer to the saved asset')
    if contract.fingerprint() != expected_hash:
        raise ValueError('checkpoint contract fingerprint mismatch')
    contract.require_trainable(allow_synthetic=contract.purpose == 'synthetic')
    if (contract.target_mode, contract.model_adapter) not in SUPPORTED_ADAPTATIONS:
        raise ValueError('inference export requires the declared AR5 joint8 or Wall-OSS-0.5 EE variant')
    rlt = WallRLTConfig.load(assets / 'rlt_config.json')
    wall = _read(assets / 'adapted_wall_config.json')
    if (rlt.input_dim != wall.get('hidden_size')
            or rlt.action_horizon != contract.action_horizon
            or rlt.model_action_dim != contract.model_action_dim):
        raise ValueError('WALL, RLT and data contract dimensions disagree')
    action_layout, state_layout = variant_layouts(contract.model_adapter)
    if (list(wall.get('dof_config', {}).items()) != list(action_layout.items())
            or list(wall.get('agent_pos_config', {}).items()) != list(state_layout.items())
            or wall.get('proj_with_mask') is not True
            or wall.get('use_state_string_representation') is not True):
        label = 'AR5 joint8' if contract.model_adapter == 'ar5_joint8_reinit_v1' else 'Wall-OSS-0.5 EE'
        raise ValueError(f'adapted WALL config does not declare the {label} layout')
    return contract, rlt, wall


def _encoder_schema(rlt):
    # Metadata-only construction, even for the full 2048-wide encoder.
    with torch.device('meta'):
        encoder = RLTTokenEncoder(**rlt.token_kwargs())
    return {name: tuple(value.shape) for name, value in encoder.state_dict().items()}


def _normalizer_values(contract):
    stats = _read(contract.normalization_path)['norm_stats']
    expected = {}
    for role, key, slots, width in (
        ('action', contract.action_key, contract.action_slots, contract.model_action_dim),
        ('propri', contract.state_key, contract.state_slots, contract.model_state_dim),
    ):
        lo, hi = torch.zeros(width), torch.ones(width)
        lo[slots] = torch.tensor(stats[key]['q01'], dtype=torch.float32)
        hi[slots] = torch.tensor(stats[key]['q99'], dtype=torch.float32)
        prefix = f'action_preprocessor.normalizer_{role}.'
        expected[prefix + 'min.' + contract.repo_id] = lo
        expected[prefix + 'delta.' + contract.repo_id] = hi - lo
    return expected


def _check_state(state, contract, rlt, config):
    if not isinstance(state, dict) or not state:
        raise ValueError('consolidated weights must be a nonempty tensor dictionary')
    wall, encoder = {}, {}
    decoder_count = 0
    for name, tensor in state.items():
        if (not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
                or tensor.layout != torch.strided or tensor.is_meta
                or tensor.device.type != 'cpu' or not tensor.is_floating_point()
                or '_flat_param' in name or '_fsdp_wrapped_module' in name):
            raise ValueError('unsupported weight key/type or unmerged shard')
        if not torch.isfinite(tensor).all():
            raise ValueError(f'non-finite checkpoint weight: {name}')
        if name.startswith('wall.'):
            key = name.removeprefix('wall.')
            if not key.startswith(('model.', 'visual.', 'lm_head.', 'action_preprocessor.')):
                raise ValueError(f'unsupported WALL weight key: {name}')
            wall[key] = tensor
        elif name.startswith('rlt_module.encoder.'):
            encoder[name.removeprefix('rlt_module.encoder.')] = tensor
        elif name.startswith('rlt_module.decoder.'):
            decoder_count += 1
        else:
            raise ValueError(f'unsupported Stage 1 weight key: {name}')
    if not wall or not encoder or not decoder_count:
        raise ValueError('checkpoint must contain trained WALL, encoder and decoder state')
    expected_encoder = _encoder_schema(rlt)
    if set(encoder) != set(expected_encoder) or any(
            tuple(encoder[key].shape) != shape for key, shape in expected_encoder.items()):
        raise ValueError('encoder weight keys/shapes disagree with RLT config')
    essential = {
        'model.embed_tokens.weight': (config['vocab_size'], config['hidden_size']),
        'action_preprocessor.w1.weight': (config['action_hidden_size'], contract.model_action_dim * 2),
        'action_preprocessor.action_proj_back.weight': (contract.model_action_dim, config['action_hidden_size']),
    }
    if config.get('norm_moe'):
        widths = config.get('dim_inputs', [])
        if len(widths) != config.get('num_experts') or len(widths) != 2:
            raise ValueError('unsupported WALL final expert normalization schema')
        for index, width in enumerate(widths):
            prefix = f'model.norms.{index}.'
            if index == 1 and config.get('use_adarms'):
                essential[prefix + 'dense.weight'] = (width * 3, config['adarms_cond_dim'])
                essential[prefix + 'dense.bias'] = (width * 3,)
            else:
                essential[prefix + 'weight'] = (width,)
    else:
        essential['model.norm.weight'] = (config['hidden_size'],)
    if not config.get('tie_word_embeddings') or 'lm_head.weight' in wall:
        essential['lm_head.weight'] = essential['model.embed_tokens.weight']
    for name, shape in essential.items():
        if name not in wall or tuple(wall[name].shape) != shape:
            raise ValueError(f'essential WALL weight missing or incorrectly shaped: {name}')
    expected_normalizers = _normalizer_values(contract)
    actual_normalizers = {name for name in wall if name.startswith((
        'action_preprocessor.normalizer_action.', 'action_preprocessor.normalizer_propri.'))}
    if actual_normalizers != set(expected_normalizers) or any(
            not torch.equal(wall[name].float(), value) for name, value in expected_normalizers.items()):
        raise ValueError('checkpoint normalizer weights differ from frozen training statistics')
    if config.get('tie_word_embeddings') and 'lm_head.weight' not in wall:
        # Some full-state producers omit this tied alias. Export both expected
        # keys so the standalone model can always be loaded with strict=True.
        wall['lm_head.weight'] = wall['model.embed_tokens.weight']
    return wall, encoder, decoder_count


def _manifest_id(manifest):
    return canonical_hash({key: manifest[key] for key in (
        'schema_version', 'kind', 'model_origin', 'training_step', 'data_contract_hash',
        'data_purpose', 'includes_decoder', 'includes_actor', 'validation_status', 'files',
    )})


def _cast_for_export(value, dtype, name):
    result = value.to(dtype).contiguous().clone()
    if not torch.isfinite(result).all():
        raise ValueError(f'weight overflow while converting inference precision: {name}')
    return result


def export_inference_artifact(checkpoint_dir, output_dir):
    """Convert one trusted, completed RLinf actor checkpoint into a new folder."""
    source, output = Path(checkpoint_dir).resolve(), Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('inference output must be a new directory')
    if output.resolve().is_relative_to(source):
        raise ValueError('inference output must be outside the source checkpoint')
    sidecar = _read(_regular(source, 'wall_rlt_data.json'))
    if sidecar.get('schema_version') != 1 or type(sidecar.get('step')) is not int or sidecar['step'] < 1:
        raise ValueError('a completed Stage 1 training step is required; no random base fallback')
    if not sidecar.get('contract_hash') or not isinstance(sidecar.get('assets_sha256'), dict):
        raise ValueError('checkpoint lacks bound training metadata/assets')
    assets = source / 'wall_rlt_assets'
    if assets.is_symlink() or not assets.is_dir():
        raise ValueError('checkpoint assets must be a real directory, not a symlink')
    if not REQUIRED_ASSETS.issubset(sidecar['assets_sha256']):
        raise ValueError('checkpoint is missing required inference assets')
    actual = _inventory(assets)
    if set(actual) != set(sidecar['assets_sha256']):
        raise ValueError('checkpoint asset inventory differs from sidecar')
    for name, expected in sidecar['assets_sha256'].items():
        if file_sha256(_regular(assets, name)) != expected:
            raise ValueError(f'checkpoint asset SHA256 mismatch: {name}')
    if not any(name in actual for name in ('processor/tokenizer.json', 'processor/vocab.json')):
        raise ValueError('checkpoint processor lacks tokenizer vocabulary')
    contract, rlt, wall_config = _bound_contract(assets, sidecar['contract_hash'])
    weights = _regular(source, 'model_state_dict/full_weights.pt')
    before = weights.stat()
    source_weight_hash = file_sha256(weights)
    # weights_only never loads optimizer or arbitrary checkpoint helper objects.
    state = torch.load(weights, weights_only=True, map_location='cpu', mmap=True)
    wall, encoder, decoder_count = _check_state(state, contract, rlt, wall_config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}.', dir=output.parent) as scratch:
        stage = Path(scratch) / 'artifact'
        stage.mkdir()
        shutil.copytree(assets, stage / 'assets')
        if any(file_sha256(_regular(stage / 'assets', name)) != expected
               for name, expected in sidecar['assets_sha256'].items()):
            raise ValueError('source assets changed while exporting')
        relocated = _read(stage / 'assets/data_contract.json')
        relocated.update(root='.', normalization_path='normalization.json', manifest_path='episodes.json')
        _write(stage / 'assets/data_contract.json', relocated)
        _bound_contract(stage / 'assets', sidecar['contract_hash'])
        # Match WALL's actual selective inference precision policy. A global
        # .bfloat16() breaks its FP32 flow projection path and rounds statistics.
        wall_export = {
            name: _cast_for_export(value, torch.float32 if any(marker in name for marker in FP32_WALL_MARKERS)
                                  else torch.bfloat16, name)
            for name, value in wall.items()
        }
        save_file(wall_export, stage / 'wall.safetensors')
        del wall_export
        encoder_export = {name: _cast_for_export(value, torch.float32, name) for name, value in encoder.items()}
        save_file(encoder_export, stage / 'encoder.safetensors')
        del encoder_export
        after = weights.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError('source weights changed while exporting')
        files = {name: {'bytes': path.stat().st_size, 'sha256': file_sha256(path)}
                 for name, path in _inventory(stage).items()}
        manifest = {
            'schema_version': 1, 'kind': KIND, 'model_origin': 'stage1_checkpoint',
            'validation_status': 'CANDIDATE_NOT_TARGET_VALIDATED',
            'training_step': sidecar['step'], 'data_contract_hash': contract.fingerprint(),
            'data_purpose': contract.purpose, 'includes_decoder': False, 'includes_actor': False,
            'source_checkpoint': str(source), 'source_weights_sha256': source_weight_hash,
            'source_sidecar_sha256': file_sha256(source / 'wall_rlt_data.json'),
            'excluded_decoder_tensors': decoder_count,
            'precision': {'wall': 'BF16 except upstream FP32 action/normalizer/layernorm parameters',
                          'encoder': 'FP32'},
            'files': files,
        }
        manifest['artifact_id'] = _manifest_id(manifest)
        _write(stage / MANIFEST, manifest)
        verify_inference_artifact(stage)
        if output.exists():
            raise FileExistsError('inference output appeared during export')
        stage.rename(output)
    return manifest


def verify_inference_artifact(root):
    """Verify the exact portable inventory and semantics, without model loading."""
    root = Path(root).resolve()
    manifest = _read(_regular(root, MANIFEST))
    if (manifest.get('schema_version') != 1 or manifest.get('kind') != KIND
            or manifest.get('model_origin') != 'stage1_checkpoint'
            or manifest.get('validation_status') != 'CANDIDATE_NOT_TARGET_VALIDATED'
            or type(manifest.get('training_step')) is not int or manifest['training_step'] < 1
            or manifest.get('includes_decoder') is not False or manifest.get('includes_actor') is not False
            or manifest.get('data_purpose') not in ('target', 'synthetic')):
        raise ValueError('unsupported or incomplete inference artifact metadata')
    files = manifest.get('files')
    if not isinstance(files, dict) or not {'wall.safetensors', 'encoder.safetensors'}.issubset(files):
        raise ValueError('inference artifact lacks weight files')
    if not {f'assets/{name}' for name in REQUIRED_ASSETS}.issubset(files):
        raise ValueError('inference artifact lacks required assets')
    if set(_inventory(root)) != set(files) | {MANIFEST}:
        raise ValueError('inference artifact file inventory mismatch')
    for name, info in files.items():
        path = _regular(root, name)
        if path.stat().st_size != info['bytes'] or file_sha256(path) != info['sha256']:
            raise ValueError(f'inference artifact checksum mismatch: {name}')
    if manifest.get('artifact_id') != _manifest_id(manifest):
        raise ValueError('inference artifact identity mismatch')
    contract, rlt, _ = _bound_contract(root / 'assets', manifest['data_contract_hash'])
    if contract.purpose != manifest['data_purpose']:
        raise ValueError('artifact data purpose differs from contract')
    with safe_open(root / 'encoder.safetensors', framework='pt', device='cpu') as handle:
        schema = _encoder_schema(rlt)
        if set(handle.keys()) != set(schema) or any(
                tuple(handle.get_slice(name).get_shape()) != shape for name, shape in schema.items()):
            raise ValueError('exported encoder schema mismatch')
    return manifest
