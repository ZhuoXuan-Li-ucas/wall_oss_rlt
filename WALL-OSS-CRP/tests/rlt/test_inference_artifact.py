"""Synthetic consolidated-checkpoint exports; no pretrained model or GPU load."""
import json
from pathlib import Path
import shutil

import pytest
import torch
from safetensors.torch import load_file

from wall_rlt.config import WallRLTConfig
from wall_rlt.data.contract import DataContract
from wall_rlt.model.ar5_adapter import AR5_ACTION_LAYOUT, AR5_STATE_LAYOUT
from wall_rlt._vendor.rlt_token_transformer import RLTTokenTransformer
from wall_rlt.inference.artifact import (
    export_inference_artifact, verify_inference_artifact, file_sha256, _normalizer_values,
)


PROJECT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def refresh_metadata(checkpoint, *, step=3):
    assets = checkpoint / 'wall_rlt_assets'
    contract = DataContract.load(assets / 'data_contract.json')
    value = {'schema_version': 1, 'step': step, 'contract_hash': contract.fingerprint(),
             'assets_sha256': {p.relative_to(assets).as_posix(): file_sha256(p)
                              for p in assets.rglob('*') if p.is_file()}}
    write(checkpoint / 'wall_rlt_data.json', value)
    return value


@pytest.fixture
def checkpoint(tmp_path):
    """Small random weights are marked synthetic, never a trained target model."""
    root = tmp_path / 'actor'
    assets = root / 'wall_rlt_assets'
    assets.mkdir(parents=True)
    value = json.loads((PROJECT / 'configs/rlt/top_error_ar5_joint_next_v1.json').read_text())
    original = DataContract.load(PROJECT / 'configs/rlt/top_error_ar5_joint_next_v1.json')
    shutil.copyfile(original.manifest_path, assets / 'episodes.json')
    shutil.copyfile(original.normalization_path, assets / 'normalization.json')
    value.update(root='/synthetic-fixture-no-recordings', purpose='synthetic',
                 normalization_path='normalization.json', manifest_path='episodes.json')
    write(assets / 'data_contract.json', value)
    contract = DataContract.load(assets / 'data_contract.json')
    rlt = WallRLTConfig(input_dim=4, embed_dim=4, prefix_seq_len=8,
                        num_layers=1, num_heads=1, mlp_ratio=1,
                        action_horizon=32, model_action_dim=26)
    rlt.save(assets / 'rlt_config.json')
    cfg = {'hidden_size': 4, 'vocab_size': 10, 'action_hidden_size': 4,
           'tie_word_embeddings': False, 'dof_config': dict(AR5_ACTION_LAYOUT),
           'agent_pos_config': dict(AR5_STATE_LAYOUT), 'proj_with_mask': True,
           'use_state_string_representation': True}
    write(assets / 'adapted_wall_config.json', cfg)
    write(assets / 'processor/config.json', {'fixture': True})
    write(assets / 'processor/preprocessor_config.json', {'fixture': True})
    write(assets / 'processor/tokenizer_config.json', {'fixture': True})
    write(assets / 'processor/tokenizer.json', {'fixture': True})
    write(assets / 'source_hashes.json', {'fixture': 'synthetic'})
    write(assets / 'training_config.json', {'fixture': 'synthetic'})
    write(assets / 'build_report.json', {'fixture': 'synthetic'})
    (assets / 'wall_train_config.yml').write_text('fixture: synthetic\n')
    state = {
        'wall.model.embed_tokens.weight': torch.randn(10, 4),
        'wall.model.norm.weight': torch.ones(4),
        'wall.lm_head.weight': torch.randn(10, 4),
        'wall.action_preprocessor.w1.weight': torch.randn(4, 52),
        'wall.action_preprocessor.action_proj_back.weight': torch.randn(26, 4),
    }
    state.update({'wall.' + key: value for key, value in _normalizer_values(contract).items()})
    token = RLTTokenTransformer(**rlt.token_kwargs())
    state.update({'rlt_module.' + key: value for key, value in token.state_dict().items()})
    (root / 'model_state_dict').mkdir()
    torch.save(state, root / 'model_state_dict/full_weights.pt')
    refresh_metadata(root)
    return root


def test_export_roundtrip_keeps_encoder_precision_and_discards_decoder(checkpoint, tmp_path):
    before = {p.relative_to(checkpoint).as_posix(): file_sha256(p)
              for p in checkpoint.rglob('*') if p.is_file()}
    destination = tmp_path / 'inference'
    exported = export_inference_artifact(checkpoint, destination)
    assert verify_inference_artifact(destination) == exported
    assert exported['kind'] == 'wall-rlt-stage1-inference'
    assert exported['model_origin'] == 'stage1_checkpoint'
    assert exported['training_step'] == 3 and exported['data_purpose'] == 'synthetic'
    assert exported['includes_decoder'] is False and exported['includes_actor'] is False
    assert exported['excluded_decoder_tensors'] > 0
    encoder = load_file(destination / 'encoder.safetensors')
    assert encoder and all(t.dtype == torch.float32 for t in encoder.values())
    assert not any('decoder' in key or key.startswith('rlt_module.') for key in encoder)
    wall = load_file(destination / 'wall.safetensors')
    assert wall['model.embed_tokens.weight'].dtype == torch.bfloat16
    assert wall['action_preprocessor.w1.weight'].dtype == torch.float32
    assert wall['model.norm.weight'].dtype == torch.float32
    relocated = DataContract.load(destination / 'assets/data_contract.json')
    assert relocated.root == str((destination / 'assets').resolve())
    assert relocated.fingerprint() == exported['data_contract_hash']
    assert not (destination / 'full_weights.pt').exists()
    assert {p.relative_to(checkpoint).as_posix(): file_sha256(p)
            for p in checkpoint.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('target', ['wall.safetensors', 'assets/normalization.json'])
def test_artifact_rejects_tampered_file(checkpoint, tmp_path, target):
    output = tmp_path / 'output'
    export_inference_artifact(checkpoint, output)
    with (output / target).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        verify_inference_artifact(output)


def test_zero_step_or_missing_training_sidecar_is_rejected(checkpoint, tmp_path):
    refresh_metadata(checkpoint, step=0)
    with pytest.raises(ValueError, match='completed Stage 1'):
        export_inference_artifact(checkpoint, tmp_path / 'zero')
    (checkpoint / 'wall_rlt_data.json').unlink()
    with pytest.raises(ValueError, match='missing'):
        export_inference_artifact(checkpoint, tmp_path / 'no-sidecar')


def test_missing_asset_and_asset_hash_corruption_are_rejected(checkpoint, tmp_path):
    asset = checkpoint / 'wall_rlt_assets/processor/tokenizer.json'
    original = asset.read_bytes()
    asset.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        export_inference_artifact(checkpoint, tmp_path / 'hash')
    asset.write_bytes(original)
    asset.unlink()
    with pytest.raises(ValueError, match='inventory'):
        export_inference_artifact(checkpoint, tmp_path / 'missing')


def test_contract_fingerprint_is_bound_to_sidecar(checkpoint, tmp_path):
    path = checkpoint / 'wall_rlt_data.json'
    metadata = json.loads(path.read_text())
    metadata['contract_hash'] = '0' * 64
    write(path, metadata)
    with pytest.raises(ValueError, match='fingerprint'):
        export_inference_artifact(checkpoint, tmp_path / 'wrong-contract')


@pytest.mark.parametrize('change', ['nan', 'foreign-key', 'normalizer', 'encoder-shape'])
def test_invalid_weight_dictionary_is_rejected(checkpoint, tmp_path, change):
    path = checkpoint / 'model_state_dict/full_weights.pt'
    weights = torch.load(path, weights_only=True)
    if change == 'nan':
        weights['wall.lm_head.weight'][0, 0] = float('nan')
    elif change == 'foreign-key':
        weights['optimizer.secret'] = torch.ones(1)
    elif change == 'normalizer':
        key = next(k for k in weights if 'normalizer_action.min.' in k)
        weights[key] = weights[key] + 1
    else:
        key = next(k for k in weights if k.startswith('rlt_module.encoder.'))
        weights[key] = torch.ones(1)
    torch.save(weights, path)
    with pytest.raises(ValueError):
        export_inference_artifact(checkpoint, tmp_path / change)
    assert not (tmp_path / change).exists()


def test_unmerged_checkpoint_and_existing_output_are_refused(checkpoint, tmp_path):
    output = tmp_path / 'present'
    output.mkdir()
    with pytest.raises(FileExistsError):
        export_inference_artifact(checkpoint, output)
    (checkpoint / 'model_state_dict/full_weights.pt').unlink()
    (checkpoint / 'dcp_checkpoint').mkdir()
    with pytest.raises(ValueError, match='full_weights'):
        export_inference_artifact(checkpoint, tmp_path / 'unmerged')


def test_extra_file_and_manifest_identity_changes_are_rejected(checkpoint, tmp_path):
    output = tmp_path / 'output'
    export_inference_artifact(checkpoint, output)
    extra = output / 'unexpected.txt'
    extra.write_text('outside inventory')
    with pytest.raises(ValueError, match='inventory'):
        verify_inference_artifact(output)
    extra.unlink()
    path = output / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['training_step'] += 1
    write(path, manifest)
    with pytest.raises(ValueError, match='identity'):
        verify_inference_artifact(output)


def test_asset_symlink_cannot_escape_checkpoint(checkpoint, tmp_path):
    path = checkpoint / 'wall_rlt_assets/source_hashes.json'
    outside = tmp_path / 'source_hashes.json'
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        export_inference_artifact(checkpoint, tmp_path / 'linked')


def test_tied_head_alias_is_materialized_for_strict_runtime_loading(checkpoint, tmp_path):
    cfgpath = checkpoint / 'wall_rlt_assets/adapted_wall_config.json'
    cfg = json.loads(cfgpath.read_text())
    cfg['tie_word_embeddings'] = True
    write(cfgpath, cfg)
    refresh_metadata(checkpoint)
    weightpath = checkpoint / 'model_state_dict/full_weights.pt'
    state = torch.load(weightpath, weights_only=True)
    del state['wall.lm_head.weight']
    torch.save(state, weightpath)
    output = tmp_path / 'tied'
    export_inference_artifact(checkpoint, output)
    state = load_file(output / 'wall.safetensors')
    torch.testing.assert_close(state['lm_head.weight'], state['model.embed_tokens.weight'], rtol=0, atol=0)


def test_original_ee_layout_is_not_accepted_as_joint8(checkpoint, tmp_path):
    cfgpath = checkpoint / 'wall_rlt_assets/adapted_wall_config.json'
    cfg = json.loads(cfgpath.read_text())
    cfg['dof_config'] = {'ee_pose': 26}
    write(cfgpath, cfg)
    refresh_metadata(checkpoint)
    with pytest.raises(ValueError, match='joint8 layout'):
        export_inference_artifact(checkpoint, tmp_path / 'wrong-layout')


def test_finite_fp32_that_overflows_bf16_is_rejected(checkpoint, tmp_path):
    path = checkpoint / 'model_state_dict/full_weights.pt'
    state = torch.load(path, weights_only=True)
    state['wall.lm_head.weight'][0, 0] = torch.finfo(torch.float32).max
    torch.save(state, path)
    with pytest.raises(ValueError, match='overflow'):
        export_inference_artifact(checkpoint, tmp_path / 'overflow')


@pytest.mark.parametrize('adaptive', [False, True])
def test_moe_final_norm_matches_actual_wall_schema(checkpoint, tmp_path, adaptive):
    cfgpath = checkpoint / 'wall_rlt_assets/adapted_wall_config.json'
    cfg = json.loads(cfgpath.read_text())
    cfg.update(norm_moe=True, num_experts=2, dim_inputs=[4,4],
               use_adarms=adaptive, adarms_cond_dim=4)
    write(cfgpath, cfg)
    refresh_metadata(checkpoint)
    path = checkpoint / 'model_state_dict/full_weights.pt'
    state = torch.load(path, weights_only=True)
    del state['wall.model.norm.weight']
    state['wall.model.norms.0.weight'] = torch.ones(4)
    if adaptive:
        state['wall.model.norms.1.dense.weight'] = torch.randn(12,4)
        state['wall.model.norms.1.dense.bias'] = torch.zeros(12)
    else:
        state['wall.model.norms.1.weight'] = torch.ones(4)
    torch.save(state, path)
    output = tmp_path / 'moe-norm'
    export_inference_artifact(checkpoint, output)
    exported = load_file(output / 'wall.safetensors')
    assert all(value.dtype == torch.float32 for name,value in exported.items()
               if name.startswith('model.norms.'))
