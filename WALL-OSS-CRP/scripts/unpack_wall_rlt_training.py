#!/usr/bin/env python3
"""Verify and unpack a training candidate into a new local project directory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

# The bootstrap lives in the checksummed candidate directory. Importing its
# sibling must not add __pycache__ and invalidate the complete file inventory.
sys.dont_write_bytecode = True
from package_wall_rlt_training import (BASE_IMAGE,DATASET_NAME,RECORD_SCHEMA,RLINF_COMMIT,RLINF_SHA256,
    WALL_COMMIT,contract_fingerprint,regular_file,safe_relative,sha256,write_json)


def extract_checked(archive,destination,spec):
    """Extract only manifest-listed regular files, rejecting links and traversal."""
    destination=Path(destination)
    expected=spec['members'];seen=set();folded=set()
    with tarfile.open(archive,'r:*') as stream:
        for member in stream:
            name=member.name
            safe_relative(name)
            if not member.isfile() or name in seen or name.casefold() in folded or name not in expected:
                raise ValueError('archive contains unexpected/duplicate/nonregular member')
            seen.add(name);folded.add(name.casefold())
            if member.size!=expected[name]['bytes']:
                raise ValueError('archive member size differs from manifest')
            target=destination/name
            target.parent.mkdir(parents=True,exist_ok=True)
            import hashlib
            digest=hashlib.sha256()
            with stream.extractfile(member) as source,target.open('xb') as output:
                for block in iter(lambda:source.read(4*1024**2),b''):
                    digest.update(block);output.write(block)
            if digest.hexdigest()!=expected[name]['sha256']:
                raise ValueError('archive member SHA256 mismatch')
    if seen!=set(expected):raise ValueError('archive omits required manifest members')


def verify_bundle(bundle):
    bundle=Path(bundle).resolve()
    record=json.loads(regular_file(bundle,'TRAINING_CANDIDATE.json').read_text())
    # Schema 1 candidates (2026-10-29) carry their own unpacker; use that copy.
    if (record.get('schema_version')!=RECORD_SCHEMA or record.get('kind')!='wall-rlt-training-addendum' or
        record.get('status')!='TRAINING_CANDIDATE_NOT_GPU_VALIDATED' or
        record.get('required_base_image_id')!=BASE_IMAGE or record.get('wall_commit')!=WALL_COMMIT or
        record.get('rlinf_commit')!=RLINF_COMMIT):
        raise ValueError('unsupported or incompatible training candidate')
    contract=safe_relative(record.get('contract_path') or '')
    if contract.suffix!='.json':raise ValueError('candidate contract path must name a JSON contract')
    data_dir=safe_relative(record.get('data_directory') or '')
    if len(data_dir.parts)!=2 or data_dir.parts[0]!='data' or not DATASET_NAME.fullmatch(data_dir.parts[1]):
        raise ValueError('candidate data directory must be data/<name>')
    required={'project.tar.gz','wall-x.bundle','rlinf-source.tar.gz',
              'package_wall_rlt_training.py','unpack_wall_rlt_training.py'}
    if record['included_data']:required.add('target-data.tar')
    if record['included_wheelhouse']:required.add('wheelhouse.tar.gz')
    if set(record['payloads'])!=required:
        raise ValueError('candidate payload inventory is incomplete or unexpected')
    actual={p.name for p in bundle.iterdir()}
    if actual!=required|{'TRAINING_CANDIDATE.json'}:
        raise ValueError('bundle directory differs from its complete inventory')
    for name,spec in record['payloads'].items():
        path=regular_file(bundle,name)
        if path.stat().st_size!=spec['bytes'] or sha256(path)!=spec['sha256']:
            raise ValueError(f'candidate payload checksum mismatch: {name}')
    if sha256(bundle/'rlinf-source.tar.gz')!=RLINF_SHA256:
        raise ValueError('candidate does not contain the pinned RLinf archive')
    return record


def unpack(bundle,destination,*,models_root,data_root=None,prepare_sources=True):
    bundle,destination=Path(bundle).resolve(),Path(destination).resolve()
    if destination.exists():raise FileExistsError('destination must be a new project directory')
    record=verify_bundle(bundle)
    models_root=Path(models_root).resolve()
    if not (models_root/'wall-oss-0.5/config.json').is_file() or not (models_root/'wall-oss-0.5/model.safetensors').is_file():
        raise ValueError('models-root must contain the already delivered wall-oss-0.5 model')
    if record['included_data'] and data_root is not None:
        raise ValueError('do not supply data-root when target data is included')
    if not record['included_data'] and data_root is None:
        raise ValueError('candidate without target data requires an existing data-root')
    existing_data=Path(data_root).resolve() if data_root is not None else None
    if existing_data is not None and not existing_data.is_dir():raise ValueError('data-root is not a directory')
    if destination.is_relative_to(models_root) or (existing_data is not None and destination.is_relative_to(existing_data)):
        raise ValueError('new project must not be placed inside model or raw data directories')
    destination.parent.mkdir(parents=True,exist_ok=True)
    # All tar extraction happens in a fresh temporary directory before publishing
    # project files. Worktrees are created after rename so gitdir paths are valid.
    with tempfile.TemporaryDirectory(prefix=f'.{destination.name}.',dir=destination.parent) as temp:
        stage=Path(temp)/'project';stage.mkdir()
        extract_checked(bundle/'project.tar.gz',stage,record['payloads']['project.tar.gz'])
        if record['included_data']:
            extract_checked(bundle/'target-data.tar',stage/record['data_directory'],record['payloads']['target-data.tar'])
            target_data=destination/record['data_directory']
        else:
            target_data=existing_data
        if record['included_wheelhouse']:
            extract_checked(bundle/'wheelhouse.tar.gz',stage/'wheelhouse',record['payloads']['wheelhouse.tar.gz'])
        contract_path=regular_file(stage,record['contract_path'])
        contract=json.loads(contract_path.read_text())
        if contract_fingerprint(contract)!=record['contract_fingerprint']:
            raise ValueError('candidate contract fingerprint mismatch')
        if any(contract.get(k)!=record.get(k) for k in ('target_mode','model_adapter','repo_id')):
            raise ValueError('candidate record disagrees with its contract semantics')
        if existing_data is not None:
            # Cheap guard against a wrong --data-root; the trainer re-hashes every recording.
            manifest=json.loads(regular_file(stage,Path(os.path.normpath(
                contract_path.parent/contract['manifest_path'])).relative_to(stage).as_posix()).read_text())
            missing=[e['npz'] for e in manifest['episodes'] if not (existing_data/e['npz']).is_file()]
            if missing:
                raise ValueError(f'data-root lacks {len(missing)} manifest recordings, e.g. {missing[0]}')
        # Same relocation as the workspace restore: only the recordings root changes;
        # manifest/normalization stay relative to the contract inside the project.
        contract['root']=str(target_data)
        write_json(contract_path,contract)
        if contract_fingerprint(contract)!=record['contract_fingerprint']:
            raise ValueError('relocation changed contract semantics')
        (stage/'models').symlink_to(models_root,target_is_directory=True)
        write_json(stage/'TRAINING_UNPACK_INCOMPLETE.json',{'source':str(bundle),'status':'SOURCE_PREPARATION_PENDING'})
        if destination.exists():raise FileExistsError('destination appeared during unpacking')
        stage.rename(destination)
    if prepare_sources:
        subprocess.run(['git','clone','--no-checkout',str(bundle/'wall-x.bundle'),str(destination/'wall-x')],
                       check=True,capture_output=True,text=True)
        subprocess.run(['git','-C',str(destination/'wall-x'),'checkout','--detach',WALL_COMMIT],
                       check=True,capture_output=True,text=True)
        subprocess.run([sys.executable,str(destination/'scripts/prepare_wall_rlt_sources.py'),
                        '--rlinf-archive',str(bundle/'rlinf-source.tar.gz')],cwd=destination,
                       check=True,capture_output=True,text=True)
    contract=str(destination/record['contract_path'])
    report={'status':'TRAINING_CANDIDATE_UNPACKED_NOT_GPU_VALIDATED','project':str(destination),
            'data_root':str(target_data),'models_root':str(models_root),
            'contract_path':contract,'target_mode':record['target_mode'],'model_adapter':record['model_adapter'],
            # configs/rlt/stage1*.yaml name the old joint8 contract; train_wall_rlt.py --set overrides it.
            'training_overrides':[f'data.train_data_paths={contract}',
                                  f'actor.model.rlt.data_contract_path={contract}'],
            'validation_override_single_rank':f'data.val_data_paths={contract}',
            'contract_fingerprint':record['contract_fingerprint'],'sources_prepared':prepare_sources,
            'native_extension_installed':False,'dependencies_installed':False,
            'required_base_image_id':BASE_IMAGE,'source_bundle':str(bundle)}
    write_json(destination/'TRAINING_CANDIDATE_UNPACKED.json',report)
    (destination/'TRAINING_UNPACK_INCOMPLETE.json').unlink()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--models-root',required=True)
    p.add_argument('--data-root',help='existing recordings root, only for candidates packaged without data')
    a=p.parse_args()
    print(json.dumps(unpack(a.bundle,a.destination,models_root=a.models_root,data_root=a.data_root),indent=2))


if __name__=='__main__':main()
