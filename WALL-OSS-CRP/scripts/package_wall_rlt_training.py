#!/usr/bin/env python3
"""Create a new portable training candidate; never alter the READY base bundle.

Everything dataset-specific comes from the explicit ``--contract``: its manifest,
normalization, data root, splits, exclusions and target/adapter semantics. The
2026-10-29 joint8 candidate is reproduced by passing its contract with
``--require-excluded mouse --require-excluded tape``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
WALL_COMMIT = 'ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f'
RLINF_COMMIT = 'e1a82ebf0c1af69da6ab9b1519f5f3692fe33566'
RLINF_SHA256 = 'd9e3d14aa19a9b7884defcebbe89497e279162a610e61a1d16ffabcfd1eb28f7'
BASE_IMAGE = 'sha256:8f067a0eb220228f444811532cc8ec0a328e4d46b7983637c1c4ea7743ef9ee8'
# 2: contract path, data directory and target semantics are recorded per candidate.
RECORD_SCHEMA = 2
DATASET_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def safe_relative(name):
    value = PurePosixPath(name)
    if not name or name=='.' or '\\' in name or value.is_absolute() or any(p in {'..','.'} for p in value.parts) or value.as_posix() != name:
        raise ValueError(f'unsafe or noncanonical archive path: {name!r}')
    return value


def regular_file(root,relative):
    relative = safe_relative(relative)
    root = Path(root).resolve()
    path = root/relative.as_posix()
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f'expected a contained regular file: {relative}')
    if any(p.is_symlink() for p in path.parents if p != root and p.is_relative_to(root)):
        raise ValueError('symlinked input directories are not packaged')
    return path


def contract_fingerprint(value):
    value = dict(value)
    for key in ('root','normalization_path','manifest_path'):
        value.pop(key,None)
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     ensure_ascii=False,allow_nan=False).encode()).hexdigest()


class HashingReader:
    def __init__(self,stream):
        self.stream,self.digest = stream,hashlib.sha256()
        self.count=0
    def read(self,size=-1):
        chunk=self.stream.read(size)
        self.digest.update(chunk);self.count += len(chunk)
        return chunk


def make_archive(destination,files,*,gzip=True):
    """Stream each source once, hash exactly the bytes placed in the archive."""
    specs={}
    with tarfile.open(destination,'w:gz' if gzip else 'w') as archive:
        folded=set()
        for name,path,expected in sorted(files):
            safe_relative(name)
            if name.casefold() in folded:
                raise ValueError('duplicate/case-colliding archive member')
            folded.add(name.casefold())
            path=Path(path)
            if path.is_symlink() or not path.is_file():
                raise ValueError('only regular input files may be archived')
            before=path.stat()
            if expected.get('bytes') is not None and before.st_size != expected['bytes']:
                raise ValueError(f'input size changed after audit: {name}')
            info=tarfile.TarInfo(name)
            info.size=before.st_size;info.mode=0o644;info.mtime=0
            with path.open('rb') as stream:
                reader=HashingReader(stream)
                archive.addfile(info,reader)
            after=path.stat()
            if (before.st_size,before.st_mtime_ns,before.st_ino) != (after.st_size,after.st_mtime_ns,after.st_ino) or reader.count != before.st_size:
                raise ValueError(f'input changed during packaging: {name}')
            digest=reader.digest.hexdigest()
            if expected.get('sha256') and digest != expected['sha256']:
                raise ValueError(f'input SHA256 changed after audit: {name}')
            specs[name]={'bytes':info.size,'sha256':digest}
    return {'sha256':sha256(destination),'bytes':Path(destination).stat().st_size,'members':specs}


def contract_relative(project,contract):
    """Project-relative POSIX path of a contract that must live inside the project."""
    project=Path(project).resolve()
    path=Path(contract)
    path=Path(os.path.normpath(path if path.is_absolute() else project/path))
    if not path.is_relative_to(project):
        raise ValueError('the contract must be inside the packaged project')
    relative=path.relative_to(project).as_posix()
    regular_file(project,relative)
    return relative


def contract_files(project,relative):
    """Contract, manifest and normalization paths, verified against the contract."""
    project=Path(project).resolve()
    cp=regular_file(project,relative)
    contract=json.loads(cp.read_text())
    paths={}
    for key in ('manifest_path','normalization_path'):
        value=contract.get(key)
        if not isinstance(value,str) or not value or Path(value).is_absolute():
            raise ValueError(f'contract {key} must be a relative path inside the project')
        target=Path(os.path.normpath(cp.parent/value))
        if not target.is_relative_to(project):
            raise ValueError(f'contract {key} points outside the project')
        paths[key]=regular_file(project,target.relative_to(project).as_posix())
    if sha256(paths['manifest_path'])!=contract.get('manifest_sha256'):
        raise ValueError('selected manifest is missing or differs from its contract')
    if sha256(paths['normalization_path'])!=contract.get('normalization_sha256'):
        raise ValueError('normalization is missing or differs from its contract')
    return contract,cp,paths['manifest_path'],paths['normalization_path']


def format_document(project,contract):
    """The data-format document named in provenance, if the project still has it."""
    prov=contract.get('provenance') or {}
    name=prov.get('format_document')
    if not name or '/' in name or not (Path(project)/name).is_file():
        return None
    digest=sha256(regular_file(project,name))
    return {'name':name,'sha256':digest,'contract_sha256':prov.get('format_document_sha256'),
            'matches_contract_provenance':digest==prov.get('format_document_sha256')}


def project_files(project,contract=None,extra=()):
    project=Path(project).resolve()
    names=set()
    for directory,suffixes in (('wall_rlt',{'.py','.md','.txt'}),('configs/rlt',{'.json','.yaml','.yml','.txt'}),
                               ('tests/rlt',{'.py'})):
        for path in (project/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and (path.suffix in suffixes or path.name.startswith('LICENSE')):
                names.add(path.relative_to(project).as_posix())
    names.update(p.relative_to(project).as_posix() for p in (project/'scripts').glob('*wall_rlt*.py'))
    for name in ('patches/wall-x-rlt-prefix.patch','metadata/wall-rlt-sources.lock.json',
                 'metadata/wall-rlt-design-audit-20261028.json',
                 'WALL_RLT_Agent_Implementation_Plan.md','WALL_RLT_Human_Review.md',
                 'docker/runtime-pins.txt','docker/expected-pip-check.txt',
                 'docker/verify_native_architectures.py',
                 'metadata/rlt-h100-memory-estimate-20261029.json',
                 'metadata/rlt-h100-access-20261029.json',
                 'metadata/rlt-training-dependency-selection-20261029.json',
                 'logs/rlt-training-env-local-20261029.json'):
        if (project/name).is_file():names.add(name)
    for path in (project/'logs/rlt-h100-preparation-20261029/cpu-check').glob('*'):
        if path.is_file() and path.suffix in {'.json','.log','.xml'}:
            names.add(path.relative_to(project).as_posix())
    if contract is not None:
        value,cp,mp,norm=contract_files(project,contract)
        names.add(contract)
        # The audit directory's records (audit, manifest, normalization, inspection
        # contract, batch checks) travel with the contract that cites them.
        for directory in {mp.parent,norm.parent}:
            for path in directory.glob('*.json'):
                if path.is_file() and not path.is_symlink():
                    names.add(path.relative_to(project).as_posix())
        document=format_document(project,value)
        if document:names.add(document['name'])
    for item in extra:
        path=project/safe_relative(item).as_posix()
        if path.is_dir() and not path.is_symlink():
            names.update(p.relative_to(project).as_posix() for p in path.rglob('*')
                         if p.is_file() and '__pycache__' not in p.parts)
        else:
            names.add(regular_file(project,item).relative_to(project).as_posix())
    return [(name,regular_file(project,name),{}) for name in sorted(names)]


def dataset_directory(contract,name=None):
    """``data/<name>``; by default the recordings root's own folder name."""
    name=name or Path(contract['root']).name
    if not DATASET_NAME.fullmatch(name) or name in {'.','..'}:
        raise ValueError(f'unsafe dataset directory name: {name!r}')
    return f'data/{name}'


def selected_data(project,contract_relative,required_excluded=()):
    project=Path(project).resolve()
    contract,_,mp,_=contract_files(project,contract_relative)
    manifest=json.loads(mp.read_text())
    excluded=sorted(manifest.get('excluded_groups',[]))
    missing=sorted(set(required_excluded)-set(excluded))
    if missing:
        raise ValueError(f'manifest must exclude groups {missing}')
    recorded=(contract.get('provenance') or {}).get('excluded_groups')
    if recorded is not None and sorted(recorded)!=excluded:
        raise ValueError('contract provenance exclusions differ from the manifest')
    if manifest['splits'] != contract['splits']:
        raise ValueError('frozen splits differ between manifest and contract')
    selected={i for values in contract['splits'].values() for i in values}
    entries=manifest['episodes']
    if selected != {e['episode_id'] for e in entries} or len(entries) != len(selected):
        raise ValueError('candidate requires a complete unique manifest of selected episodes')
    root=Path(contract['root']).resolve()
    result={};groups={}
    for entry in entries:
        group=entry['task_group']
        if group in excluded or entry.get('structural_errors') or entry.get('semantic_errors'):
            raise ValueError('excluded or failed recording in target manifest')
        if PurePosixPath(entry['npz']).parts[0] != group:
            raise ValueError('recording task folder differs from manifest group')
        if entry['npz'] in result:raise ValueError('duplicate recording path')
        result[entry['npz']]={'sha256':entry['npz_sha256']}
        for video in entry['videos'].values():
            if PurePosixPath(video['path']).parts[0] != group:
                raise ValueError('video task folder differs from manifest group')
            if video['path'] in result:raise ValueError('duplicate data path')
            result[video['path']]={'bytes':video['bytes']}
        prior=groups.setdefault(group,entry['dataset_metadata_sha256'])
        if prior != entry['dataset_metadata_sha256']:raise ValueError('task metadata digest disagrees across episodes')
    for group,digest in groups.items():
        result[f'{group}/dataset.json']={'sha256':digest}
    # wall_collect.py's settings record, when prepare_ar5 bound it into provenance.
    record=(contract.get('provenance') or {}).get('collection_record_sha256')
    if record:
        result['wall_collection.json']={'sha256':record}
    return contract,excluded,[(name,regular_file(root,name),spec) for name,spec in sorted(result.items())]


def find_rlinf_archive(project):
    candidates=[Path(project)/f'.cache/rlinf-{RLINF_COMMIT}.tar.gz']
    candidates.extend(sorted(Path('/tmp').glob('wall-rlt-review-*/rlinf.tar.gz')))
    for path in candidates:
        if path.is_file() and sha256(path)==RLINF_SHA256:return path
    raise FileNotFoundError('supply the verified pinned RLinf archive using --rlinf-archive')


def wheelhouse_files(wheelhouse):
    """Use the resolver's exact wheel inventory, without inventing a second lock."""
    wh=Path(wheelhouse).resolve()
    plan_path=regular_file(wh,'dependency-plan.json')
    plan=json.loads(plan_path.read_text())
    if (plan.get('schema_version')!=1 or plan.get('base_image_id')!=BASE_IMAGE or
            plan.get('status')!='WHEELS_PREPARED_NOT_CONTAINER_VALIDATED'):
        raise ValueError('wheelhouse is not a prepared overlay for this base image')
    wheels=plan.get('wheels',{})
    pins=plan.get('selected_overlay',{})
    if not wheels or wheels.keys()!=pins.keys():raise ValueError('wheelhouse overlay inventory is incomplete')
    expected_names={'dependency-plan.json','overlay-pins.txt','SHA256SUMS'}
    entries=[]
    for name,spec in wheels.items():
        filename=spec['filename']
        if '/' in filename or not filename.endswith('.whl') or filename in expected_names:
            raise ValueError('invalid or duplicate wheel filename')
        path=regular_file(wh,filename)
        expected_names.add(filename)
        entries.append((filename,path,{'bytes':spec['bytes'],'sha256':spec['sha256']}))
    if {p.name for p in wh.iterdir()}!=expected_names:
        raise ValueError('wheelhouse files differ from its complete dependency plan')
    pin_lines=regular_file(wh,'overlay-pins.txt').read_text().splitlines()
    if len(pin_lines)!=len(pins) or set(pin_lines)!={f'{name}=={version}' for name,version in pins.items()}:
        raise ValueError('overlay-pins.txt disagrees with dependency plan')
    sums={}
    for line in regular_file(wh,'SHA256SUMS').read_text().splitlines():
        digest,filename=line.split('  ',1)
        if len(digest)!=64 or filename in sums:raise ValueError('invalid wheelhouse checksum inventory')
        safe_relative(filename)
        sums[filename]=digest
    if set(sums)!=expected_names-{'SHA256SUMS'}:
        raise ValueError('wheelhouse checksum inventory is incomplete')
    for filename,path,spec in entries:
        if sums[filename]!=spec['sha256']:raise ValueError('wheel checksum differs from dependency plan')
    for filename in ('dependency-plan.json','overlay-pins.txt'):
        path=regular_file(wh,filename)
        entries.append((filename,path,{'sha256':sums[filename]}))
    entries.append(('SHA256SUMS',regular_file(wh,'SHA256SUMS'),{}))
    return entries


def package(project,output,*,contract,include_data=False,wheelhouse=None,rlinf_archive=None,
            required_excluded=(),dataset_name=None,extra=()):
    project,output=Path(project).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('candidate output must not already exist')
    if any(p.name.startswith('WALL-OSS-CRP-transfer-') for p in (output,*output.parents)):
        raise ValueError('training candidates must not be written inside a base delivery bundle')
    contract_path=contract_relative(project,contract)
    contract,excluded,data_files=selected_data(project,contract_path,required_excluded)
    data_dir=dataset_directory(contract,dataset_name)
    if output.is_relative_to(Path(contract['root']).resolve()):
        raise ValueError('candidate cannot be written inside original recordings')
    archive=Path(rlinf_archive) if rlinf_archive else find_rlinf_archive(project)
    if sha256(archive)!=RLINF_SHA256:raise ValueError('RLinf archive SHA256 mismatch')
    lock=json.loads((project/'metadata/wall-rlt-sources.lock.json').read_text())
    if lock['wall_commit']!=WALL_COMMIT or lock['rlinf_commit']!=RLINF_COMMIT or lock['rlinf_archive_sha256']!=RLINF_SHA256:
        raise ValueError('source lock differs from package pins')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}.',dir=output.parent) as temporary:
        stage=Path(temporary)/'candidate';stage.mkdir()
        payloads={}
        payloads['project.tar.gz']=make_archive(stage/'project.tar.gz',project_files(project,contract_path,extra))
        subprocess.run(['git','-C',str(project/'wall-x'),'cat-file','-e',f'{WALL_COMMIT}^{{commit}}'],check=True,capture_output=True)
        subprocess.run(['git','-C',str(project/'wall-x'),'bundle','create',str(stage/'wall-x.bundle'),'--all'],check=True,capture_output=True)
        shutil.copyfile(archive,stage/'rlinf-source.tar.gz')
        for name in ('wall-x.bundle','rlinf-source.tar.gz'):
            payloads[name]={'sha256':sha256(stage/name),'bytes':(stage/name).stat().st_size}
        if include_data:
            payloads['target-data.tar']=make_archive(stage/'target-data.tar',data_files,gzip=False)
        if wheelhouse:
            payloads['wheelhouse.tar.gz']=make_archive(stage/'wheelhouse.tar.gz',wheelhouse_files(wheelhouse))
        for name in ('package_wall_rlt_training.py','unpack_wall_rlt_training.py'):
            shutil.copyfile(project/'scripts'/name,stage/name)
            payloads[name]={'sha256':sha256(stage/name),'bytes':(stage/name).stat().st_size}
        record={'schema_version':RECORD_SCHEMA,'status':'TRAINING_CANDIDATE_NOT_GPU_VALIDATED',
                'kind':'wall-rlt-training-addendum','required_base_image_id':BASE_IMAGE,
                'base_delivery_is_separate':True,'wall_commit':WALL_COMMIT,'rlinf_commit':RLINF_COMMIT,
                'contract_path':contract_path,'contract_fingerprint':contract_fingerprint(contract),
                'repo_id':contract.get('repo_id'),'target_mode':contract.get('target_mode'),
                'model_adapter':contract.get('model_adapter'),'data_directory':data_dir,
                'excluded_groups':excluded,'split_counts':{k:len(v) for k,v in contract['splits'].items()},
                'format_document':format_document(project,contract),
                'included_data':bool(include_data),
                'included_wheelhouse':bool(wheelhouse),'payloads':payloads,
                'not_included':['base image','model weights','native extensions','credentials','host virtualenv'],
                'required_next_steps':['unpack to new project directory','prepare isolated training overlay',
                                       'copy verified sm90 native extension from the base environment',
                                       'H100 single-step and training/recovery validation'],
                'usage':'python3 unpack_wall_rlt_training.py --bundle CANDIDATE_DIR --destination NEW_PROJECT '
                        '--models-root EXISTING_MODELS [--data-root EXISTING_RECORDINGS_ROOT]'}
        write_json(stage/'TRAINING_CANDIDATE.json',record)
        if output.exists():raise FileExistsError('candidate output appeared during packaging')
        stage.rename(output)
    return {'output':str(output),'status':record['status'],'payloads':len(payloads),
            'included_data':bool(include_data),'contract_path':contract_path,
            'data_directory':data_dir,'target_mode':record['target_mode'],
            'contract_fingerprint':record['contract_fingerprint']}


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project',default=str(ROOT));p.add_argument('--output',required=True)
    p.add_argument('--contract',required=True,
                   help='target data contract inside the project (e.g. written by wall_rlt.data.prepare_ar5)')
    p.add_argument('--require-excluded',action='append',default=[],
                   help='task group the manifest must have excluded; repeatable (old joint8: mouse, tape)')
    p.add_argument('--dataset-name',help='bundled data directory data/<name>; default: the recordings root folder name')
    p.add_argument('--include',action='append',default=[],
                   help='extra project file or directory to package, e.g. a CPU-check log directory; repeatable')
    p.add_argument('--include-data',action='store_true');p.add_argument('--wheelhouse')
    p.add_argument('--rlinf-archive')
    a=p.parse_args()
    print(json.dumps(package(a.project,a.output,contract=a.contract,include_data=a.include_data,
                             wheelhouse=a.wheelhouse,rlinf_archive=a.rlinf_archive,
                             required_excluded=a.require_excluded,dataset_name=a.dataset_name,
                             extra=a.include),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
