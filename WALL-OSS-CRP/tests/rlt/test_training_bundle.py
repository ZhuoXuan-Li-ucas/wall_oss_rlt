"""Portable candidate integrity tests, using only tiny local fixture files."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import package_wall_rlt_training as pack
import unpack_wall_rlt_training as unpack


CONTRACT='configs/rlt/fixture_contract.json'
JOINT8=('ar5_joint_next_v1','ar5_joint8_reinit_v1')
EE=('ar5_ee_rel6d_next_v1','wall_oss05_right_arm_ee_v1')


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return pack.sha256(path)


def make_fixture(tmp_path,monkeypatch,*,root_name='top_error',semantics=JOINT8,
                 excluded=('mouse','tape'),recorded_excluded=None,collection=False):
    project=tmp_path/'project';project.mkdir()
    repo=project/'wall-x';repo.mkdir()
    def git(*args):
        return subprocess.run(['git','-C',str(repo),*args],capture_output=True,text=True,check=True).stdout.strip()
    git('init');git('config','user.name','fixture');git('config','user.email','fixture@example.invalid')
    (repo/'tracked.py').write_text('print("fixed source")\n')
    git('add','.');git('commit','-m','fixture')
    commit=git('rev-parse','HEAD')
    raw=tmp_path/'raw'/root_name
    (raw/'banana').mkdir(parents=True)
    (raw/'mouse').mkdir();(raw/'mouse'/'secret').write_text('do not include')
    metadata_hash=dump(raw/'banana/dataset.json',{'prompt':'pick banana'})
    npz=raw/'banana/episode_0002.npz';npz.write_bytes(b'fixture numerical recording')
    vp=raw/'banana/episode_0002_cam_front.mp4';vp.write_bytes(b'fixture video')
    episode={'episode_id':9,'task_group':'banana','npz':'banana/episode_0002.npz',
             'npz_sha256':pack.sha256(npz),'dataset_metadata_sha256':metadata_hash,
             'videos':{'cam_front':{'path':'banana/episode_0002_cam_front.mp4','bytes':vp.stat().st_size}}}
    manifest={'episodes':[episode],'splits':{'train':[9],'dev':[],'test':[]},'excluded_groups':list(excluded)}
    md=project/'metadata/fixture-audit'
    mh=dump(md/'episodes.json',manifest)
    nh=dump(md/f'normalization.{semantics[0]}.json',{'norm_stats':{}})
    dump(md/'audit.json',{'structural_status':'PASS'})
    (project/'数据集格式说明.md').write_text('edited after the contract was prepared')
    provenance={'format_document':'数据集格式说明.md','format_document_sha256':'0'*64,
                'excluded_groups':sorted(excluded if recorded_excluded is None else recorded_excluded)}
    if collection:
        provenance['collection_record_sha256']=dump(raw/'wall_collection.json',{'settings':{'逆解':'weighted'}})
    contract={'root':str(raw),'manifest_path':'../../metadata/fixture-audit/episodes.json',
              'manifest_sha256':mh,'normalization_path':f'../../metadata/fixture-audit/normalization.{semantics[0]}.json',
              'normalization_sha256':nh,'splits':manifest['splits'],'action_names':['target'],'schema_version':1,
              'repo_id':f'local/{root_name}','target_mode':semantics[0],'model_adapter':semantics[1],
              'provenance':provenance}
    dump(project/CONTRACT,contract)
    archive=tmp_path/'rlinf.tar.gz';archive.write_bytes(b'pinned source archive fixture')
    digest=pack.sha256(archive)
    for module in (pack,unpack):
        monkeypatch.setattr(module,'WALL_COMMIT',commit)
        monkeypatch.setattr(module,'RLINF_SHA256',digest)
    dump(project/'metadata/wall-rlt-sources.lock.json',{'wall_commit':commit,'rlinf_commit':pack.RLINF_COMMIT,'rlinf_archive_sha256':digest})
    (project/'scripts').mkdir()
    for name in ('package_wall_rlt_training.py','unpack_wall_rlt_training.py'):
        (project/'scripts'/name).write_bytes((SCRIPTS/name).read_bytes())
    # A minimal stand-in for the separately tested source preparer. Confirms the
    # checkout was actually reconstructed and receives the local archive path.
    (project/'scripts/prepare_wall_rlt_sources.py').write_text(
        'from pathlib import Path\nimport sys\nassert Path("wall-x/tracked.py").is_file()\n'
        'assert Path(sys.argv[sys.argv.index("--rlinf-archive")+1]).is_file()\n'
        'Path("source-preparation-ran").write_text("yes")\n')
    (project/'wall_rlt').mkdir();(project/'wall_rlt/__init__.py').write_text('')
    (project/'env').mkdir();(project/'env/secret').write_text('do not package')
    (project/'H100使用说明.md').write_text('do not package accounts')
    (project/'wall_rlt/host.so').write_bytes(b'do not package native binaries')
    models=tmp_path/'models';(models/'wall-oss-0.5').mkdir(parents=True)
    (models/'wall-oss-0.5/config.json').write_text('{}')
    (models/'wall-oss-0.5/model.safetensors').write_bytes(b'existing model')
    return project,raw,archive,models,contract


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    return make_fixture(tmp_path,monkeypatch)


def test_full_candidate_unpacks_sources_and_relocates_without_semantic_change(fixture,tmp_path):
    project,raw,archive,models,contract=fixture
    candidate=tmp_path/'candidate'
    # The 2026-10-29 joint8 flow: its contract plus the mouse/tape exclusions, explicitly.
    result=pack.package(project,candidate,contract=CONTRACT,include_data=True,rlinf_archive=archive,
                        required_excluded=('mouse','tape'))
    assert result['status']=='TRAINING_CANDIDATE_NOT_GPU_VALIDATED'
    record=unpack.verify_bundle(candidate)
    assert (record['contract_path'],record['data_directory'])==(CONTRACT,'data/top_error')
    assert (record['target_mode'],record['model_adapter'])==JOINT8
    assert record['excluded_groups']==['mouse','tape']
    assert set(record['payloads']['target-data.tar']['members'])=={
        'banana/dataset.json','banana/episode_0002.npz','banana/episode_0002_cam_front.mp4'}
    names=record['payloads']['project.tar.gz']['members']
    assert not any('env/' in name or name.endswith('.so') or 'H100使用说明' in name or '.git' in name for name in names)
    assert {CONTRACT,'metadata/fixture-audit/episodes.json','metadata/fixture-audit/audit.json',
            'metadata/fixture-audit/normalization.ar5_joint_next_v1.json'}<=set(names)
    dest=tmp_path/'unpacked'
    result=unpack.unpack(candidate,dest,models_root=models)
    assert result['sources_prepared'] is True
    assert (dest/'source-preparation-ran').read_text()=='yes'
    assert (dest/'data/top_error/banana/episode_0002.npz').read_bytes()==(raw/'banana/episode_0002.npz').read_bytes()
    relocated=json.loads((dest/CONTRACT).read_text())
    assert relocated['root']==str(dest/'data/top_error')
    assert pack.contract_fingerprint(relocated)==pack.contract_fingerprint(contract)
    assert result['training_overrides']==[f'data.train_data_paths={dest/CONTRACT}',
                                          f'actor.model.rlt.data_contract_path={dest/CONTRACT}']
    assert (dest/'models').resolve()==models
    assert not (dest/'TRAINING_UNPACK_INCOMPLETE.json').exists()
    assert (raw/'mouse/secret').read_text()=='do not include'


def test_ee_candidate_records_its_semantics_and_carries_the_collection_record(tmp_path,monkeypatch):
    project,raw,archive,models,contract=make_fixture(tmp_path,monkeypatch,root_name='wall_unlocked',
                                                     semantics=EE,excluded=(),collection=True)
    candidate=tmp_path/'candidate'
    result=pack.package(project,candidate,contract=project/CONTRACT,include_data=True,rlinf_archive=archive)
    assert (result['data_directory'],result['target_mode'])==('data/wall_unlocked',EE[0])
    record=unpack.verify_bundle(candidate)
    assert (record['target_mode'],record['model_adapter'],record['repo_id'])==(*EE,'local/wall_unlocked')
    assert record['excluded_groups']==[] and record['split_counts']=={'train':1,'dev':0,'test':0}
    assert record['format_document']['name']=='数据集格式说明.md'
    assert record['format_document']['matches_contract_provenance'] is False
    assert 'wall_collection.json' in record['payloads']['target-data.tar']['members']
    dest=tmp_path/'unpacked'
    unpack.unpack(candidate,dest,models_root=models)
    assert (dest/'data/wall_unlocked/wall_collection.json').read_bytes()==(raw/'wall_collection.json').read_bytes()
    relocated=json.loads((dest/CONTRACT).read_text())
    assert relocated['root']==str(dest/'data/wall_unlocked')
    assert pack.contract_fingerprint(relocated)==pack.contract_fingerprint(contract)


def test_changed_collection_record_is_rejected(tmp_path,monkeypatch):
    project,raw,archive,_,_=make_fixture(tmp_path,monkeypatch,semantics=EE,collection=True)
    dump(raw/'wall_collection.json',{'settings':{'逆解':'weighted'},'sessions':['appended later']})
    with pytest.raises(ValueError,match='SHA256 changed'):
        pack.package(project,tmp_path/'candidate',contract=CONTRACT,include_data=True,rlinf_archive=archive)


def test_exclusions_come_from_the_manifest_and_must_agree(tmp_path,monkeypatch):
    project,_,archive,_,_=make_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match=r"must exclude groups \['other'\]"):
        pack.package(project,tmp_path/'c1',contract=CONTRACT,rlinf_archive=archive,required_excluded=('mouse','other'))
    other=tmp_path/'second';other.mkdir()
    project,_,archive,_,_=make_fixture(other,monkeypatch,recorded_excluded=('mouse',))
    with pytest.raises(ValueError,match='provenance exclusions differ'):
        pack.package(project,tmp_path/'c2',contract=CONTRACT,rlinf_archive=archive)


def test_contract_must_be_in_project_with_relative_artifact_paths(fixture,tmp_path):
    project,_,archive,_,contract=fixture
    outside=tmp_path/'outside.json';outside.write_text(json.dumps(contract))
    with pytest.raises(ValueError,match='inside the packaged project'):
        pack.package(project,tmp_path/'c1',contract=outside,rlinf_archive=archive)
    absolute=dict(contract,manifest_path=str(project/'metadata/fixture-audit/episodes.json'))
    dump(project/'configs/rlt/absolute.json',absolute)
    with pytest.raises(ValueError,match='relative path inside the project'):
        pack.package(project,tmp_path/'c2',contract='configs/rlt/absolute.json',rlinf_archive=archive)
    assert not (tmp_path/'c1').exists() and not (tmp_path/'c2').exists()


def test_dataset_directory_name_can_be_chosen_but_not_escape(fixture,tmp_path):
    project,_,archive,_,_=fixture
    result=pack.package(project,tmp_path/'candidate',contract=CONTRACT,rlinf_archive=archive,dataset_name='renamed')
    assert result['data_directory']=='data/renamed'
    for name in ('../escape','a/b','.hidden',''):
        with pytest.raises(ValueError,match='unsafe dataset directory'):
            pack.dataset_directory({'root':'/x/y'},name or '.')


def test_extra_evidence_is_packaged_only_when_named(fixture):
    project,_,_,_,_=fixture
    (project/'logs/ee-check').mkdir(parents=True);(project/'logs/ee-check/report.json').write_text('{}')
    (project/'logs/other.json').write_text('{}')
    names={entry[0] for entry in pack.project_files(project,CONTRACT,extra=('logs/ee-check',))}
    assert 'logs/ee-check/report.json' in names and 'logs/other.json' not in names
    with pytest.raises(ValueError,match='unsafe'):pack.project_files(project,CONTRACT,extra=('../outside',))


def test_source_only_candidate_requires_existing_data_and_never_copies_models(fixture,tmp_path):
    project,raw,archive,models,_=fixture
    candidate=tmp_path/'candidate'
    pack.package(project,candidate,contract=CONTRACT,rlinf_archive=archive)
    with pytest.raises(ValueError,match='requires an existing data-root'):
        unpack.unpack(candidate,tmp_path/'fail',models_root=models)
    empty=tmp_path/'empty';empty.mkdir()
    with pytest.raises(ValueError,match='data-root lacks 1 manifest recordings'):
        unpack.unpack(candidate,tmp_path/'fail',models_root=models,data_root=empty)
    assert not (tmp_path/'fail').exists()
    dest=tmp_path/'unpacked'
    unpack.unpack(candidate,dest,models_root=models,data_root=raw)
    assert not (dest/'data').exists()
    assert json.loads((dest/CONTRACT).read_text())['root']==str(raw)


def test_pack_rejects_changed_recording_without_publishing_candidate(fixture,tmp_path):
    project,raw,archive,_,_=fixture
    (raw/'banana/episode_0002.npz').write_bytes(b'changed after audit')
    with pytest.raises(ValueError,match='SHA256 changed'):
        pack.package(project,tmp_path/'candidate',contract=CONTRACT,include_data=True,rlinf_archive=archive)
    assert not (tmp_path/'candidate').exists()


def test_pack_rejects_changed_video_size(fixture,tmp_path):
    project,raw,archive,_,_=fixture
    (raw/'banana/episode_0002_cam_front.mp4').write_bytes(b'changed video payload length')
    with pytest.raises(ValueError,match='size changed'):
        pack.package(project,tmp_path/'candidate',contract=CONTRACT,include_data=True,rlinf_archive=archive)


def test_no_overwrite_of_candidate_destination_or_base_delivery(fixture,tmp_path):
    project,raw,archive,models,_=fixture
    candidate=tmp_path/'candidate'
    pack.package(project,candidate,contract=CONTRACT,include_data=True,rlinf_archive=archive)
    with pytest.raises(FileExistsError):pack.package(project,candidate,contract=CONTRACT,rlinf_archive=archive)
    dest=tmp_path/'existing';dest.mkdir()
    with pytest.raises(FileExistsError):unpack.unpack(candidate,dest,models_root=models)
    with pytest.raises(ValueError,match='base delivery'):
        pack.package(project,tmp_path/'WALL-OSS-CRP-transfer-fixture'/'new',contract=CONTRACT,rlinf_archive=archive)


def test_payload_corruption_is_rejected_before_destination_is_created(fixture,tmp_path):
    project,_,archive,models,_=fixture
    candidate=tmp_path/'candidate'
    pack.package(project,candidate,contract=CONTRACT,include_data=True,rlinf_archive=archive)
    with (candidate/'project.tar.gz').open('ab') as stream:stream.write(b'corrupt')
    dest=tmp_path/'unpacked'
    with pytest.raises(ValueError,match='checksum mismatch'):
        unpack.unpack(candidate,dest,models_root=models)
    assert not dest.exists()


@pytest.mark.parametrize('name',['../escape','/absolute','a/../../escape','a\\b','./a','a//b','.'])
def test_unsafe_archive_paths_rejected(tmp_path,name):
    with pytest.raises(ValueError,match='unsafe'):
        pack.safe_relative(name)


@pytest.mark.parametrize('kind',['symlink','hardlink','duplicate','missing','digest'])
def test_unsafe_or_incomplete_archive_content_is_rejected(tmp_path,kind):
    archive=tmp_path/'archive.tar'
    spec={'members':{'file':{'bytes':1,'sha256':hashlib.sha256(b'x').hexdigest()}}}
    with tarfile.open(archive,'w') as stream:
        info=tarfile.TarInfo('file');info.size=1
        if kind in {'symlink','hardlink'}:
            info.type=tarfile.SYMTYPE if kind=='symlink' else tarfile.LNKTYPE
            info.linkname='../escape';info.size=0
            stream.addfile(info)
        elif kind!='missing':
            stream.addfile(info,io.BytesIO(b'y' if kind=='digest' else b'x'))
            if kind=='duplicate':stream.addfile(info,io.BytesIO(b'x'))
    with pytest.raises(ValueError):unpack.extract_checked(archive,tmp_path/'out',spec)


def test_input_symlink_is_rejected(fixture,tmp_path):
    project,raw,archive,_,_=fixture
    path=raw/'banana/episode_0002.npz';target=tmp_path/'same-bytes';target.write_bytes(path.read_bytes())
    path.unlink();path.symlink_to(target)
    with pytest.raises(ValueError,match='regular file'):
        pack.package(project,tmp_path/'candidate',contract=CONTRACT,include_data=True,rlinf_archive=archive)


def test_optional_wheelhouse_is_included_and_bound(fixture,tmp_path):
    project,_,archive,models,_=fixture
    wheels=tmp_path/'wheels';wheels.mkdir()
    wheel=wheels/'fixture-1-py3-none-any.whl';wheel.write_bytes(b'wheel')
    dump(wheels/'dependency-plan.json',{'schema_version':1,'base_image_id':pack.BASE_IMAGE,
        'status':'WHEELS_PREPARED_NOT_CONTAINER_VALIDATED','selected_overlay':{'fixture':'1'},
        'wheels':{'fixture':{'filename':wheel.name,'bytes':wheel.stat().st_size,'sha256':pack.sha256(wheel)}}})
    (wheels/'overlay-pins.txt').write_text('fixture==1\n')
    (wheels/'SHA256SUMS').write_text(''.join(f'{pack.sha256(p)}  {p.name}\n' for p in sorted(wheels.iterdir())))
    candidate=tmp_path/'candidate'
    pack.package(project,candidate,contract=CONTRACT,include_data=True,wheelhouse=wheels,rlinf_archive=archive)
    result=unpack.unpack(candidate,tmp_path/'unpacked',models_root=models)
    assert (Path(result['project'])/'wheelhouse/fixture-1-py3-none-any.whl').read_bytes()==b'wheel'


def test_exported_training_environment_loads_native_helper_after_relocation(tmp_path):
    """Exercise dynamic runtime dependencies from the real candidate inventory."""
    required={'scripts/prepare_wall_rlt_training_env.py',
              'docker/verify_native_architectures.py','docker/runtime-pins.txt',
              'configs/rlt/constraints-training.txt','configs/rlt/requirements-training.txt'}
    selected=[entry for entry in pack.project_files(SCRIPTS.parent) if entry[0] in required]
    assert {entry[0] for entry in selected}==required
    archive=tmp_path/'runtime.tar.gz'
    manifest=pack.make_archive(archive,selected)
    relocated=tmp_path/'new-runtime'
    unpack.extract_checked(archive,relocated,manifest)
    spec=importlib.util.spec_from_file_location('relocated_training_env',
             relocated/'scripts/prepare_wall_rlt_training_env.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ROOT==relocated
    assert module.base_pins()['torch']=='2.10.0+cu128'
    helper=module.native_tools()
    assert Path(helper.__file__).is_relative_to(relocated)
    assert callable(helper.inspect_extension)
    constraint=relocated/'configs/rlt/constraints-training.txt'
    nested=[line[3:] for line in constraint.read_text().splitlines() if line.startswith('-c ')]
    assert nested and all((constraint.parent/path).is_file() for path in nested)


def test_candidate_cli_bootstrap_does_not_mutate_candidate_inventory(tmp_path):
    """The actual CLI import path must preserve even a read-only candidate."""
    candidate=tmp_path/'candidate';candidate.mkdir()
    for name in ('package_wall_rlt_training.py','unpack_wall_rlt_training.py'):
        (candidate/name).write_bytes((SCRIPTS/name).read_bytes())
    before={path.name:pack.sha256(path) for path in candidate.iterdir()}
    result=subprocess.run([sys.executable,'-E',str(candidate/'unpack_wall_rlt_training.py'),'--help'],
                          capture_output=True,text=True,check=True)
    assert '--models-root' in result.stdout
    assert {path.name for path in candidate.iterdir()}==set(before)
    assert {path.name:pack.sha256(path) for path in candidate.iterdir()}==before
