import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from scripts import prepare_wall_rlt_sources as prep


def git(root,*args):
    return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True).stdout


@pytest.fixture
def small_repo(tmp_path):
    repo=tmp_path/'base';repo.mkdir()
    git(repo,'init','-q')
    source=repo/'model.py';source.write_text('value = 1\n')
    git(repo,'add','model.py')
    git(repo,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','initial')
    commit=git(repo,'rev-parse','HEAD').decode().strip()
    source.write_text('value = 2\n')
    patch=tmp_path/'extension.patch';patch.write_bytes(git(repo,'diff','--binary','HEAD','--'))
    source.write_text('value = 1\n')
    return repo,tmp_path/'isolated',patch,commit


def test_prepare_new_worktree_and_repeat_preserves_original(small_repo):
    repo,dest,patch,commit=small_repo
    for _ in range(2):
        assert prep.prepare_wall(repo,dest,patch,commit)['status']=='PASS'
    assert (dest/'model.py').read_text()=='value = 2\n'
    assert (repo/'model.py').read_text()=='value = 1\n'
    assert git(repo,'status','--porcelain')==b''


def test_prepare_refuses_additional_edits_without_reset(small_repo):
    repo,dest,patch,commit=small_repo
    prep.prepare_wall(repo,dest,patch,commit)
    (dest/'model.py').write_text('user_owned = 3\n')
    with pytest.raises(ValueError,match='no changes were reset'):
        prep.prepare_wall(repo,dest,patch,commit)
    assert (dest/'model.py').read_text()=='user_owned = 3\n'


@pytest.mark.parametrize('fault',['untracked','staged'])
def test_prepare_refuses_unknown_worktree_state(small_repo,fault):
    repo,dest,patch,commit=small_repo
    prep.prepare_wall(repo,dest,patch,commit)
    if fault=='untracked':(dest/'user_notes.txt').write_text('keep me')
    else:git(dest,'add','model.py')
    with pytest.raises(ValueError,match=fault):prep.prepare_wall(repo,dest,patch,commit)


def test_prepare_refuses_existing_non_worktree_and_original_destination(small_repo):
    repo,dest,patch,commit=small_repo
    dest.mkdir();(dest/'notes').write_text('keep')
    with pytest.raises(ValueError,match='not a worktree'):prep.prepare_wall(repo,dest,patch,commit)
    assert (dest/'notes').read_text()=='keep'
    with pytest.raises(ValueError,match='outside'):prep.prepare_wall(repo,repo,patch,commit)


def test_source_manifest_checks_all_files_and_symlinks_but_ignores_bytecode(tmp_path):
    (tmp_path/'code.py').write_text('value = 1\n')
    (tmp_path/'code-link').symlink_to('code.py')
    expected=prep.source_manifest(tmp_path)
    assert len(expected)==2
    (tmp_path/'__pycache__').mkdir()
    (tmp_path/'__pycache__/code.cpython-310.pyc').write_bytes(b'ignored')
    prep.verify_manifest(tmp_path,expected)
    (tmp_path/'code.py').write_text('value = 2\n')
    with pytest.raises(ValueError,match='changed'):prep.verify_manifest(tmp_path,expected)


def test_archive_hash_and_paths_are_checked_before_extract(tmp_path,monkeypatch):
    archive=tmp_path/'source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name,data in [(f'RLinf-{prep.RLINF_COMMIT}/code.py',b'x=1\n')]:
            member=tarfile.TarInfo(name);member.size=len(data)
            tar.addfile(member,io.BytesIO(data))
    with pytest.raises(ValueError,match='SHA256'):prep.archive_manifest(archive)
    monkeypatch.setattr(prep,'RLINF_ARCHIVE_SHA256',prep.file_digest(archive))
    files=prep.archive_manifest(archive)
    dest=tmp_path/'unpacked'
    prep.unpack_sources(archive,dest,files)
    assert (dest/'code.py').read_bytes()==b'x=1\n'
    with pytest.raises(ValueError,match='will not be replaced'):prep.unpack_sources(archive,dest,files)


def test_missing_archive_never_downloads_without_fetch(tmp_path,monkeypatch):
    def no_network(*a,**kw):raise AssertionError('network used')
    monkeypatch.setattr(prep.urllib.request,'urlopen',no_network)
    with pytest.raises(FileNotFoundError,match='--fetch'):
        prep.acquire_archive(tmp_path/'missing.tar.gz',False)
