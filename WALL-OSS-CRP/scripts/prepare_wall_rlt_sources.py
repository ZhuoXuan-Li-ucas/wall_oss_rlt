#!/usr/bin/env python3
"""Prepare/verify isolated pinned sources; no installs, weights or native builds.

Default operation is offline. --fetch explicitly permits downloading only the
pinned RLinf source archive. An existing checkout with extra changes is refused,
never reset. --write-lock requires the independently verified source archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import subprocess
import tarfile
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
WALL_COMMIT = 'ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f'
RLINF_COMMIT = 'e1a82ebf0c1af69da6ab9b1519f5f3692fe33566'
RLINF_ARCHIVE_SHA256 = 'd9e3d14aa19a9b7884defcebbe89497e279162a610e61a1d16ffabcfd1eb28f7'
RLINF_URL = f'https://codeload.github.com/RLinf/RLinf/tar.gz/{RLINF_COMMIT}'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def git(repo,*args,check=True):
    return subprocess.run(['git','-C',str(repo),*map(str,args)],check=check,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE)


def prepare_wall(repo,destination,patch,commit=WALL_COMMIT):
    """Accept only a pristine base or exactly the reviewed patch, idempotently."""
    repo,destination,patch = Path(repo).resolve(),Path(destination).resolve(),Path(patch).resolve()
    if destination == repo or destination.is_relative_to(repo):
        raise ValueError('isolated WALL destination must be outside the base source directory')
    git(repo,'cat-file','-e',f'{commit}^{{commit}}')
    if not destination.exists():
        destination.parent.mkdir(parents=True,exist_ok=True)
        git(repo,'worktree','add','--detach',destination,commit)
    if not (destination/'.git').is_file():
        raise ValueError('existing WALL destination is not a worktree; refusing to replace it')
    if git(destination,'rev-parse','HEAD').stdout.decode().strip() != commit:
        raise ValueError('existing WALL worktree is at a different commit; no reset performed')
    original_common=Path(git(repo,'rev-parse','--git-common-dir').stdout.decode().strip())
    candidate_common=Path(git(destination,'rev-parse','--git-common-dir').stdout.decode().strip())
    if (repo/original_common).resolve() != (destination/candidate_common).resolve():
        raise ValueError('WALL destination belongs to a different git repository')
    if git(destination,'diff','--cached','--quiet',check=False).returncode != 0:
        raise ValueError('WALL worktree has staged changes; refusing to modify it')
    untracked=git(destination,'ls-files','--others','--exclude-standard').stdout.decode().splitlines()
    # Existing platform-specific runtime extension is outside the source lock.
    unexpected=[p for p in untracked if not (
        p.startswith('wall_x/model/core/ops/_cuda_ext_bin.') and p.endswith('.so'))]
    if unexpected:
        raise ValueError(f'WALL worktree has untracked files: {unexpected[:8]}')
    current=git(destination,'diff','--binary','HEAD','--').stdout
    expected=patch.read_bytes()
    if current:
        if current != expected:
            raise ValueError('WALL worktree differs from the locked patch; no changes were reset')
    else:
        git(destination,'apply','--check',patch)
        git(destination,'apply',patch)
        if git(destination,'diff','--binary','HEAD','--').stdout != expected:
            raise ValueError('applied WALL patch differs from canonical git diff; review the new worktree')
    return {'path':str(destination),'commit':commit,'patch_sha256':file_digest(patch),
            'status':'PASS','native_extension':'not built or supplied by this script'}


def ignored(relative):
    return any(p in {'__pycache__','.git','.pytest_cache'} for p in relative.parts) or relative.suffix in {'.pyc','.pyo'}


def source_manifest(root):
    root=Path(root)
    files={}
    for path in sorted(root.rglob('*')):
        relative=path.relative_to(root)
        if ignored(relative):continue
        if path.is_symlink():
            files[relative.as_posix()]={'kind':'symlink','target':os.readlink(path)}
        elif path.is_file():
            files[relative.as_posix()]={'kind':'file','sha256':file_digest(path)}
    return files


def archive_manifest(archive):
    if file_digest(archive) != RLINF_ARCHIVE_SHA256:
        raise ValueError('RLinf source archive SHA256 mismatch')
    files={}
    prefix=f'RLinf-{RLINF_COMMIT}'
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            path=PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or path.parts[0] != prefix:
                raise ValueError('unsafe or unexpected archive member path')
            relative=PurePosixPath(*path.parts[1:])
            if member.isdir():continue
            if ignored(relative):continue
            if relative.as_posix() in files:raise ValueError('duplicate source archive member')
            if member.issym():
                target=posixpath.normpath(str(relative.parent/member.linkname))
                if PurePosixPath(member.linkname).is_absolute() or target == '..' or target.startswith('../'):
                    raise ValueError('source archive symlink escapes the archive root')
                files[relative.as_posix()]={'kind':'symlink','target':member.linkname}
            elif member.isfile():
                files[relative.as_posix()]={'kind':'file','sha256':digest(tar.extractfile(member).read())}
            else:
                raise ValueError('unsupported source archive member type')
    return dict(sorted(files.items()))


def verify_manifest(root,expected):
    actual=source_manifest(root)
    missing=sorted(set(expected)-set(actual))
    added=sorted(set(actual)-set(expected))
    changed=sorted(k for k in set(expected)&set(actual) if expected[k] != actual[k])
    if missing or added or changed:
        raise ValueError(f'RLinf source tree differs: missing={missing[:5]}, added={added[:5]}, changed={changed[:5]}')


def unpack_sources(archive,destination,expected):
    """Extract regular files then contained symlinks into a fresh temporary tree."""
    destination=Path(destination).resolve()
    if destination.exists():
        raise ValueError('existing RLinf directory will not be replaced')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='rlinf-source-',dir=destination.parent) as scratch:
        stage=Path(scratch)/'source';stage.mkdir()
        with tarfile.open(archive,'r:gz') as tar:
            for member in tar.getmembers():
                relative=PurePosixPath(*PurePosixPath(member.name).parts[1:])
                spec=expected.get(str(relative))
                if not spec or spec['kind']!='file':continue
                target=stage/str(relative)
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
                target.chmod(member.mode & 0o777)
        for relative,spec in expected.items():
            if spec['kind']=='symlink':
                target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True)
                target.symlink_to(spec['target'])
        verify_manifest(stage,expected)
        stage.rename(destination)


def acquire_archive(path,fetch):
    path=Path(path)
    if not path.is_file():
        if not fetch:raise FileNotFoundError('source archive absent; supply --rlinf-archive or explicitly use --fetch')
        path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as tmp:
            tmp_path=Path(tmp.name)
            try:
                with urllib.request.urlopen(RLINF_URL,timeout=60) as response:
                    for chunk in iter(lambda:response.read(1024*1024),b''):
                        tmp.write(chunk)
                tmp.close()
                if file_digest(tmp_path)!=RLINF_ARCHIVE_SHA256:
                    raise ValueError('downloaded RLinf archive SHA256 mismatch')
                tmp_path.rename(path)
            finally:
                tmp_path.unlink(missing_ok=True)
    return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall-repo',type=Path,default=ROOT/'wall-x')
    parser.add_argument('--wall-dest',type=Path,default=ROOT/'.cache/wall-x-rlt')
    parser.add_argument('--patch',type=Path,default=ROOT/'patches/wall-x-rlt-prefix.patch')
    parser.add_argument('--rlinf-dest',type=Path,default=ROOT/'.cache/rlinf-rlt')
    parser.add_argument('--rlinf-archive',type=Path,default=ROOT/f'.cache/rlinf-{RLINF_COMMIT}.tar.gz')
    parser.add_argument('--lock',type=Path,default=ROOT/'metadata/wall-rlt-sources.lock.json')
    parser.add_argument('--fetch',action='store_true')
    parser.add_argument('--write-lock',action='store_true',help='Derive a source lock from the verified pinned archive')
    args=parser.parse_args(argv)
    if args.write_lock:
        archive=acquire_archive(args.rlinf_archive,args.fetch)
        files=archive_manifest(archive)
        lock={'schema_version':1,'wall_commit':WALL_COMMIT,'wall_patch_sha256':file_digest(args.patch),
              'rlinf_commit':RLINF_COMMIT,'rlinf_archive_url':RLINF_URL,
              'rlinf_archive_sha256':RLINF_ARCHIVE_SHA256,'rlinf_files':files}
    else:
        lock=json.loads(args.lock.read_text())
        if (lock.get('schema_version')!=1 or lock.get('wall_commit')!=WALL_COMMIT or
                lock.get('rlinf_commit')!=RLINF_COMMIT or lock.get('rlinf_archive_sha256')!=RLINF_ARCHIVE_SHA256 or
                lock.get('wall_patch_sha256')!=file_digest(args.patch)):
            raise ValueError('source lock version/patch mismatch; review changes before regenerating it')
        files=lock['rlinf_files']
    if args.rlinf_dest.exists():
        verify_manifest(args.rlinf_dest,files)
    else:
        archive=acquire_archive(args.rlinf_archive,args.fetch)
        if archive_manifest(archive)!=files:raise ValueError('archive differs from source lock')
        unpack_sources(archive,args.rlinf_dest,files)
    wall=prepare_wall(args.wall_repo,args.wall_dest,args.patch)
    if args.write_lock:
        args.lock.parent.mkdir(parents=True,exist_ok=True)
        args.lock.write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'wall':wall,'rlinf':{'path':str(args.rlinf_dest.resolve()),'commit':RLINF_COMMIT,
        'verified_entries':len(files),'status':'PASS'},'lock':str(args.lock.resolve()),
        'dependencies_installed':False,'weights_loaded':False,'native_build_started':False},indent=2))


if __name__=='__main__':main()
