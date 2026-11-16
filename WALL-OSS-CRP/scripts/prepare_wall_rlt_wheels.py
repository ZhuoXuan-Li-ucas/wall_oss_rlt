#!/usr/bin/env python3
"""Build a small offline overlay wheelhouse against the fixed D0 base packages.

Resolve from the locally tested isolated environment, honoring extras and
version constraints. Packages supplied by D0 are not downloaded or replaced.
This records a dependency plan, not validation inside the inaccessible D0
container or on H100. Run with .venv-rlt/bin/python (Linux x86_64 Python 3.10).
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def read_requirements(path):
    return [Requirement(s.strip()) for s in Path(path).read_text().splitlines()
            if s.strip() and not s.lstrip().startswith(('#','-'))]


def overlay_closure(requirements,base,lookup=metadata.distribution):
    """Use exact installed metadata for non-base dependencies; never guess pins."""
    pending=list(requirements); selected={}; visited={}; provided={}
    while pending:
        req=pending.pop()
        name=canonicalize_name(req.name)
        if req.marker and not req.marker.evaluate():
            continue
        if name in base:
            if not req.specifier.contains(base[name],prereleases=True):
                raise ValueError(f'D0 base conflict: {req}; supplied {base[name]}')
            if req.extras:
                raise ValueError(f'D0 package extras need explicit review: {req}')
            provided[name]=base[name]
            continue
        dist=lookup(req.name)
        if not req.specifier.contains(dist.version,prereleases=True):
            raise ValueError(f'local environment does not satisfy {req}: {dist.version}')
        extras=set(req.extras)|{''}
        new_extras=extras-visited.get(name,set())
        if name in selected and not new_extras:
            continue
        selected[name]=dist.version
        visited.setdefault(name,set()).update(extras)
        for raw in dist.requires or []:
            child=Requirement(raw)
            if child.marker and not any(child.marker.evaluate({'extra':e}) for e in new_extras):
                continue
            # The parent's extra condition is resolved here; retain dependency
            # extras, but do not re-evaluate that parent condition on queue pop.
            child.marker=None
            pending.append(child)
    return dict(sorted(selected.items())),dict(sorted(provided.items()))


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    if sys.version_info[:2]!=(3,10) or platform.system()!='Linux' or platform.machine()!='x86_64':
        parser.error('use the tested Linux x86_64 Python 3.10 environment')
    output=Path(args.output).resolve()
    if output.exists():
        parser.error('output must be a new directory')
    base={canonicalize_name(r.name):next(iter(r.specifier)).version
          for r in read_requirements(root/'docker/runtime-pins.txt')}
    base.update({'torch':'2.10.0+cu128','torchvision':'0.25.0+cu128',
                 'flash-attn':'2.8.3','setuptools':'80.9.0','wheel':'0.48.0','pip':'26.1.2'})
    reqs=read_requirements(root/'configs/rlt/requirements-training.txt')+[Requirement('pytest==8.4.2')]
    pins,provided=overlay_closure(reqs,base)
    output.mkdir(parents=True)
    (output/'overlay-pins.txt').write_text(''.join(f'{k}=={v}\n' for k,v in pins.items()))
    report={'schema_version':1,'status':'RESOLVED_FROM_LOCAL_TEST_ENV',
            'base_image_id':'sha256:8f067a0eb220228f444811532cc8ec0a328e4d46b7983637c1c4ea7743ef9ee8',
            'selected_overlay':pins,'required_base_packages':provided,
            'base_dependencies_replaced':False,'container_and_h100_validation':'NOT_RUN'}
    (output/'dependency-plan.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.download:
        for name,version in pins.items():
            # OmegaConf pins a pure-Python dependency whose upstream release has
            # no wheel; build only that source into a wheel, without dependencies.
            if name=='antlr4-python3-runtime' and version=='4.9.3':
                command=[sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation',
                         '--wheel-dir',str(output),f'{name}=={version}']
            else:
                command=[sys.executable,'-m','pip','download','--no-deps','--only-binary=:all:',
                         '--dest',str(output),f'{name}=={version}']
            subprocess.run(command,check=True)
        from packaging.utils import parse_wheel_filename
        wheels={}
        for p in output.glob('*.whl'):
            name,version,_,_=parse_wheel_filename(p.name)
            if name in wheels or name not in pins or str(version)!=pins[name]:
                raise ValueError(f'unexpected/duplicate wheel {p.name}')
            wheels[name]={'filename':p.name,'bytes':p.stat().st_size,
                          'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        if wheels.keys()!=pins.keys():
            raise ValueError('wheelhouse is incomplete')
        report['status']='WHEELS_PREPARED_NOT_CONTAINER_VALIDATED'
        report['wheels']=wheels
        (output/'dependency-plan.json').write_text(json.dumps(report,indent=2)+'\n')
        # Complete inventory, including metadata/pins; SHA256SUMS excludes only
        # itself. The candidate packager rechecks every streamed byte.
        (output/'SHA256SUMS').write_text(''.join(
            f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n'
            for p in sorted(output.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
    print(json.dumps({'status':report['status'],'overlay_packages':len(pins),'output':str(output)},indent=2))


if __name__=='__main__':main()
