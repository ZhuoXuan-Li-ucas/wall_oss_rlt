#!/usr/bin/env python3
"""Fetch explicitly selected missing runtime wheels once from official PyPI."""
import importlib.util
import json
from pathlib import Path
import sys
import urllib.request
from packaging.utils import parse_wheel_filename

LOG=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('runtime_cache',LOG/'prepare_runtime_wheels.py')
cache=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)
STAGE=LOG/'official-wheel-downloads'
STAGE.mkdir(exist_ok=True)

for item in sys.argv[1:]:
    name,version=item.split('==',1)
    report={'package':name,'version':version,'status':'FAIL','source':{'kind':'official PyPI wheel download'}}
    try:
        metadata_url=f'https://pypi.org/pypi/{name}/{version}/json'
        with urllib.request.urlopen(metadata_url, timeout=45) as response:
            metadata=json.load(response)
        candidates=[]
        for artifact in metadata['urls']:
            if artifact['packagetype']!='bdist_wheel':continue
            _,_,_,tags=parse_wheel_filename(artifact['filename'])
            if not tags.intersection(cache.TARGET_TAGS):continue
            # Prefer a matching CPython extension wheel over the pure-Python fallback.
            priority=0 if any(t.interpreter=='cp310' and t.abi=='cp310' for t in tags) else 1
            candidates.append((priority,artifact['filename'],artifact))
        if not candidates:raise ValueError('No Ubuntu22/CPython3.10 compatible wheel')
        artifact=sorted(candidates,key=lambda x:x[:2])[0][2]
        target=STAGE/artifact['filename']
        partial=target.with_suffix(target.suffix+'.partial')
        expected=artifact['digests']['sha256']
        print('DOWNLOAD',name,version,artifact['size'],artifact['filename'],flush=True)
        if not target.exists() or cache.file_sha(target)!=expected:
            with urllib.request.urlopen(artifact['url'], timeout=90) as response, partial.open('wb') as output:
                while True:
                    block=response.read(1024*1024)
                    if not block:break
                    output.write(block)
            if partial.stat().st_size!=artifact['size'] or cache.file_sha(partial)!=expected:
                raise ValueError('Official PyPI SHA256/size mismatch')
            partial.replace(target)
        description,verified=cache.verify_zip(target)
        found_name,found_version,_,tags=description
        if cache.canonical(found_name)!=cache.canonical(name) or found_version!=version:
            raise ValueError('Downloaded metadata identity mismatch')
        report.update(status='PASS',filename=target.name,wheel_sha256=expected,
            wheel_size_bytes=target.stat().st_size,wheel_tags=tags,
            record_verified_files=verified,excluded_installation_files=[])
        report['source'].update(metadata_url=metadata_url,url=artifact['url'],pypi_sha256=expected)
        print('VERIFIED',name,version,expected,flush=True)
    except Exception as error:
        report['error']=str(error)
        print('FAIL',name,version,repr(error),flush=True)
    cache.save_json(STAGE/(cache.canonical(name)+'-'+version+'.json'),report)
