#!/usr/bin/env python3
"""One exact HTTP Range request for each already-started official wheel."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import urllib.request

LOG=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('runtime_cache',LOG/'prepare_runtime_wheels.py')
cache=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)
STAGE=LOG/'official-wheel-downloads'
for name,version,filename in [
    ('numpy','1.26.4','numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'),
    ('sympy','1.14.0','sympy-1.14.0-py3-none-any.whl')]:
    report={'package':name,'version':version,'status':'FAIL','source':{'kind':'official PyPI wheel download'}}
    try:
        metadata_url=f'https://pypi.org/pypi/{name}/{version}/json'
        with urllib.request.urlopen(metadata_url,timeout=45) as response:
            metadata=json.load(response)
        artifact=next(a for a in metadata['urls'] if a['filename']==filename)
        target=STAGE/filename
        partial=target.with_suffix(target.suffix+'.partial')
        expected=artifact['digests']['sha256']
        report['source'].update(metadata_url=metadata_url,url=artifact['url'],pypi_sha256=expected)
        start=partial.stat().st_size
        end=artifact['size']-1
        report.update(resumed_from_byte=start,expected_bytes=artifact['size'])
        if not 0<start<=end:raise ValueError('Partial length is not resumable')
        print('RESUME',name,'offset',start,'remaining',artifact['size']-start,flush=True)
        request=urllib.request.Request(artifact['url'],headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(request,timeout=90) as response:
            if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{artifact["size"]}':
                raise ValueError('Server did not honor exact range; partial unchanged')
            with partial.open('ab') as output:
                for block in iter(lambda:response.read(65536),b''):
                    output.write(block)
        if partial.stat().st_size!=artifact['size'] or cache.file_sha(partial)!=expected:
            raise ValueError('Resumed official wheel SHA256/size mismatch; partial retained')
        description,verified=cache.verify_zip(partial)
        found_name,found_version,_,tags=description
        if cache.canonical(found_name)!=cache.canonical(name) or found_version!=version:
            raise ValueError('Downloaded wheel identity mismatch')
        partial.replace(target)
        report.update(status='PASS',filename=filename,wheel_sha256=expected,wheel_size_bytes=target.stat().st_size,
            wheel_tags=tags,record_verified_files=verified,excluded_installation_files=[],checked_at_utc=datetime.now(timezone.utc).isoformat())
        print('VERIFIED',name,expected,flush=True)
    except Exception as error:
        report['error']=str(error)
        print('FAIL',name,repr(error),flush=True)
    cache.save_json(STAGE/(name+'-'+version+'.json'),report)
