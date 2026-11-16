#!/usr/bin/env python3
"""Recover cached wheels or rebuild only verified installed payload, without network."""
import base64
import csv
from datetime import datetime, timezone
from email.parser import Parser
import hashlib
import io
from importlib.metadata import distribution
import itertools
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile

from packaging.tags import cpython_tags, compatible_tags, parse_tag

ROOT = Path('/home/pangk/workspaces/WALL-OSS-CRP')
LOG = ROOT / 'logs/runtime-wheel-cache-20261028'
OUT = Path('/media/pangk/My Passport/WALL-OSS-CRP-build-cache-20261028/wheelhouse')
SOURCE = ROOT / 'env/lib/python3.10/site-packages'
PREFIX = ROOT / 'env'
PINS = [line.split('==') for line in (ROOT/'docker/runtime-pins.txt').read_text().splitlines()
        if line and not line.startswith('#')]
PINS += [('pip', '26.1.2'), ('setuptools', '80.9.0'), ('wheel', '0.48.0')]
PLATFORMS = [f'manylinux_2_{n}_x86_64' for n in range(35, 4, -1)]
PLATFORMS += ['manylinux2014_x86_64', 'manylinux2010_x86_64', 'manylinux1_x86_64']
TARGET_TAGS = set(cpython_tags((3, 10), abis=['cp310'], platforms=PLATFORMS))
TARGET_TAGS.update(compatible_tags((3, 10), interpreter='cp310', platforms=PLATFORMS))

def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()

def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def save_json(path, data):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)

def describe_wheel(metadata_text, wheel_text, dist):
    metadata = Parser().parsestr(metadata_text)
    wheel = Parser().parsestr(wheel_text)
    tags = wheel.get_all('Tag', [])
    parsed = set().union(*(parse_tag(tag) for tag in tags)) if tags else set()
    if not parsed.intersection(TARGET_TAGS):
        raise ValueError('No compatible CPython 3.10 / Ubuntu 22.04 wheel tag: ' + repr(tags))
    parts = [str(tag).split('-') for tag in sorted(parsed, key=str)]
    groups = [sorted({p[i] for p in parts}) for i in range(3)]
    if set('-'.join(p) for p in itertools.product(*groups)) != {str(t) for t in parsed}:
        raise ValueError('Wheel tags cannot be represented without adding compatibility claims')
    build = wheel.get('Build')
    name = dist[:-len('.dist-info')]
    filename = name + ('-' + build if build else '') + '-' + '-'.join('.'.join(g) for g in groups) + '.whl'
    return metadata['Name'], metadata['Version'], filename, tags

def verify_zip(path):
    with zipfile.ZipFile(path) as archive:
        members = [x for x in archive.infolist() if not x.is_dir()]
        names = [x.filename for x in members]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate wheel members')
        if any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in names):
            raise ValueError('Unsafe wheel path')
        records = [n for n in names if n.count('/') == 1 and n.endswith('.dist-info/RECORD')]
        if len(records) != 1:
            raise ValueError('Expected exactly one RECORD')
        record = records[0]
        dist = record.rsplit('/', 1)[0]
        rows = list(csv.reader(io.StringIO(archive.read(record).decode())))
        if any(len(r) != 3 for r in rows) or len(rows) != len({r[0] for r in rows}):
            raise ValueError('Malformed RECORD')
        if set(names) != {r[0] for r in rows}:
            raise ValueError('Unrecorded or missing archive member')
        verified = []
        for rel, expected, size_text in rows:
            if rel == record:
                if expected or size_text:
                    raise ValueError('RECORD must not hash itself')
                continue
            if not expected or not size_text:
                raise ValueError('Missing member hash/size: ' + rel)
            algorithm, expected_hash = expected.split('=', 1)
            if algorithm not in {'sha256', 'sha384', 'sha512'}:
                raise ValueError('Unsupported hash: ' + algorithm)
            digest = hashlib.new(algorithm)
            sha256 = hashlib.sha256()
            count = 0
            with archive.open(rel) as stream:
                for block in iter(lambda: stream.read(4*1024*1024), b''):
                    digest.update(block); sha256.update(block); count += len(block)
            actual = base64.urlsafe_b64encode(digest.digest()).decode().rstrip('=')
            if actual != expected_hash or count != int(size_text):
                raise ValueError('Wheel RECORD mismatch: ' + rel)
            verified.append({'path': rel, 'size_bytes': count, 'sha256': sha256.hexdigest()})
        description = describe_wheel(archive.read(dist+'/METADATA').decode(), archive.read(dist+'/WHEEL').decode(), dist)
        return description, verified

def recover_cached(name, version, cached):
    source = Path(cached['source'])
    description, verified = verify_zip(source)
    found_name, found_version, filename, tags = description
    if canonical(found_name) != canonical(name) or found_version != version:
        raise ValueError('Cached package identity mismatch')
    target = OUT / filename
    if target.exists():
        raise FileExistsError(str(target))
    partial = target.with_suffix(target.suffix + '.partial')
    original_sha = file_sha(source)
    shutil.copyfile(source, partial)
    if file_sha(partial) != original_sha:
        partial.unlink()
        raise ValueError('Cached wheel copy changed bytes')
    partial.replace(target)
    return {'status': 'PASS', 'package': name, 'version': version, 'filename': filename,
            'wheel_sha256': original_sha, 'wheel_size_bytes': target.stat().st_size,
            'wheel_tags': tags, 'source': {'kind': 'pip cache original wheel archive', 'path': str(source)},
            'record_verified_files': verified, 'excluded_installation_files': []}

def rebuild_installed(name, version):
    package = distribution(name)
    if package.version != version or package._path.parent.resolve() != SOURCE.resolve():
        raise ValueError('Installed version/location mismatch')
    dist = package._path.name
    record = package._path/'RECORD'
    rows = list(csv.reader(record.read_text().splitlines()))
    if any(len(r) != 3 for r in rows):
        raise ValueError('Malformed installed RECORD')
    payload_paths = [r[0] for r in rows if not r[0].endswith('.pyc') and '__pycache__' not in PurePosixPath(r[0]).parts]
    if len(payload_paths) != len(set(payload_paths)):
        raise ValueError('Duplicate installed payload RECORD entries')
    found_name, found_version, filename, tags = describe_wheel(package.read_text('METADATA'), package.read_text('WHEEL'), dist)
    if canonical(found_name) != canonical(name) or found_version != version:
        raise ValueError('Installed metadata identity mismatch')
    entries = {ep.name for ep in package.entry_points if ep.group in {'console_scripts', 'gui_scripts'}}
    if canonical(name) == 'pip':
        entries.update({'pip', 'pip3', 'pip3.10'})
    report = {'status': 'FAIL', 'package': name, 'version': version, 'filename': filename,
              'source': {'kind': 'RECORD-verified installed package', 'site_packages': str(SOURCE)},
              'source_record_sha256': file_sha(record), 'wheel_tags': tags,
              'record_verified_files': [], 'excluded_installation_files': []}
    target = OUT/filename
    partial = OUT/(filename+'.partial')
    if target.exists():
        raise FileExistsError(str(target))
    try:
        output_rows = []
        with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for rel, expected, size_text in rows:
                relpath = PurePosixPath(rel)
                if relpath.is_absolute():
                    raise ValueError('Absolute external RECORD path: ' + rel)
                if rel == dist+'/RECORD':
                    if expected or size_text:
                        raise ValueError('RECORD must not hash itself')
                    continue
                if rel.endswith('.pyc') or '__pycache__' in relpath.parts:
                    report['excluded_installation_files'].append({'path': rel, 'reason': 'generated bytecode omitted; not source payload or original wheel content'})
                    continue
                source = SOURCE/rel
                mapped = rel
                generated = False
                reason = ''
                if '..' in relpath.parts:
                    resolved = source.resolve()
                    if resolved.parent != (PREFIX/'bin').resolve():
                        raise ValueError('External data payload not supported: ' + rel)
                    if resolved.name in entries:
                        generated = True
                        reason = 'verified console entry point; installer regenerates portable script'
                    elif canonical(name) == 'ninja' and resolved.name == 'ninja':
                        mapped = dist[:-len('.dist-info')]+'.data/scripts/ninja'
                    else:
                        raise ValueError('External script is not a declared entry point: ' + rel)
                elif not source.resolve().is_relative_to(SOURCE.resolve()):
                    raise ValueError('Payload escapes site-packages: ' + rel)
                if not source.is_file() or source.is_symlink():
                    raise ValueError('Missing or nonregular installed payload: ' + rel)
                if relpath.name in {'INSTALLER', 'REQUESTED', 'direct_url.json'} and relpath.parent == PurePosixPath(dist):
                    generated = True
                    reason = 'verified installation metadata; omitted from rebuilt wheel'
                if rel.endswith('.pyc') or '__pycache__' in relpath.parts:
                    generated = True
                    reason = 'generated bytecode; omitted from rebuilt wheel'
                    if not expected and not size_text:
                        report['excluded_installation_files'].append({'path': rel, 'reason': reason})
                        continue
                if not expected or not size_text:
                    raise ValueError('Missing payload hash/size: ' + rel)
                algorithm, expected_hash = expected.split('=', 1)
                if algorithm not in {'sha256', 'sha384', 'sha512'}:
                    raise ValueError('Unsupported source hash: ' + algorithm)
                digest = hashlib.new(algorithm)
                sha256 = hashlib.sha256()
                count = 0
                member = zipfile.ZipInfo(mapped, date_time=(1980,1,1,0,0,0))
                member.create_system = 3
                member.external_attr = (stat.S_IMODE(source.stat().st_mode) | stat.S_IFREG) << 16
                member.compress_type = zipfile.ZIP_DEFLATED
                member._compresslevel = 1
                output = None if generated else archive.open(member, 'w', force_zip64=True)
                try:
                    with source.open('rb') as stream:
                        for block in iter(lambda: stream.read(4*1024*1024), b''):
                            digest.update(block); sha256.update(block); count += len(block)
                            if output: output.write(block)
                finally:
                    if output: output.close()
                actual = base64.urlsafe_b64encode(digest.digest()).decode().rstrip('=')
                if actual != expected_hash or count != int(size_text):
                    raise ValueError('Installed RECORD hash/size mismatch: ' + rel)
                report['record_verified_files'].append({'path': rel, 'wheel_path': mapped, 'size_bytes': count, 'sha256': sha256.hexdigest()})
                if generated:
                    report['excluded_installation_files'].append({'path': rel, 'reason': reason})
                else:
                    output_rows.append((mapped, 'sha256='+base64.urlsafe_b64encode(sha256.digest()).decode().rstrip('='), str(count)))
            required = {dist+'/METADATA', dist+'/WHEEL'}
            if not required.issubset({r[0] for r in output_rows}):
                raise ValueError('RECORD lacks mandatory metadata')
            content = io.StringIO(newline='')
            writer = csv.writer(content, lineterminator='\n')
            writer.writerows(output_rows)
            writer.writerow((dist+'/RECORD', '', ''))
            archive.writestr(dist+'/RECORD', content.getvalue())
        _, verified = verify_zip(partial)
        report.update(status='PASS', wheel_sha256=file_sha(partial), wheel_size_bytes=partial.stat().st_size,
                      rebuilt_record_verified_files=len(verified))
        partial.replace(target)
        return report
    except Exception as error:
        report['error'] = str(error)
        save_json(LOG/(canonical(name)+'-'+version+'.json'), report)
        partial.unlink(missing_ok=True)
        raise

def publish(reports):
    compact = [{k:v for k,v in r.items() if k not in {'record_verified_files','excluded_installation_files'}} for r in reports]
    save_json(OUT/'RUNTIME-SOURCES.json', {'scope': 'Offline runtime/build-tool wheels; rebuilt archives are not upstream wheel archive hashes',
        'target': 'CPython 3.10, Linux x86_64, glibc <=2.35 compatible declared tags', 'packages': compact})
    (OUT/'RUNTIME-SHA256SUMS').write_text(''.join(r['wheel_sha256']+'  '+r['filename']+'\n' for r in reports if r['status']=='PASS'))
    save_json(LOG/'summary.json', {'status': 'PASS' if len(reports)==len(PINS) and all(r['status']=='PASS' for r in reports) else 'INCOMPLETE',
        'expected_packages':len(PINS), 'verified_packages':sum(r['status']=='PASS' for r in reports),
        'failed_packages':[r for r in compact if r['status']!='PASS'],
        'wheel_bytes':sum(r.get('wheel_size_bytes',0) for r in reports), 'updated_at_utc':datetime.now(timezone.utc).isoformat()})

def main():
    cached=json.loads((LOG/'cache-inventory.json').read_text())
    reports=[]
    for name, version in PINS:
        print('START', name, version, flush=True)
        saved=LOG/(canonical(name)+'-'+version+'.json')
        if saved.exists():
            previous=json.loads(saved.read_text())
            if previous.get('status')=='PASS' and (OUT/previous['filename']).is_file() and file_sha(OUT/previous['filename'])==previous['wheel_sha256']:
                reports.append(previous); publish(reports); print('ALREADY VERIFIED',name,flush=True); continue
        candidates=[c for c in cached if canonical(c['name'])==canonical(name) and c['version']==version]
        report=None; cache_errors=[]
        for candidate in candidates:
            try:
                report=recover_cached(name,version,candidate); break
            except Exception as error:
                cache_errors.append({'source':candidate['source'],'error':str(error)})
        if report is None:
            try:
                report=rebuild_installed(name,version)
            except Exception as error:
                report={'status':'FAIL','package':name,'version':version,'error':str(error)}
        report['cache_attempt_errors']=cache_errors
        report['checked_at_utc']=datetime.now(timezone.utc).isoformat()
        save_json(saved,report)
        reports.append(report); publish(reports)
        print(report['status'], name, report.get('filename',report.get('error')), flush=True)

if __name__=='__main__':
    main()
