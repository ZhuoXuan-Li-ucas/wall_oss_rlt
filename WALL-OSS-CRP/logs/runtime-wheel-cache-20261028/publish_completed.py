#!/usr/bin/env python3
"""Publish verified official replacements and final runtime provenance manifests."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil

LOG=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('runtime_cache',LOG/'prepare_runtime_wheels.py')
cache=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)
STAGE=LOG/'official-wheel-downloads'
reports=[]
rejections=[]
for name,version in cache.PINS:
    basename=cache.canonical(name)+'-'+version+'.json'
    path=LOG/basename
    previous=json.loads(path.read_text())
    if previous['status']!='PASS':
        rejections.append(previous)
        downloaded=STAGE/basename
        if not downloaded.exists():raise RuntimeError('No verified replacement for '+name)
        report=json.loads(downloaded.read_text())
        if report['status']!='PASS':raise RuntimeError('Replacement failed for '+name)
        source=STAGE/report['filename']
        target=cache.OUT/report['filename']
        if cache.file_sha(source)!=report['wheel_sha256']:raise ValueError('Staged artifact changed: '+name)
        if target.exists():raise FileExistsError(target)
        partial=target.with_suffix(target.suffix+'.partial')
        shutil.copyfile(source,partial)
        if cache.file_sha(partial)!=report['wheel_sha256']:raise ValueError('Copied artifact changed: '+name)
        partial.replace(target)
    else:
        report=previous
    report['record_validation']='PASS'
    report['wheel_record_verified_file_count']=report.get('rebuilt_record_verified_files',len(report['record_verified_files']))
    report['source_record_verified_file_count']=len(report['record_verified_files'])
    report['checked_at_utc']=datetime.now(timezone.utc).isoformat()
    cache.save_json(path,report)
    reports.append(report)
cache.save_json(LOG/'repack-rejections.json',rejections)
cache.publish(reports)
cache.save_json(cache.OUT/'RUNTIME-RECORD-VERIFICATION.json',{'scope':'Per-file RECORD evidence for all runtime and packaging-tool wheels; source payload hashes precede rebuild transformations', 'packages':reports})
print(json.dumps({'status':'PASS','packages':len(reports),'wheel_bytes':sum(r['wheel_size_bytes'] for r in reports)},indent=2))
