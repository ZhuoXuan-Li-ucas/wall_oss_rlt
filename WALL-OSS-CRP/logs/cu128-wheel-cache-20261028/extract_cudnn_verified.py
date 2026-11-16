#!/usr/bin/env python3
from pathlib import Path,PurePosixPath
import hashlib,json,os,tarfile
ROOT=Path(__file__).resolve().parent
archive=Path('/media/pangk/My Passport/humaniod/lingbot-vla-v2-ubuntu24-cu128.tar.gz')
blob_hash='b5cb27424da4bb228eaf91dc3ffed3a5f92c52fa4c69eccdc56febc01d203450'
blob='blobs/sha256/'+blob_hash
prefix=PurePosixPath('opt/conda/envs/lingbotvla/lib/python3.12/site-packages')
stage=ROOT/'cudnn-source.partial';target=ROOT/'cudnn-source'
if stage.exists() or target.exists():raise RuntimeError('Use an empty staging location')
stage.mkdir();files=[];total=0
class HashReader:
 def __init__(self,src):self.src=src;self.h=hashlib.sha256();self.size=0
 def read(self,size=-1):
  b=self.src.read(size);self.h.update(b);self.size+=len(b);return b
 def close(self):pass
with tarfile.open(archive,'r|gz') as outer:
 found=False
 for member in outer:
  if member.name!=blob:continue
  found=True
  reader=HashReader(outer.extractfile(member))
  with tarfile.open(fileobj=reader,mode='r|*') as inner:
   for item in inner:
    p=PurePosixPath(item.name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe source layer path')
    try:rel=p.relative_to(prefix)
    except ValueError:continue
    name=str(rel)
    if not (name.startswith('nvidia/cudnn/') or name.startswith('nvidia_cudnn_cu12-9.10.2.21.dist-info/')):continue
    if '__pycache__' in rel.parts or name.endswith('.pyc'):continue
    if item.isdir():continue
    if not item.isfile():raise ValueError('Nonregular package member: '+name)
    out=stage/name
    if out.exists():raise ValueError('Duplicate package member: '+name)
    out.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with inner.extractfile(item) as src, out.open('wb') as dest:
     for b in iter(lambda:src.read(8*1024*1024),b''):dest.write(b);count+=len(b)
    if count!=item.size:raise ValueError('Incomplete source layer member')
    out.chmod(item.mode&0o777)
    files.append(name);total+=count
    print('EXTRACTED',name,count,flush=True)
  for _ in iter(lambda:reader.read(8*1024*1024),b''):pass
  if reader.size!=member.size or reader.h.hexdigest()!=blob_hash:raise ValueError('Source layer hash/size mismatch')
  print('SOURCE_LAYER_SHA256_OK',blob_hash,reader.size,flush=True)
  break
 if not found:raise ValueError('Required source layer absent')
if not (stage/'nvidia_cudnn_cu12-9.10.2.21.dist-info/RECORD').is_file():raise ValueError('Package RECORD absent')
stage.rename(target)
(ROOT/'cudnn-extraction.json').write_text(json.dumps({'status':'PASS','source_archive':str(archive),'layer_sha256':blob_hash,'selected_files':files,'selected_bytes':total,'scope':'selected cudnn package only; RECORD verification follows during wheel repack'},indent=2)+'\n')
print('CUDNN_SELECTED_EXTRACTION_PASS',len(files),total,flush=True)
