from pathlib import Path
import concurrent.futures,subprocess,hashlib,json,time,sys
root=Path(__file__).resolve().parent
packages=json.loads((root/'packages.json').read_text())
parts=root/'parts';parts.mkdir(exist_ok=True)
chunk=128*1024
jobs=[]
for p in packages:
 for start in range(0,p['size'],chunk): jobs.append((p,start,min(start+chunk,p['size'])-1))
def download(job):
 p,start,end=job
 path=parts/(p['filename']+f'.{start:09d}')
 expected=end-start+1
 if path.exists() and path.stat().st_size==expected:return p['filename'],start,'cached'
 for attempt in range(4):
  tmp=path.with_suffix(path.suffix+'.tmp')
  r=subprocess.run(['curl','-q','--proxy','http://127.0.0.1:7897','--noproxy','','--connect-timeout','5','--max-time','35','--range',f'{start}-{end}','--output',str(tmp),'--write-out','%{http_code}','-fsSL',p['url']],capture_output=True,text=True)
  if r.returncode==0 and r.stdout=='206' and tmp.stat().st_size==expected:
   tmp.replace(path);return p['filename'],start,'downloaded'
  if r.returncode==0 and r.stdout=='200' and start==0 and tmp.stat().st_size==p['size']:
   data=tmp.read_bytes()
   if hashlib.sha256(data).hexdigest()==p['sha256']:
    for offset in range(0,len(data),chunk):(parts/(p['filename']+f'.{offset:09d}')).write_bytes(data[offset:offset+chunk])
    tmp.unlink();return p['filename'],start,'complete-response'
  time.sleep(1+attempt)
 raise RuntimeError(f'{p["filename"]} bytes {start}-{end}: rc={r.returncode} status={r.stdout}, {r.stderr.strip()}')
t0=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
 futures=[ex.submit(download,j) for j in jobs]
 for i,f in enumerate(concurrent.futures.as_completed(futures),1):
  name,start,status=f.result()
  if i%8==0 or i==len(jobs):print(f'Progress {i}/{len(jobs)} ranges in {time.monotonic()-t0:.1f}s: {name} {start} {status}',flush=True)
for p in packages:
 dest=root/p['filename']
 with dest.open('wb') as out:
  for start in range(0,p['size'],chunk):out.write((parts/(p['filename']+f'.{start:09d}')).read_bytes())
 assert dest.stat().st_size==p['size'],p['filename']
 got=hashlib.sha256(dest.read_bytes()).hexdigest()
 assert got==p['sha256'],(p['filename'],got,p['sha256'])
 print('SHA256 OK',p['filename'],p['size'],flush=True)
print('ALL NVIDIA DEBS VERIFIED',flush=True)
