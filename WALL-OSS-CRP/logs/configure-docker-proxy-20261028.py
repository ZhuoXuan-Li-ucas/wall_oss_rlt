import datetime,json,os,pathlib,shutil,subprocess,urllib.parse
p=pathlib.Path('/etc/docker/daemon.json')
d=json.loads(p.read_text()) if p.exists() else {}
values={}
for key,names in [('http-proxy',('http_proxy','HTTP_PROXY')),('https-proxy',('https_proxy','HTTPS_PROXY'))]:
 value=next((os.environ[n] for n in names if os.environ.get(n)),None)
 if not value:raise RuntimeError('No existing shell proxy available')
 u=urllib.parse.urlsplit(value)
 if u.hostname not in ('localhost','127.0.0.1') or u.username or u.password:raise RuntimeError('Expected the existing unauthenticated local proxy')
 values[key]=value
values['no-proxy']='localhost,127.0.0.1,::1'
if p.exists():shutil.copy2(p,str(p)+'.before-walloss-proxy-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
d['proxies']=values
p.write_text(json.dumps(d,indent=2)+'\n')
subprocess.run(['systemctl','restart','docker'],check=True)
print('Docker daemon now uses the existing local proxy; previous daemon.json backed up.')
