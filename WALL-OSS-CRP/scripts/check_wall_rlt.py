#!/usr/bin/env python3
"""Run the local CPU implementation checks; never start GPU training or Ray."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    project=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='logs/rlt-cpu-check')
    args=parser.parse_args()
    output=Path(args.output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1',
             PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TOKENIZERS_PARALLELISM='false',
             PYTHONPATH=os.pathsep.join([str(project/'.cache/wall-x-rlt'),
                                        str(project/'.cache/rlinf-rlt'),str(project)]))
    checks=[('sources',[sys.executable,'scripts/prepare_wall_rlt_sources.py']),
            ('preflight',[sys.executable,'scripts/train_wall_rlt.py','--validate-only']),
            ('tests',[sys.executable,'-m','pytest','tests/rlt','-q',
                      f'--junitxml={output/"tests.xml"}']),
            ('stage2',[sys.executable,'-m','wall_rlt.stage2','--synthetic',
                       '--output',str(output/'stage2.json')])]
    report={'scope':'CPU implementation checks; no pretrained model forward, Ray or GPU training',
            'status':'RUNNING','checks':[]}
    for name,command in checks:
        logfile=output/f'{name}.log'
        print(f'Running {name}; log: {logfile}',flush=True)
        with logfile.open('w') as stream:
            try:
                result=subprocess.run(command,cwd=project,env=env,stdout=stream,
                                      stderr=subprocess.STDOUT,timeout=600)
                code=result.returncode
            except subprocess.TimeoutExpired:
                code=124
        report['checks'].append({'name':name,'exit_code':code,'command':command,
                                 'log':str(logfile),'status':'PASS' if code==0 else 'FAIL'})
        if code:
            report['status']='FAIL'
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            raise SystemExit(code)
    tree=ET.parse(output/'tests.xml')
    suites=tree.getroot().findall('testsuite')
    report['tests']={key:sum(int(s.get(key,0)) for s in suites)
                     for key in ('tests','failures','errors','skipped')}
    report['source_lock_sha256']=hashlib.sha256((project/'metadata/wall-rlt-sources.lock.json').read_bytes()).hexdigest()
    report['wall_patch_sha256']=hashlib.sha256((project/'patches/wall-x-rlt-prefix.patch').read_bytes()).hexdigest()
    report['status']='PASS'
    report['not_validated']=['pretrained WALL + RLT full-size load/forward','H100 BF16/FlashAttention/FSDP',
                             '100-step target-data training and distributed restore','Ubuntu 16GB inference',
                             'real replay/online RLT improvement']
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
