#!/usr/bin/env python3
"""Read-only foreground monitor; the systemd build is independent of this process."""
from pathlib import Path
from datetime import datetime, timezone
import json
import re
import subprocess
import time

root = Path('/home/pangk/workspaces/WALL-OSS-CRP')
while True:
    job = json.loads((root / 'logs/docker-bundle-job-20261028.json').read_text())
    log = (root / 'logs/docker-pipeline-final-20261029.log').read_text()
    counters = re.findall(r'\[(\d+)/(\d+)\]', log)
    stages = re.findall(r'^#\d+ \[runtime[^\n]+', log, re.M)
    mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    unit = subprocess.run(['systemctl', 'is-active', 'walloss-bundle-20261029-final.service'],
                          text=True, capture_output=True, timeout=10).stdout.strip()
    state = {'time_utc': datetime.now(timezone.utc).isoformat(), 'job': job['status'],
             'service': unit, 'progress': '/'.join(counters[-1]) if counters else 'starting',
             'stage': stages[-1][:180] if stages else 'starting',
             'available_ram_gib': round(mem['MemAvailable']/1024**2, 2),
             'swap_used_gib': round((mem['SwapTotal']-mem['SwapFree'])/1024**2, 2)}
    print(json.dumps(state, ensure_ascii=False), flush=True)
    if job['status'] in ('BUNDLE_READY', 'FAILED') or unit not in ('active', 'activating'):
        break
    if 'exporting to image' in log:
        print('IMAGE_EXPORT_STAGE: native builds finished; temporary swap may now be removed.', flush=True)
        break
    time.sleep(45)
