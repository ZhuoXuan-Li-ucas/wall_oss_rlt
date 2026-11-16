#!/usr/bin/env python3
"""Capture task-relevant host information without environment secrets."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
commands = [
    ["uname", "-a"], ["cat", "/etc/os-release"], ["lscpu"],
    ["nvidia-smi"],
    ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.free,compute_cap", "--format=csv"],
    ["nvcc", "--version"], ["gcc", "--version"], ["free", "-h"],
    ["df", "-h", str(root)], [str(root / "env/bin/python"), "--version"],
]
lines = ["# Host environment", "", "Captured: " + datetime.now(timezone.utc).isoformat(), ""]
for command in commands:
    result = subprocess.run(command, capture_output=True, text=True)
    lines += ["```text", "$ " + " ".join(command), result.stdout.rstrip(),
              result.stderr.rstrip(), f"exit_code={result.returncode}", "```", ""]
(root / "environment.host.md").write_text("\n".join(lines))
print(root / "environment.host.md")
