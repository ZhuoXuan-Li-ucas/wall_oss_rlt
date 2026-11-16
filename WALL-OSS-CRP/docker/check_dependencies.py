"""Record known declaration failures; reject new failures without hiding metadata."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
expected = set((Path(__file__).parent / 'expected-pip-check.txt').read_text().splitlines())
run = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
lines = set(run.stdout.strip().splitlines()) if run.returncode else set()
unexpected = sorted(lines - expected)
accepted = (run.returncode in (0, 1) and not unexpected and not run.stderr.strip()
            and (run.returncode == 0 or bool(lines)))
report = {
    'pip_check_status': 'PASS' if run.returncode == 0 else 'FAIL',
    'returncode': run.returncode, 'issues': sorted(lines), 'stderr': run.stderr,
    'unexpected': unexpected, 'known_declaration_exceptions_only': accepted,
    'scope': 'runtime exception inventory; not complete LeRobot or training validation',
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
sys.exit(0 if accepted else 1)
