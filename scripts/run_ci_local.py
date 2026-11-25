"""Run the same checks as hosted CI, without accessing GitHub."""
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = [
    ['-m', 'ruff', 'check', '.'],
    ['-m', 'pytest', '-q'],
    ['scripts/make_data.py', '--preset', 'tiny'],
    ['scripts/train_quick.py'],
    ['scripts/run_gate.py', '--candidate', 'models/ci/candidate.joblib',
     '--champion', 'models/ci/reference.joblib', '--data', 'data/tiny.csv',
     '--output', 'models/ci/metrics_summary.json'],
    ['scripts/deploy_model.py', '--candidate', 'models/ci/candidate.joblib',
     '--baseline', 'models/ci/reference.joblib', '--data', 'data/tiny.csv',
     '--directory', 'models/ci/deployed'],
    ['scripts/smoke_api.py', '--directory', 'models/ci/deployed',
     '--data', 'data/tiny.csv', '--output', 'models/ci/api_smoke.json'],
]

if __name__ == '__main__':
    start = perf_counter()
    for args in COMMANDS:
        print('$ python ' + ' '.join(args), flush=True)
        result = subprocess.run([sys.executable, *args], cwd=ROOT)
        if result.returncode:
            print(f'FAIL: {args[0]} (exit {result.returncode})')
            raise SystemExit(result.returncode)
    print(f'PASS local CI: {perf_counter()-start:.2f} seconds')
