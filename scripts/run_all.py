"""Generate data, execute both notebooks from fresh kernels, and report timings."""
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from execute_notebook import execute

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.version_info[:2] != (3, 11):
        raise SystemExit('Use Python 3.11: py -3.11 -m venv .venv')
    os.chdir(ROOT)
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
    os.environ['MLFLOW_ENABLE_TELEMETRY'] = 'false'
    os.environ['DO_NOT_TRACK'] = 'true'
    start = perf_counter()
    timings = {}
    try:
        subprocess.run([sys.executable, 'scripts/make_data.py'], check=True)
        for name in ['01_fraud_detection.ipynb', '02_mlops.ipynb']:
            timings[name] = execute(ROOT / 'notebooks' / name)
        for name, seconds in timings.items():
            if seconds >= 60:
                raise RuntimeError(f'{name} exceeded the 60-second runtime target: {seconds:.2f}s')
        result = {'passed': True, 'notebooks': timings, 'total_seconds': perf_counter()-start,
                  'python': platform.python_version(), 'platform': platform.platform(),
                  'cpu': platform.processor(), 'logical_cpus': os.cpu_count()}
        (ROOT / 'models/run_summary.json').write_text(json.dumps(result, indent=2))
        print(f'PASS all: {result["total_seconds"]:.2f} seconds; notebooks saved with outputs.')
        print('MLflow UI: http://localhost:5000 (local process remains running).')
        return 0
    except Exception as error:
        print(f'FAIL: {type(error).__name__}: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
