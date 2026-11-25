"""Start the dashboard and FastAPI on localhost; no Node or frontend build needed."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def main():
    venv_python = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if venv_python.exists() and Path(sys.executable).absolute() != venv_python.absolute():
        return subprocess.call([str(venv_python), __file__, *sys.argv[1:]])
    if sys.version_info[:2] != (3, 11):
        raise SystemExit('Use Python 3.11 and install requirements.txt first.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '8765')))
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--strictPort', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault('MLFLOW_ENABLE_TELEMETRY', 'false')
    os.environ.setdefault('DO_NOT_TRACK', 'true')
    if not (ROOT / 'data/transactions.csv').exists() or not (ROOT / 'models/candidate.joblib').exists():
        subprocess.run([sys.executable, 'scripts/run_all.py'], check=True)
    if not (ROOT / 'models/deployed/active.json').exists():
        subprocess.run([sys.executable, 'scripts/deploy_model.py'], check=True)
    sys.path.insert(0, str(ROOT / 'src'))
    import uvicorn
    from fraud.api import create_app
    url = f'http://127.0.0.1:{args.port}'
    print(f'Fraud Lab dashboard: {url}\nAPI reference: {url}/docs\nPress Ctrl+C to stop.', flush=True)
    if not args.no_browser:
        def open_when_ready():
            for _ in range(100):
                try:
                    with urllib.request.urlopen(url + '/health', timeout=0.5) as response:
                        if response.status == 200:
                            webbrowser.open(url)
                            return
                except OSError:
                    time.sleep(0.1)
        threading.Thread(target=open_when_ready, daemon=True).start()
    uvicorn.run(create_app(), host=args.host, port=args.port, access_log=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
