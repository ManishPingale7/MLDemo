"""Execute a notebook in a fresh kernel with this Python interpreter."""
import os
from pathlib import Path
from time import perf_counter
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]


def execute(path):
    # Explicit interpreter avoids accidentally running a system Python kernel.
    import json
    import sys
    kernel_dir = ROOT / '.runtime' / 'kernels' / 'fraud-project'
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Fraud project Python 3.11', 'language': 'python'}))
    os.environ['JUPYTER_PATH'] = str(ROOT / '.runtime')
    notebook = nbformat.read(path, as_version=4)
    start = perf_counter()
    client = NotebookClient(notebook, timeout=180, kernel_name='fraud-project',
                            resources={'metadata': {'path': str(ROOT)}})
    client.km = client.create_kernel_manager()
    client.km.kernel_spec_manager = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    seconds = perf_counter() - start
    print(f'PASS {Path(path).name}: {seconds:.2f} seconds', flush=True)
    return seconds


if __name__ == '__main__':
    import sys
    execute(ROOT / 'notebooks' / sys.argv[1])
