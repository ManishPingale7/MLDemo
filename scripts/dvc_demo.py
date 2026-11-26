"""Version and restore a small dataset copy; never modify the teaching dataset."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/transactions.csv'))
    parser.add_argument('--output', type=Path, default=Path('models/dvc_demo.json'))
    args = parser.parse_args()
    source = ROOT / args.data
    if not source.exists():
        raise SystemExit('Generate the full data with scripts/make_data.py first.')
    environment = {**os.environ, 'DVC_NO_ANALYTICS': '1', 'DO_NOT_TRACK': 'true'}
    runtime = ROOT / '.runtime'
    runtime.mkdir(exist_ok=True)
    sandbox = Path(tempfile.mkdtemp(prefix='dvc-demo-', dir=runtime))
    print(f'DVC sandbox: {sandbox}\nThe original dataset will not change.', flush=True)

    def run(*command):
        print('$ ' + ' '.join(command), flush=True)
        return subprocess.run(command, cwd=sandbox, env=environment, check=True,
                              text=True, capture_output=True).stdout.strip()

    def dvc(*arguments):
        output = run(sys.executable, '-m', 'dvc', *arguments)
        if output:
            print(output, flush=True)
        return output

    def commit(message):
        run('git', 'add', '.dvc', '.dvcignore', 'data/transactions.csv.dvc', 'data/.gitignore')
        run('git', '-c', 'user.name=DVC Classroom', '-c', 'user.email=classroom@example.invalid',
            'commit', '-m', message)
        return run('git', 'rev-parse', 'HEAD')

    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    run('git', 'init', '-b', 'main')
    run('git', 'config', 'core.autocrlf', 'false')
    dvc('init')
    dvc('config', 'core.analytics', 'false')
    dvc('remote', 'add', '-d', 'localstore', '../' + sandbox.name + '-storage')
    dataset = sandbox / 'data/transactions.csv'
    dataset.parent.mkdir()
    with source.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = [row for _, row in zip(range(200), reader)]

    def write_rows():
        with dataset.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)

    write_rows()
    first_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    dvc('add', 'data/transactions.csv')
    first_commit = commit('Dataset version one')
    dvc('push')
    # A one-rupee correction illustrates a new file version, not a model improvement.
    rows[0]['amount_inr'] = str(float(rows[0]['amount_inr']) + 1)
    write_rows()
    second_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    dvc('add', 'data/transactions.csv')
    commit('Dataset version two: correct one amount')
    dvc('push')
    diff = json.loads(dvc('diff', '--json', 'HEAD~1', 'HEAD'))
    run('git', 'restore', '--source=' + first_commit, '--', 'data/transactions.csv.dvc')
    dataset.unlink()

    def remove_readonly(function, path, _):
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        function(path)

    # Remove only this disposable demo's cache to prove restoration uses local storage.
    shutil.rmtree(sandbox / '.dvc/cache', onerror=remove_readonly)
    dvc('pull')
    restored_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert first_hash != second_hash and restored_hash == first_hash
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    report = {'passed': True, 'dvc_version': dvc('--version'), 'rows': len(rows),
              'first_sha256': first_hash, 'second_sha256': second_hash,
              'restored_sha256': restored_hash, 'original_unchanged': True,
              'local_cache_removed_before_pull': True, 'diff': diff, 'sandbox': str(sandbox)}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS DVC: two versions saved; version one restored byte-for-byte from local storage; '
          'original data unchanged.')


if __name__ == '__main__':
    main()
