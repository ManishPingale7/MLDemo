"""Real, local Git commands; no remote or global identity configuration."""
import subprocess
from fraud.config import ROOT


def git(*args, check=True):
    command = ['git', '-c', 'user.name=Fraud Classroom',
               '-c', 'user.email=classroom@example.invalid', *args]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=check)
    return result.stdout.strip() or result.stderr.strip()


def demonstrate():
    output = []

    def run(*args):
        output.append('$ git ' + ' '.join(args))
        output.append(git(*args))

    if not (ROOT / '.git').exists():
        run('init', '-b', 'main')
    run('status', '--short')
    run('add', '.')
    # An empty checkpoint keeps every rerun demonstrable without rewriting history.
    run('commit', '--allow-empty', '-m', 'Checkpoint classroom project')
    original = git('branch', '--show-current')
    if not original:
        raise RuntimeError('Checkout a branch before the Git demonstration.')
    count = git('rev-list', '--count', 'HEAD')
    branch = f'lesson/config-{count}'
    run('switch', '-c', branch)
    path = ROOT / 'config.yaml'
    lines = path.read_text().splitlines()
    lines = [f"demo_note: 'local classroom example {count}'" if line.startswith('demo_note:')
             else line for line in lines]
    path.write_text('\n'.join(lines) + '\n')
    run('add', 'config.yaml')
    run('commit', '-m', 'Clarify classroom configuration note')
    run('switch', original)
    run('merge', '--no-ff', branch, '-m', 'Merge configuration lesson')
    run('tag', f'lesson-v{count}')
    run('log', '--graph', '--oneline', '-7')
    return '\n'.join(output)
