from pathlib import Path
import nbformat
import yaml
from fraud.config import ROOT


def test_notebook_structure_and_short_cells():
    for name, count in [('01_fraud_detection.ipynb', 10), ('02_mlops.ipynb', 5)]:
        notebook = nbformat.read(ROOT / 'notebooks' / name, as_version=4)
        headings = [cell.source for cell in notebook.cells
                    if cell.cell_type == 'markdown' and cell.source.startswith('# ')]
        assert len(headings) == count
        assert all(text.startswith(f'# {i}. ') for i, text in enumerate(headings, 1))
        assert all(len(cell.source.splitlines()) <= 15 for cell in notebook.cells
                   if cell.cell_type == 'code')


def test_ci_workflow_commands_exist():
    # BaseLoader avoids YAML 1.1 interpreting the GitHub key `on` as a boolean.
    workflow = yaml.load((ROOT / '.github/workflows/ci.yml').read_text(), Loader=yaml.BaseLoader)
    assert workflow['on'] == ['push', 'pull_request']
    for step in workflow['jobs']['check']['steps']:
        if 'run' not in step:
            continue
        for word in step['run'].split():
            if word.startswith('scripts/') and word.endswith('.py'):
                assert (ROOT / Path(word)).is_file()
